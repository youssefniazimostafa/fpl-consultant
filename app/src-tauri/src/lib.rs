//! FPL Consultant desktop shell -- backend.
//!
//! Two jobs:
//! 1. Load the most recent *local* latest.json (produced by a local refresh
//!    in the engine copy); the front end falls back to the published one.
//! 2. Refresh: run the full Python pipeline in an isolated working copy of
//!    the repo (~/Library/Application Support/FPL Consultant/engine), never
//!    in the dev repo, streaming the "=== stage ===" lines to the UI.
//!    The engine copy is only ever updated *from* GitHub; the app never
//!    commits or pushes (its push URL is deliberately set to an invalid value).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

const REPO_URL: &str = "https://github.com/youssefniazimostafa/fpl-consultant.git";
const BRANCH: &str = "main";
const DISABLED_PUSH_URL: &str = "DISABLED--the-FPL-Consultant-app-never-pushes";
const LOG_TAIL_LINES: usize = 40;

/// Only one refresh at a time.
#[derive(Default)]
struct RefreshLock(AtomicBool);

/// Clears the lock however the refresh thread exits (success, error, panic).
struct LockGuard<'a>(&'a AtomicBool);
impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[derive(Serialize, Clone)]
struct LocalData {
    json: String,
    path: String,
    modified_ms: u128,
}

#[derive(Serialize, Clone)]
struct Progress {
    stage: String,
    line: Option<String>,
}

#[derive(Serialize, Clone)]
struct Done {
    ok: bool,
    error: Option<String>,
    seconds: f64,
}

fn support_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Application Support/FPL Consultant"))
}

fn engine_dir() -> Result<PathBuf, String> {
    Ok(support_dir()?.join("engine"))
}

/// Plain-text diagnostics log (~/Library/Application Support/FPL Consultant/
/// app.log), started fresh on every launch: what was loaded, refresh stages
/// and outcomes. Best-effort -- logging never fails the app.
fn app_log(message: &str) {
    let Ok(dir) = support_dir() else { return };
    let _ = std::fs::create_dir_all(&dir);
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("app.log")) {
        let _ = writeln!(f, "[{secs}] {message}");
    }
}

fn reset_log() {
    if let Ok(dir) = support_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("app.log"), "");
    }
    app_log(&format!("FPL Consultant {} started", env!("CARGO_PKG_VERSION")));
}

#[tauri::command]
fn ui_log(message: String) {
    app_log(&format!("ui: {message}"));
}

/// Launching with FPL_CONSULTANT_AUTOREFRESH=1 makes the UI start one
/// refresh after its first data load -- used to test the whole flow without
/// clicking (e.g. `open --env FPL_CONSULTANT_AUTOREFRESH=1 "FPL Consultant.app"`).
/// Screenshot aid: FPL_CONSULTANT_TAB=league opens that tab, and
/// FPL_CONSULTANT_RIVAL=<manager id> opens that manager's team on the
/// League page; FPL_CONSULTANT_SCROLL=1 scrolls the page to the bottom.
#[tauri::command]
fn start_view() -> (Option<String>, Option<i64>, bool) {
    (
        std::env::var("FPL_CONSULTANT_TAB").ok(),
        std::env::var("FPL_CONSULTANT_RIVAL").ok().and_then(|v| v.parse().ok()),
        std::env::var("FPL_CONSULTANT_SCROLL").as_deref() == Ok("1"),
    )
}

#[tauri::command]
fn autorefresh_requested() -> u8 {
    // "1": one refresh; "2": two at once (test aid for the double-refresh guard)
    match std::env::var("FPL_CONSULTANT_AUTOREFRESH").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    }
}

#[tauri::command]
fn load_local_data() -> Result<Option<LocalData>, String> {
    let path = engine_dir()?.join("site/data/latest.json");
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str::<serde_json::Value>(&json)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    let modified_ms = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Ok(Some(LocalData { json, path: path.display().to_string(), modified_ms }))
}

// ---------------------------------------------------------------------------
// offline support
// ---------------------------------------------------------------------------

/// Last successfully fetched *published* latest.json, so the app still has
/// data offline even before the first local refresh.
fn published_cache_path() -> Result<PathBuf, String> {
    Ok(support_dir()?.join("published-latest.json"))
}

#[tauri::command]
fn save_published_cache(json: String) -> Result<(), String> {
    let dir = support_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = published_cache_path()?;
    std::fs::write(&path, json).map_err(|e| format!("writing {}: {e}", path.display()))
}

#[tauri::command]
fn load_published_cache() -> Result<Option<LocalData>, String> {
    let path = published_cache_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let modified_ms = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Ok(Some(LocalData { json, path: path.display().to_string(), modified_ms }))
}

/// A club's shirt from the FPL site's image server, cached on disk under
/// ~/Library/Application Support/FPL Consultant/shirts/ (personal use only)
/// and returned as a data: URL. Errors (offline with no cached copy, unknown
/// club) make the UI fall back to a drawn shirt in the club's colours.
#[tauri::command]
async fn shirt_image(team_code: u32, goalkeeper: bool) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || shirt_image_blocking(team_code, goalkeeper))
        .await
        .map_err(|e| e.to_string())?
}

/// Test aid: FPL_CONSULTANT_SIMULATE_OFFLINE=1 makes the app behave as if
/// there were no network (no published-data fetch, no shirt downloads), to
/// check it still works from its on-disk caches.
fn simulate_offline() -> bool {
    std::env::var("FPL_CONSULTANT_SIMULATE_OFFLINE").as_deref() == Ok("1")
}

#[tauri::command]
fn offline_simulated() -> bool {
    simulate_offline()
}

fn shirt_image_blocking(team_code: u32, goalkeeper: bool) -> Result<String, String> {
    use base64::Engine;
    let name = format!("shirt_{team_code}{}-110.png", if goalkeeper { "_1" } else { "" });
    let dir = support_dir()?.join("shirts");
    let path = dir.join(&name);
    if !path.exists() && simulate_offline() {
        return Err(format!("offline (simulated) and {name} isn't cached"));
    }
    if !path.exists() {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let url = format!("https://fantasy.premierleague.com/dist/img/shirts/standard/{name}");
        let tmp = dir.join(format!("{name}.{}.part", std::process::id()));
        let status = Command::new("/usr/bin/curl")
            .args(["-fsSL", "--max-time", "15", "-o"])
            .arg(&tmp)
            .arg(&url)
            .stdin(Stdio::null())
            .status()
            .map_err(|e| format!("couldn't run curl: {e}"))?;
        if !status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("couldn't download {url}"));
        }
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if !bytes.starts_with(b"\x89PNG") {
        let _ = std::fs::remove_file(&path);
        return Err(format!("{name} isn't a PNG"));
    }
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
}

// ---------------------------------------------------------------------------
// betting-odds key (Settings): stored ONLY in the macOS Keychain
// ---------------------------------------------------------------------------

const KEYCHAIN_SERVICE: &str = "FPL Consultant";
const KEYCHAIN_ODDS_ACCOUNT: &str = "odds-api-key";

fn odds_entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYCHAIN_SERVICE, KEYCHAIN_ODDS_ACCOUNT).map_err(|e| format!("Keychain: {e}"))
}

fn keychain_odds_key() -> Option<String> {
    odds_entry().ok()?.get_password().ok().filter(|k| !k.trim().is_empty())
}

/// Saves the key in the Keychain. The key is never logged or returned.
#[tauri::command]
fn set_odds_key(key: String) -> Result<(), String> {
    let key = key.trim();
    if key.is_empty() {
        return Err("The key is empty.".into());
    }
    odds_entry()?.set_password(key).map_err(|e| format!("Couldn't save to the Keychain: {e}"))?;
    app_log("settings: odds key saved to the Keychain");
    Ok(())
}

#[tauri::command]
fn clear_odds_key() -> Result<(), String> {
    match odds_entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => {
            app_log("settings: odds key removed from the Keychain");
            Ok(())
        }
        Err(e) => Err(format!("Couldn't remove it from the Keychain: {e}")),
    }
}

/// Only whether a key is saved -- never the key itself.
#[tauri::command]
fn odds_key_saved() -> bool {
    keychain_odds_key().is_some()
}

#[tauri::command]
fn is_refreshing(lock: State<'_, RefreshLock>) -> bool {
    lock.0.load(Ordering::SeqCst)
}

/// Cross-process lock (an OS advisory lock on a file next to the engine
/// copy), so a dev build and the installed app can't refresh the same
/// engine copy at once. Released automatically when the file is closed --
/// including if the app crashes -- so it can't go stale.
fn acquire_engine_lock() -> Result<std::fs::File, String> {
    let dir = support_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let path = dir.join("refresh.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| format!("opening {}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err("Another FPL Consultant window is already refreshing. Wait for it to finish, then try again.".into())
        }
        Err(std::fs::TryLockError::Error(e)) => Err(format!("locking {}: {e}", path.display())),
    }
}

#[tauri::command]
fn start_refresh(app: AppHandle, lock: State<'_, RefreshLock>) -> Result<(), String> {
    if lock.0.swap(true, Ordering::SeqCst) {
        return Err("A refresh is already running.".into());
    }
    let engine_lock = match acquire_engine_lock() {
        Ok(f) => f,
        Err(e) => {
            lock.0.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    thread::spawn(move || {
        let lock = app.state::<RefreshLock>();
        let _guard = LockGuard(&lock.0);
        let _engine_lock = engine_lock; // held until this thread ends
        let started = Instant::now();
        app_log("refresh: started");
        let result = run_refresh(&app);
        match &result {
            Ok(()) => app_log(&format!("refresh: OK in {:.0}s", started.elapsed().as_secs_f64())),
            Err(e) => app_log(&format!("refresh: FAILED after {:.0}s: {e}", started.elapsed().as_secs_f64())),
        }
        let _ = app.emit(
            "refresh-done",
            Done { ok: result.is_ok(), error: result.err(), seconds: started.elapsed().as_secs_f64() },
        );
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// refresh steps
// ---------------------------------------------------------------------------

fn emit_stage(app: &AppHandle, stage: &str) {
    app_log(&format!("refresh: stage {stage}"));
    let _ = app.emit("refresh-progress", Progress { stage: stage.to_string(), line: None });
}

/// A Finder-launched app gets a minimal PATH and none of the shell's
/// exported variables, so ask the user's login shell for PATH and
/// ODDS_API_KEY (an already-set process variable wins, e.g. under
/// `npm run tauri dev` started from a terminal).
struct ShellEnv {
    path: String,
    odds_api_key: Option<String>,
}

fn shell_env() -> ShellEnv {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    let script = r#"printf '__FPL_PATH__=%s\n' "$PATH"; printf '__FPL_ODDS__=%s\n' "${ODDS_API_KEY-}""#;
    let mut login_path = String::new();
    let mut login_odds = String::new();
    if let Ok(out) = Command::new(&shell).args(["-ilc", script]).stdin(Stdio::null()).output() {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if let Some(v) = line.strip_prefix("__FPL_PATH__=") {
                login_path = v.to_string();
            } else if let Some(v) = line.strip_prefix("__FPL_ODDS__=") {
                login_odds = v.to_string();
            }
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let mut dirs: Vec<String> = Vec::new();
    for part in [login_path, std::env::var("PATH").unwrap_or_default()] {
        dirs.extend(part.split(':').filter(|s| !s.is_empty()).map(String::from));
    }
    // common install locations, in case neither PATH has them
    for extra in [
        format!("{home}/.local/bin"),
        format!("{home}/.cargo/bin"),
        "/opt/homebrew/bin".into(),
        "/usr/local/bin".into(),
        "/usr/bin".into(),
        "/bin".into(),
    ] {
        dirs.push(extra);
    }
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));

    // Order: the app's own setting (macOS Keychain), then the process
    // environment (e.g. `npm run tauri dev` from a terminal), then the
    // login shell. Never logged or written anywhere.
    let odds_api_key = keychain_odds_key()
        .or_else(|| std::env::var("ODDS_API_KEY").ok().filter(|v| !v.is_empty()))
        .or_else(|| Some(login_odds).filter(|v| !v.is_empty()));
    ShellEnv { path: dirs.join(":"), odds_api_key }
}

fn find_program(name: &str, path: &str) -> Result<PathBuf, String> {
    path.split(':')
        .map(|d| Path::new(d).join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("couldn't find `{name}` on PATH or in the usual install locations"))
}

fn git(git: &Path, cwd: &Path, args: &[&str], env_path: &str) -> Result<(), String> {
    let out = Command::new(git)
        .args(args)
        .current_dir(cwd)
        .env("PATH", env_path)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("couldn't run git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!("git {} failed:\n{}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(())
}

fn run_refresh(app: &AppHandle) -> Result<(), String> {
    emit_stage(app, "preparing");
    let env = shell_env();
    let git_bin = find_program("git", &env.path)?;
    let uv_bin = find_program("uv", &env.path)?;
    let engine = engine_dir()?;
    let parent = engine.parent().ok_or("bad engine path")?.to_path_buf();
    std::fs::create_dir_all(&parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;

    if !engine.join(".git").exists() {
        if engine.exists() {
            // half-finished earlier clone; the folder is app-owned
            std::fs::remove_dir_all(&engine).map_err(|e| format!("clearing {}: {e}", engine.display()))?;
        }
        emit_stage(app, "cloning engine copy (first run)");
        git(&git_bin, &parent, &["clone", "--branch", BRANCH, REPO_URL, "engine"], &env.path)?;
    }
    // Belt and braces: make pushing from the engine copy impossible.
    git(&git_bin, &engine, &["remote", "set-url", "--push", "origin", DISABLED_PUSH_URL], &env.path)?;

    // "git pull", made safe for a copy the pipeline writes into: the
    // pipeline modifies tracked data files (prediction snapshots, accuracy
    // log), which would make a plain pull refuse to merge on the second
    // refresh. The engine never keeps local commits, so match GitHub
    // exactly; untracked/ignored caches (data/raw, data/tables, .venv) stay.
    emit_stage(app, "updating engine copy (git pull)");
    git(&git_bin, &engine, &["fetch", "origin", BRANCH], &env.path)?;
    git(&git_bin, &engine, &["reset", "--hard", &format!("origin/{BRANCH}")], &env.path)?;

    emit_stage(app, "starting pipeline");
    let mut cmd = Command::new(&uv_bin);
    // Test aid: FPL_CONSULTANT_TEST_FAIL=1 runs a stand-in pipeline that
    // fails after its first stage, to exercise the error path end to end.
    if std::env::var("FPL_CONSULTANT_TEST_FAIL").as_deref() == Ok("1") {
        cmd.args(["run", "python", "-c", "import sys; print('=== fetch ==='); print('test failure (FPL_CONSULTANT_TEST_FAIL)'); sys.exit(3)"]);
    } else {
        cmd.args(["run", "python", "-m", "fpl.pipeline"]);
    }
    cmd
        .current_dir(&engine)
        .env("PATH", &env.path)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    app_log(&format!(
        "refresh: uv={} ODDS_API_KEY {}",
        uv_bin.display(),
        if env.odds_api_key.is_some() { "passed through" } else { "not set (pipeline will skip odds)" }
    ));
    match &env.odds_api_key {
        Some(key) => cmd.env("ODDS_API_KEY", key),
        None => cmd.env_remove("ODDS_API_KEY"),
    };
    let mut child = cmd.spawn().map_err(|e| format!("couldn't start the pipeline: {e}"))?;

    // merge stdout + stderr, line by line
    let (tx, rx) = mpsc::channel::<String>();
    let streams: Vec<Box<dyn Read + Send>> = vec![
        Box::new(child.stdout.take().ok_or("no stdout")?),
        Box::new(child.stderr.take().ok_or("no stderr")?),
    ];
    for stream in streams {
        let tx = tx.clone();
        thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);

    let mut tail: VecDeque<String> = VecDeque::with_capacity(LOG_TAIL_LINES);
    let mut stage = String::from("starting pipeline");
    let mut completed = false;
    for line in rx {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("=== ").and_then(|s| s.strip_suffix(" ===")) {
            stage = name.to_string();
            app_log(&format!("refresh: stage {stage}"));
        }
        if trimmed == "Pipeline complete." {
            completed = true;
        }
        if tail.len() == LOG_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line.clone());
        let _ = app.emit("refresh-progress", Progress { stage: stage.clone(), line: Some(line) });
    }
    let status = child.wait().map_err(|e| format!("waiting for the pipeline: {e}"))?;
    if !status.success() || !completed {
        let code = status.code().map(|c| format!("exit code {c}")).unwrap_or_else(|| "killed by a signal".into());
        let tail_text = tail.into_iter().collect::<Vec<_>>().join("\n");
        return Err(format!("Pipeline failed during \"{stage}\" ({code}).\nLast output:\n{tail_text}"));
    }
    if load_local_data()?.is_none() {
        return Err("Pipeline finished but site/data/latest.json wasn't written.".into());
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    reset_log();
    tauri::Builder::default()
        .manage(RefreshLock::default())
        .setup(|app| {
            // Screenshot aid: FPL_CONSULTANT_WINDOW=390,844 opens the window at
            // that logical size, top-left at (40, 60) on the main screen.
            if let Ok(spec) = std::env::var("FPL_CONSULTANT_WINDOW") {
                let dims: Vec<f64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if let (Some(window), [w, h]) = (app.get_webview_window("main"), dims.as_slice()) {
                    let _ = window.set_size(tauri::LogicalSize::new(*w, *h));
                    let _ = window.set_position(tauri::LogicalPosition::new(40.0, 60.0));
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_local_data,
            save_published_cache,
            load_published_cache,
            shirt_image,
            set_odds_key,
            clear_odds_key,
            odds_key_saved,
            offline_simulated,
            start_refresh,
            is_refreshing,
            ui_log,
            autorefresh_requested,
            start_view
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Save, read back, remove -- against the real macOS Keychain.
    #[test]
    fn odds_key_roundtrip_in_keychain() {
        let fake = "FPLTEST-keychain-roundtrip-not-a-real-key-7731";
        let had_key = odds_key_saved();
        assert!(!had_key, "a real key is saved; refusing to overwrite it in a test");
        set_odds_key(format!("  {fake}  ")).unwrap();
        assert!(odds_key_saved());
        assert_eq!(keychain_odds_key().as_deref(), Some(fake));
        clear_odds_key().unwrap();
        assert!(!odds_key_saved());
        assert!(set_odds_key("   ".into()).is_err());
    }
}
