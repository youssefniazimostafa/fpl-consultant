//! FPL Consultant -- Tauri backend.
//!
//! macOS ("desktop" mode): everything in desktop.rs -- local refresh of the
//! Python pipeline, the Live consultant chat, API keys in the Keychain.
//!
//! iOS ("viewer" mode): none of that is compiled in. The app is a plain
//! webview over the viewer front end, which reads the published
//! latest.json from GitHub Pages; the only command is the launch-view
//! screenshot aid below (it reads environment variables, nothing else).

#[cfg(desktop)]
mod desktop;

#[cfg(mobile)]
#[tauri::command]
fn start_view() -> (Option<String>, Option<i64>, bool, Option<String>) {
    // Simulator screenshot aid: `SIMCTL_CHILD_FPL_CONSULTANT_TAB=league
    // xcrun simctl launch ...` opens that tab.
    (std::env::var("FPL_CONSULTANT_TAB").ok(), None, false, None)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(desktop)]
    desktop::run();

    #[cfg(mobile)]
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![start_view])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
