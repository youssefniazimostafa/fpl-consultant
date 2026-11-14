import { useEffect, useState } from "react";
import type { Latest } from "../data";
import { Card, Pill } from "../components/Card";
import { Bench, Pitch } from "../components/Pitch";
import { Shirt } from "../components/Shirt";
import { defaultRival, readLeague, type LeaguePageData, type OwnedPlayer, type RivalInArea } from "../model/league";
import { PHONE_BREAKPOINT } from "../theme/theme";

// Your leagues page: everything comes from latest.json's `league_page`
// (fpl/league_page.py). Rivals' teams are their last confirmed ones --
// FPL hides upcoming picks until the deadline.

function usePhone(): boolean {
  const query = `(max-width: ${PHONE_BREAKPOINT - 1}px)`;
  const [phone, setPhone] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setPhone(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return phone;
}

const pct = (p: number | null) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function HeadlineCards({ l }: { l: LeaguePageData }) {
  const y = l.you;
  return (
    <div className="headline-cards">
      <div className="stat-card dark">
        <div className="stat-big">
          {y.rankText} <span className="stat-of">of {y.managers}</span>
        </div>
        <div className="stat-sub">
          {y.rank === 1
            ? y.lead != null
              ? `${plural(y.lead, "point")} clear at the top`
              : "Top of the league"
            : `${plural(y.behindFirst, "point")} behind 1st`}
        </div>
      </div>
      <div className="stat-card bright">
        <div className="stat-label">Your chances</div>
        <div className="stat-line">
          <b>{pct(l.chances.thisGameweek)}</b> <span>to win GW{l.chances.gameweek}</span>
        </div>
        <div className="stat-line">
          <b>{pct(l.chances.topBy)}</b> <span>to be top by GW{l.chances.topByGameweek}</span>
        </div>
      </div>
    </div>
  );
}

function Standings({ l, selected, onPick }: { l: LeaguePageData; selected: number | null; onPick: (id: number) => void }) {
  return (
    <Card title="Standings" aside={`after GW${l.gameweek} · tap a team`} className="standings-card">
      <div className="standings">
        <div className="st-row st-head">
          <span />
          <span>Team</span>
          <span className="num">GW</span>
          <span className="num">Total</span>
        </div>
        {l.standings.map((s) => (
          <button
            key={s.managerId}
            className={`st-row${s.isYou ? " you" : ""}${s.managerId === selected ? " selected" : ""}`}
            onClick={() => onPick(s.managerId)}
          >
            <span className={`st-dot${s.inYourArea ? " area" : ""}${s.isYou ? " me" : ""}`} title={s.inYourArea ? "In your area" : undefined} />
            <span className="st-team">
              <span className="st-rank">{s.rank}</span>
              <span className="st-name">
                <b>{s.team}</b>
                <small>
                  {s.manager}
                  {s.isYou ? " · you" : ""}
                  {s.inYourArea && (
                    <span className="area-tag">
                      <span className="tag-long">in your area</span>
                      <span className="tag-short">near you</span>
                    </span>
                  )}
                </small>
              </span>
            </span>
            <span className="num">{s.gwPoints}</span>
            <span className="num total">{s.total}</span>
          </button>
        ))}
      </div>
      <p className="note muted">"In your area": rivals within about a gameweek's points of you.</p>
    </Card>
  );
}

function RivalTeamView({ l, managerId, onBack }: { l: LeaguePageData; managerId: number; onBack?: () => void }) {
  const s = l.standings.find((x) => x.managerId === managerId);
  const t = l.teams[String(managerId)];
  if (!s || !t) return null;
  const facts = [
    t.captain ? `Captain ${t.captain}` : null,
    t.chip ? `Played ${t.chip}` : "No chip played",
    t.freeTransfers != null ? plural(t.freeTransfers, "free transfer") : null,
  ].filter(Boolean);
  return (
    <>
    {onBack && (
      <button className="back back-row" onClick={onBack} aria-label="Back to standings">
        ‹ Back to standings
      </button>
    )}
    <Card
      title={s.isYou ? "Your team" : s.team}
      aside={`${s.rank}${s.rank === 1 ? "st" : s.rank === 2 ? "nd" : s.rank === 3 ? "rd" : "th"} · ${s.total} pts`}
      className="rival-team"
    >
      <p className="muted small">
        Last confirmed team (GW{t.gameweek}). FPL hides next week's picks until the deadline.
      </p>
      <div className="facts">
        {facts.map((f) => (
          <Pill key={f} tone="purple">
            {f}
          </Pill>
        ))}
      </div>
      <Pitch players={t.xi} legend={`GW${t.gameweek} points under each name`} />
      <Bench players={t.bench} />
      {!s.isYou && (t.likelyIn || t.likelyOut) && (
        <p className="note">
          Likely next transfer: {t.likelyOut ? `${t.likelyOut} out` : "someone out"}, {t.likelyIn ? `${t.likelyIn} in` : "unclear who in"}.
        </p>
      )}
    </Card>
    </>
  );
}

function Names({ names }: { names: string[] }) {
  return names.length ? <span>{names.join(", ")}</span> : <span className="muted">nobody</span>;
}

function ChipIcons({ chips }: { chips: RivalInArea["chips"] }) {
  return (
    <span className="chip-icons">
      {chips.map((c) => (
        <span
          key={c.chip}
          className={`chip-ic ${c.available ? "avail" : "used"}`}
          title={c.available ? `${c.label}: still available` : `${c.label}: used${c.usedGameweek ? ` in GW${c.usedGameweek}` : ""}`}
        >
          {c.short}
          {!c.available && c.usedGameweek ? <small> · GW{c.usedGameweek}</small> : null}
        </span>
      ))}
    </span>
  );
}

const gapText = (diff: number) => (diff === 0 ? "level" : diff > 0 ? `${diff} ahead` : `${-diff} behind`);

function likelyMove(r: RivalInArea): string {
  if (r.likelyOut && r.likelyIn) return `${r.likelyOut} → ${r.likelyIn}`;
  if (r.likelyIn) return `→ ${r.likelyIn}`;
  if (r.likelyOut) return `${r.likelyOut} out`;
  return r.freeTransfers ? "Save the transfer" : "No free transfer";
}

function RivalRow({ r, open, onToggle, phone, l }: { r: RivalInArea; open: boolean; onToggle: () => void; phone: boolean; l: LeaguePageData }) {
  return (
    <div className={`ra-row${open ? " open" : ""}`}>
      <button className="ra-line" onClick={onToggle} aria-expanded={open}>
        <span className="ra-who">
          <b>{r.team}</b>
          <small>{r.manager}</small>
        </span>
        <span className="ra-rank">{r.rank}{r.rank === 1 ? "st" : r.rank === 2 ? "nd" : r.rank === 3 ? "rd" : "th"}</span>
        <span className={`ra-gap ${r.pointsVsYou > 0 ? "ahead" : r.pointsVsYou < 0 ? "behind" : ""}`}>{gapText(r.pointsVsYou)}</span>
        <span className="ra-move" title={[r.likelyOutReason, r.likelyInReason].filter(Boolean).join(" · ") || undefined}>
          {likelyMove(r)}
        </span>
        <span className="ra-ft" title="Free transfers">{r.freeTransfers ?? "–"} FT</span>
        <ChipIcons chips={r.chips} />
        <span className="ra-caret" aria-hidden>{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="ra-detail">
          <dl className="ra-facts">
            <dt>They have, you don't</dt>
            <dd><Names names={r.theyHave} /></dd>
            <dt>You have, they don't</dt>
            <dd><Names names={r.youHave} /></dd>
            {(r.likelyInReason || r.likelyOutReason) && (
              <>
                <dt>Why that move</dt>
                <dd>{[r.likelyOutReason, r.likelyInReason].filter((x) => x && x !== "no clear signal").filter((x, i, a) => a.indexOf(x) === i).join("; ") || "no clear signal"}</dd>
              </>
            )}
          </dl>
          {phone && <RivalPitch l={l} managerId={r.managerId} />}
        </div>
      )}
    </div>
  );
}

function RivalPitch({ l, managerId }: { l: LeaguePageData; managerId: number }) {
  const s = l.standings.find((x) => x.managerId === managerId);
  const t = l.teams[String(managerId)];
  if (!s || !t) return null;
  return (
    <div className="ra-pitch">
      <div className="ra-pitch-head">
        <b>{s.team}</b>
        <span className="muted small">
          GW{t.gameweek} team{t.captain ? ` · captain ${t.captain}` : ""}{t.chip ? ` · played ${t.chip}` : ""}
        </span>
      </div>
      <Pitch players={t.xi} legend={`GW${t.gameweek} points under each name`} />
      <Bench players={t.bench} />
    </div>
  );
}

function RivalsInArea({ l, phone }: { l: LeaguePageData; phone: boolean }) {
  const [open, setOpen] = useState<number | null>(() => (phone ? null : l.rivalsInArea[0]?.managerId ?? null));
  const shown = open ?? l.rivalsInArea[0]?.managerId ?? null;
  return (
    <Card title="Rivals in your area" aside={`${plural(l.rivalsInArea.length, "rival")} · tap one to compare`} className="rivals-card">
      {l.rivalsInArea.length ? (
        <div className="ra-layout">
          <div className="ra-list">
            <div className="ra-line ra-headrow" aria-hidden>
              <span>Rival</span><span>Rank</span><span>Gap</span><span>Likely transfer</span><span>FT</span><span>Chips</span><span />
            </div>
            {l.rivalsInArea.map((r) => (
              <RivalRow
                key={r.managerId}
                r={r}
                l={l}
                phone={phone}
                open={open === r.managerId}
                onToggle={() => setOpen(open === r.managerId ? null : r.managerId)}
              />
            ))}
            <p className="note muted ra-how">
              How we guess each rival's next move: a rival who usually saves transfers is expected to save. Otherwise their
              injured or doubtful player (or, if none, their weakest starter by expected points) goes out, replaced in the
              same position within their budget: by a player who just hauled if they usually chase hauls (most of your
              rivals do), or by the best expected points otherwise. Chips: <span className="chip-ic avail">FH</span> still
              available, <span className="chip-ic used">FH<small> · GW6</small></span> used.
            </p>
          </div>
          {!phone && shown != null && <RivalPitch l={l} managerId={shown} />}
        </div>
      ) : (
        <p className="muted">No rivals within a gameweek's points of you right now.</p>
      )}
    </Card>
  );
}

function PlayerList({ players, rivals, mode }: { players: OwnedPlayer[]; rivals: number; mode: "owned" | "notOwned" | "diff" }) {
  return (
    <div className="plist">
      {players.map((p) => (
        <div key={p.id} className="plist-row">
          <Shirt teamCode={p.teamCode} teamShort={p.teamShort} goalkeeper={p.position === "GKP"} size={30} title={p.name} />
          <span className="pl-name">
            <b>{p.name}</b>
            <small>
              {p.teamShort} · {p.position}
            </small>
          </span>
          <span className="pl-own">
            {p.rivalsOwning === 0 ? "No rivals own him" : `${p.rivalsOwning} of ${rivals} rivals own him`}
            {mode !== "diff" && p.rivalsCaptaining > 0 ? `, ${p.rivalsCaptaining} captained` : ""}
            {mode !== "diff" && p.rivalsBenching > 0 ? `, ${p.rivalsBenching} benched him` : ""}
          </span>
          {mode === "owned" ? (
            <Pill tone={p.youOwn ? (p.youBenched ? "cyan" : "neon") : "grey"}>
              {p.youOwn ? (p.youBenched ? "You own, benched" : "You own") : "You don't"}
            </Pill>
          ) : (
            <span className="pl-pts">
              <b>{p.projected != null ? p.projected.toFixed(1) : "–"}</b>
              <small>projected</small>
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export function LeaguePage({ data, startRival = null }: { data: Latest; startRival?: number | null }) {
  const l = readLeague(data);
  const phone = usePhone();
  const [selected, setSelected] = useState<number | null>(() => startRival ?? (l ? defaultRival(l, l.rivalsInArea[0]?.managerId ?? null) : null));
  const [overlay, setOverlay] = useState<number | null>(() => (phone ? startRival : null));
  const [leagueId, setLeagueId] = useState<number | null>(l?.leagueId ?? null);

  if (!l) {
    return (
      <Card title="Your leagues">
        <p className="muted">No league data in this update yet. Press Refresh to run a fresh update.</p>
      </Card>
    );
  }

  const pick = (id: number) => (phone ? setOverlay(id) : setSelected(id));
  const supported = l.leagues.find((x) => x.leagueId === leagueId)?.supported ?? leagueId === l.leagueId;

  const selector = (
    <div className="league-select">
      <label htmlFor="league">League</label>
      <select id="league" value={leagueId ?? ""} onChange={(e) => setLeagueId(Number(e.target.value))}>
        {(l.leagues.length ? l.leagues : [{ leagueId: l.leagueId ?? 0, name: l.leagueName ?? "Your league", supported: true }]).map(
          (x) => (
            <option key={x.leagueId} value={x.leagueId}>
              {x.name}
            </option>
          ),
        )}
      </select>
    </div>
  );

  if (!supported) {
    return (
      <div className="league-page">
        <div className="page-title">
          <h1>Your leagues</h1>
          {selector}
        </div>
        <Card title="Coming soon">
          <p>Support for more leagues is coming. For now the League page covers {l.leagueName ?? "your main league"}.</p>
        </Card>
      </div>
    );
  }

  if (phone && overlay != null) {
    return (
      <div className="league-overlay">
        <RivalTeamView l={l} managerId={overlay} onBack={() => setOverlay(null)} />
      </div>
    );
  }

  return (
    <div className="league-page">
      <div className="page-title">
        <h1>Your leagues</h1>
        {selector}
      </div>
      <div className="league-grid">
        <div className="lg-cards">
          <HeadlineCards l={l} />
          <p className="strategy-line">{l.strategy.text}</p>
        </div>
        <div className="lg-table">
          <Standings l={l} selected={phone ? null : selected} onPick={pick} />
        </div>
        <div className="lg-side">
          {!phone && selected != null && <RivalTeamView l={l} managerId={selected} />}
        </div>
        <div className="lg-rivals">
          <RivalsInArea l={l} phone={phone} />
        </div>
        <div className="lg-below">
          <Card title="Who owns what" aside={`your ${l.rivalsCount} rivals' latest teams`}>
            <PlayerList players={l.topOwned} rivals={l.rivalsCount} mode="owned" />
          </Card>
          <Card title="Popular players you don't own" aside="their big week would hurt you most">
            <PlayerList players={l.popularNotOwned} rivals={l.rivalsCount} mode="notOwned" />
          </Card>
          <Card title="Differentials" aside="few or none of your rivals own them">
            <PlayerList players={l.differentials} rivals={l.rivalsCount} mode="diff" />
          </Card>
        </div>
      </div>
    </div>
  );
}
