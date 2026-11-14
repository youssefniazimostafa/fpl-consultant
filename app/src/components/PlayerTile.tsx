import type { CSSProperties } from "react";
import { Shirt } from "./Shirt";
import type { PlayerView } from "../model/team";

// Names longer than this many characters shrink to fit instead of being
// cut off; below MIN_SCALE they'd be unreadable, so only then abbreviate.
const FIT_CHARS = 10;
const MIN_SCALE = 0.62; // ~7.5px on laptop -- fits FPL's longest usual web_names, e.g. "Alexander-Arnold"

/** FPL's web_name, shrunk to fit the label (never truncated with "..."). */
export function fitName(name: string): { text: string; scale: number } {
  const scale = Math.min(1, FIT_CHARS / Math.max(name.length, 1));
  if (scale >= MIN_SCALE) return { text: name, scale };
  // Still too long at the smallest size: drop leading words (keep the
  // surname the FPL app shows), never add initials.
  const words = name.split(" ");
  const short = words.length > 1 ? words[words.length - 1] : name;
  return { text: short, scale: Math.max(MIN_SCALE, Math.min(1, FIT_CHARS / short.length)) };
}

/** A player as on FPL's pick-team screen: shirt, badges, name, fixture + projected points. */
export function PlayerTile({ p, label, size }: { p: PlayerView; label?: string; size?: number }) {
  const { text, scale } = fitName(p.name);
  return (
    <div className={`player-tile${p.flagged ? " flagged" : ""}`} title={p.news ?? p.name}>
      {label && <div className="tile-label">{label}</div>}
      <div className="tile-shirt">
        <Shirt teamCode={p.teamCode} teamShort={p.teamShort} goalkeeper={p.position === "GKP"} size={size} title={p.name} />
        {p.isCaptain && <span className="badge badge-c" title="Captain">C</span>}
        {p.isVice && <span className="badge badge-v" title="Vice-captain">V</span>}
        {p.isNew && <span className="badge badge-new" title="New signing this week" />}
        {p.flagged && <span className="badge badge-flag" title={p.news ?? "Injury or availability doubt"}>!</span>}
      </div>
      <div className="tile-name" style={{ "--name-scale": scale } as CSSProperties}>
        {text}
      </div>
      <div className="tile-meta">
        <span>{p.opponent ?? "–"}</span>
        <b>{p.projected != null ? p.projected.toFixed(1) : "–"}</b>
      </div>
    </div>
  );
}
