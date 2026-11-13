import { Shirt } from "./Shirt";
import type { PlayerView } from "../model/team";

/** A player as on FPL's pick-team screen: shirt, badges, name, fixture + projected points. */
export function PlayerTile({ p, label, size }: { p: PlayerView; label?: string; size?: number }) {
  return (
    <div className={`player-tile${p.flagged ? " flagged" : ""}`} title={p.news ?? undefined}>
      {label && <div className="tile-label">{label}</div>}
      <div className="tile-shirt">
        <Shirt teamCode={p.teamCode} teamShort={p.teamShort} goalkeeper={p.position === "GKP"} size={size} title={p.name} />
        {p.isCaptain && <span className="badge badge-c" title="Captain">C</span>}
        {p.isVice && <span className="badge badge-v" title="Vice-captain">V</span>}
        {p.isNew && <span className="badge badge-new" title="New signing this week" />}
        {p.flagged && <span className="badge badge-flag" title={p.news ?? "Injury or availability doubt"}>!</span>}
      </div>
      <div className="tile-name">{p.name}</div>
      <div className="tile-meta">
        <span>{p.opponent ?? "–"}</span>
        <b>{p.projected != null ? p.projected.toFixed(1) : "–"}</b>
      </div>
    </div>
  );
}
