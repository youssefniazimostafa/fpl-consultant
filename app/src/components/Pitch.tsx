import { PlayerTile } from "./PlayerTile";
import type { PlayerView } from "../model/team";

const ROWS = ["GKP", "DEF", "MID", "FWD"] as const;

/** Striped pitch with the starting XI in formation, goalkeeper at the top. */
export function Pitch({ players }: { players: PlayerView[] }) {
  return (
    <div className="pitch">
      <div className="pitch-box" aria-hidden />
      {ROWS.map((pos) => {
        const row = players.filter((p) => p.position === pos);
        if (!row.length) return null;
        return (
          <div key={pos} className="pitch-row">
            {row.map((p) => (
              <PlayerTile key={p.id} p={p} />
            ))}
          </div>
        );
      })}
      <div className="pitch-legend">Projected points under each name</div>
    </div>
  );
}

/** The four substitutes in bench order, under the pitch. */
export function Bench({ players }: { players: PlayerView[] }) {
  return (
    <div className="bench">
      {players.map((p) => (
        <PlayerTile key={p.id} p={p} label={p.benchLabel ?? undefined} size={44} />
      ))}
    </div>
  );
}
