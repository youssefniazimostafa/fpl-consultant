import type { ReactNode } from "react";
import { Shirt } from "./Shirt";
import { fitName } from "./PlayerTile";

export interface PlayerCardProps {
  name: string;
  teamCode: number | null;
  teamShort: string | null;
  position: string | null;
  youOwn?: boolean;
  /** the card's headline number, e.g. "5.4 pts" (green pill) */
  keyNumber: ReactNode;
  /** small line under the name, e.g. "£15.8m · FUL (H)" */
  sub?: ReactNode;
  /** optional extra line (news text, "2nd week running" badge...) */
  extra?: ReactNode;
}

/** A player card as in the FPL overview mock-up: purple header strip,
 *  shirt, name and one key number. Shared by every overview section. */
export function PlayerCard({ name, teamCode, teamShort, position, youOwn, keyNumber, sub, extra }: PlayerCardProps) {
  const { text, scale } = fitName(name);
  return (
    <div className="pcard">
      <div className="pcard-head">
        <span>{[teamShort, position === "GKP" ? "GK" : position].filter(Boolean).join(" · ")}</span>
        {youOwn && <span className="pcard-own">You own</span>}
      </div>
      <div className="pcard-body">
        <Shirt teamCode={teamCode} teamShort={teamShort} goalkeeper={position === "GKP"} size={52} title={name} />
        <div className="pcard-name" style={{ fontSize: `${15 * scale}px` }} title={name}>
          {text}
        </div>
        {sub && <div className="pcard-sub">{sub}</div>}
        <div className="pcard-key">{keyNumber}</div>
        {extra && <div className="pcard-extra">{extra}</div>}
      </div>
    </div>
  );
}
