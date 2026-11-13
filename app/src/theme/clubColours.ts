// Club colours for the drawn fallback shirt (used when a real shirt image
// can't be loaded, or everywhere when SHIRT_STYLE is "club-colours").
// Keyed by FPL's three-letter team short name, which latest.json carries
// (team_short / fixture opponent_short). Unknown clubs get a neutral kit.

export interface Kit {
  body: string;
  sleeves: string;
  stripes?: string; // vertical stripes over the body
}

export const CLUB_KITS: Record<string, Kit> = {
  ARS: { body: "#EF0107", sleeves: "#FFFFFF" },
  AVL: { body: "#670E36", sleeves: "#95BFE5" },
  BOU: { body: "#DA291C", sleeves: "#DA291C", stripes: "#000000" },
  BRE: { body: "#E30613", sleeves: "#E30613", stripes: "#FFFFFF" },
  BHA: { body: "#0057B8", sleeves: "#0057B8", stripes: "#FFFFFF" },
  BUR: { body: "#6C1D45", sleeves: "#99D6EA" },
  CHE: { body: "#034694", sleeves: "#034694" },
  CRY: { body: "#1B458F", sleeves: "#1B458F", stripes: "#C4122E" },
  EVE: { body: "#003399", sleeves: "#003399" },
  FUL: { body: "#FFFFFF", sleeves: "#000000" },
  IPS: { body: "#3A64A3", sleeves: "#FFFFFF" },
  LEE: { body: "#FFFFFF", sleeves: "#1D428A" },
  LEI: { body: "#003090", sleeves: "#003090" },
  LIV: { body: "#C8102E", sleeves: "#C8102E" },
  MCI: { body: "#6CABDD", sleeves: "#6CABDD" },
  MUN: { body: "#DA291C", sleeves: "#DA291C" },
  NEW: { body: "#FFFFFF", sleeves: "#241F20", stripes: "#241F20" },
  NFO: { body: "#DD0000", sleeves: "#DD0000" },
  SHU: { body: "#EE2737", sleeves: "#EE2737", stripes: "#FFFFFF" },
  SOU: { body: "#D71920", sleeves: "#D71920", stripes: "#FFFFFF" },
  SUN: { body: "#EB172B", sleeves: "#EB172B", stripes: "#FFFFFF" },
  TOT: { body: "#FFFFFF", sleeves: "#132257" },
  WAT: { body: "#FBEE23", sleeves: "#ED2127" },
  WHU: { body: "#7A263A", sleeves: "#1BB1E7" },
  WOL: { body: "#FDB913", sleeves: "#231F20" },
};

export const NEUTRAL_KIT: Kit = { body: "#d9cfdc", sleeves: "#b7a9ba" };
export const GOALKEEPER_KIT: Kit = { body: "#f5c400", sleeves: "#37003c" };

export function kitFor(teamShort: string | null | undefined, goalkeeper = false): Kit {
  if (goalkeeper) return GOALKEEPER_KIT;
  return (teamShort && CLUB_KITS[teamShort]) || NEUTRAL_KIT;
}
