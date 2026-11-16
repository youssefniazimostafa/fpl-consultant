// UCL club shirts: original, simple designs in each club's real colours and
// main pattern -- no crests, sponsors or logos. Keyed by UEFA's club code
// (club_code in data/ucl/latest.json). Edit a line here to correct a club.
//
// pattern: "plain" | "stripes" (vertical) | "hoops" (horizontal) |
//          "centre" (one central vertical stripe, edged in `edge`) |
//          "halves" (left `body`, right `second`) | "band" (chest band)
// sleeves defaults to body; trim is the collar and cuffs; gk is the
// goalkeeper shirt (plain), chosen to stand out from the outfield kit.

export type Pattern = "plain" | "stripes" | "hoops" | "centre" | "halves" | "band";

export interface UclKit {
  body: string;
  second?: string; // stripes / hoops / centre stripe / right half / band
  edge?: string; // edging of the centre stripe
  sleeves?: string;
  trim?: string;
  pattern: Pattern;
  gk?: string;
}

const GK_GREEN = "#1FA85A";
const GK_ORANGE = "#F28C28";
const GK_PURPLE = "#6B3FA0";

export const UCL_KITS: Record<string, UclKit> = {
  AEK: { body: "#F6D10C", trim: "#111111", pattern: "plain", gk: GK_GREEN }, // AEK Athens: yellow, black trim
  ARS: { body: "#DB0007", sleeves: "#FFFFFF", trim: "#FFFFFF", pattern: "plain", gk: GK_GREEN }, // red body, white sleeves
  ATM: { body: "#CB3524", second: "#FFFFFF", sleeves: "#CB3524", trim: "#1B3A6B", pattern: "stripes", gk: GK_GREEN }, // red and white stripes
  AVL: { body: "#670E36", sleeves: "#95BFE5", trim: "#95BFE5", pattern: "plain", gk: GK_GREEN }, // claret body, sky-blue sleeves
  BAR: { body: "#004D98", second: "#A50044", sleeves: "#004D98", trim: "#EDBB00", pattern: "stripes", gk: GK_ORANGE }, // blue and red stripes
  BAY: { body: "#DC052D", trim: "#FFFFFF", sleeves: "#DC052D", pattern: "plain", gk: GK_GREEN }, // red with white accents
  BET: { body: "#00954C", second: "#FFFFFF", sleeves: "#00954C", trim: "#FFFFFF", pattern: "stripes", gk: GK_ORANGE }, // green and white stripes
  BOD: { body: "#FFE600", trim: "#111111", pattern: "plain", gk: GK_PURPLE }, // Bodø/Glimt: yellow
  BRU: { body: "#0067B1", second: "#111111", sleeves: "#0067B1", trim: "#111111", pattern: "stripes", gk: GK_ORANGE }, // blue and black stripes
  BVB: { body: "#FDE100", trim: "#111111", sleeves: "#FDE100", pattern: "plain", gk: GK_PURPLE }, // Dortmund: yellow, black trim
  COM: { body: "#1D63B0", trim: "#FFFFFF", pattern: "plain", gk: GK_ORANGE }, // Como: blue
  FEN: { body: "#FFED00", second: "#00215B", sleeves: "#00215B", trim: "#00215B", pattern: "stripes", gk: GK_GREEN }, // yellow and navy stripes
  FEY: { body: "#E3001B", second: "#FFFFFF", trim: "#111111", pattern: "halves", gk: GK_GREEN }, // Feyenoord: red and white halves
  GAL: { body: "#FDB912", second: "#A90432", trim: "#A90432", pattern: "halves", gk: GK_GREEN }, // Galatasaray: yellow and red halves
  INT: { body: "#0068A8", second: "#111111", sleeves: "#0068A8", trim: "#111111", pattern: "stripes", gk: GK_ORANGE }, // blue and black stripes
  JUV: { body: "#FFFFFF", second: "#111111", sleeves: "#111111", trim: "#111111", pattern: "stripes", gk: GK_ORANGE }, // black and white stripes
  LASK: { body: "#111111", second: "#FFFFFF", sleeves: "#111111", trim: "#FFFFFF", pattern: "stripes", gk: GK_GREEN }, // black and white stripes
  LEN: { body: "#FFD700", sleeves: "#E2001A", trim: "#E2001A", pattern: "plain", gk: GK_GREEN }, // Lens: gold body, red sleeves
  LIL: { body: "#E01E13", trim: "#20325F", pattern: "plain", gk: GK_GREEN }, // Lille: red, navy trim
  LIV: { body: "#C8102E", trim: "#FFFFFF", pattern: "plain", gk: GK_GREEN }, // Liverpool: red
  MCI: { body: "#6CABDD", trim: "#FFFFFF", pattern: "plain", gk: GK_ORANGE }, // Man City: sky blue
  MUN: { body: "#DA291C", trim: "#FFFFFF", pattern: "plain", gk: GK_GREEN }, // Man Utd: red, white trim
  NAP: { body: "#12A0D7", trim: "#FFFFFF", pattern: "plain", gk: GK_ORANGE }, // Napoli: azzurro
  POR: { body: "#FFFFFF", second: "#00428C", sleeves: "#00428C", trim: "#00428C", pattern: "stripes", gk: GK_GREEN }, // blue and white stripes
  PSG: { body: "#004170", second: "#DA291C", edge: "#FFFFFF", trim: "#DA291C", pattern: "centre", gk: GK_GREEN }, // navy, red centre stripe edged white
  PSV: { body: "#ED1C24", second: "#FFFFFF", sleeves: "#ED1C24", trim: "#111111", pattern: "stripes", gk: GK_GREEN }, // red and white stripes
  RBL: { body: "#FFFFFF", trim: "#DD0741", sleeves: "#FFFFFF", pattern: "plain", gk: GK_GREEN }, // Leipzig: white, red trim
  RMA: { body: "#FFFFFF", trim: "#C9A227", pattern: "plain", gk: GK_GREEN }, // Real Madrid: all white
  ROM: { body: "#8E1F2F", trim: "#F0BC42", pattern: "plain", gk: GK_GREEN }, // Roma: dark red, gold trim
  SAB: { body: "#111111", sleeves: "#111111", trim: "#F0509A", pattern: "plain", gk: GK_GREEN }, // Sabah: black with pink accents
  SBR: { body: "#5DADE2", trim: "#FFFFFF", pattern: "plain", gk: GK_ORANGE }, // Slovan Bratislava: sky blue
  SHK: { body: "#F58220", sleeves: "#111111", trim: "#111111", pattern: "plain", gk: GK_GREEN }, // Shakhtar: orange, black sleeves
  SLA: { body: "#FFFFFF", second: "#D7141A", trim: "#D7141A", pattern: "halves", gk: GK_GREEN }, // Slavia Praha: white and red halves
  SPO: { body: "#008057", second: "#FFFFFF", sleeves: "#008057", trim: "#FFFFFF", pattern: "hoops", gk: GK_ORANGE }, // Sporting: green and white hoops
  STU: { body: "#FFFFFF", second: "#E32219", trim: "#E32219", pattern: "band", gk: GK_GREEN }, // Stuttgart: white, red chest band
  VFK: { body: "#003876", trim: "#FFFFFF", pattern: "plain", gk: GK_ORANGE }, // Viking: dark blue
  VIL: { body: "#FFE667", trim: "#00285E", pattern: "plain", gk: GK_PURPLE }, // Villarreal: yellow, navy trim
};

const NEUTRAL: UclKit = { body: "#C9CCE0", trim: "#5A5F80", pattern: "plain", gk: GK_GREEN };

export function uclKit(code: string | null | undefined): UclKit {
  return (code && UCL_KITS[code]) || NEUTRAL;
}

/** Black or white text, whichever reads better on `hex`. */
export function inkOn(hex: string): string {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.6 ? "#0A0F2E" : "#FFFFFF";
}
