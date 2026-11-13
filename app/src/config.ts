// App-wide switches.

/**
 * "real": each club's real shirt image from the FPL site (personal use
 * only), cached on disk; falls back to a drawn club-colours shirt if an
 * image can't be loaded.
 * "club-colours": always the drawn shirt -- switch to this before sharing
 * the app with anyone.
 */
export const SHIRT_STYLE: "real" | "club-colours" = "real";
