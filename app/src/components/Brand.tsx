// XPert's own marks (original; no league, club or FPL artwork). Everything
// is drawn in currentColor, so the marks stay neutral and follow the text
// colour they sit in. Font: Barlow Condensed Bold (OFL), bundled, used only
// for the logo and wordmark. Source SVGs: docs/brand/.

const FONT = "'Barlow Condensed', 'Arial Narrow', sans-serif";

/** The tile with "xP". ``full``: faint pitch markings (both penalty boxes and
 * a halfway line that stops at the centre circle, so it never crosses the
 * letters) and a slightly stronger circle. ``simple`` (small sizes): only
 * the centre circle, the halfway-line stubs and "xP". */
export function XpMark({ size = 32, variant = "simple", title = "XPert" }: { size?: number; variant?: "full" | "simple"; title?: string }) {
  const full = variant === "full";
  return (
    <svg className="xp-mark" viewBox="0 0 100 100" width={size} height={size} role="img" aria-label={title}>
      <rect x="3.5" y="3.5" width="93" height="93" rx="22" fill="none" stroke="currentColor" strokeWidth={full ? 4 : 6} />
      {full && (
        <g fill="none" stroke="currentColor" strokeWidth="2" opacity="0.28">
          <path d="M3.5 30 H20 V70 H3.5" />
          <path d="M96.5 30 H80 V70 H96.5" />
          <path d="M3.5 41 H9 V59 H3.5" />
          <path d="M96.5 41 H91 V59 H96.5" />
          <path d="M50 3.5 V26" />
          <path d="M50 74 V96.5" />
        </g>
      )}
      {!full && (
        <g stroke="currentColor" strokeWidth="5" opacity="0.6">
          <path d="M50 3.5 V15" />
          <path d="M50 85 V96.5" />
        </g>
      )}
      <circle cx="50" cy="50" r="24" fill="none" stroke="currentColor" strokeWidth={full ? 2.6 : 4} opacity={full ? 0.55 : 0.6} />
      <text
        x="50"
        y="51.5"
        textAnchor="middle"
        dominantBaseline="central"
        fontFamily={FONT}
        fontWeight={700}
        fontSize={full ? 34 : 38}
        fill="currentColor"
        letterSpacing="-0.5"
      >
        xP
      </text>
    </svg>
  );
}

/** The simplified icon, then "XPert" and the tagline (dropped when ``compact``). */
export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="xp-wordmark">
      <XpMark size={compact ? 30 : 34} />
      {!compact && (
        <span className="xp-words">
          <b>XPert</b>
          <small>your fantasy consultant</small>
        </span>
      )}
    </span>
  );
}
