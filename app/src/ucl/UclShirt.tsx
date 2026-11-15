import { useId } from "react";
import { inkOn, uclKit } from "./kits";

// A UCL club shirt in the same shape as the PL drawn shirt
// (components/Shirt.tsx), coloured from ucl/kits.ts. The club's short name
// sits on the chest when the shirt is big enough to read it, and as a small
// caption under it when it isn't.

const SHIRT_PATH =
  "M34 8 L22 12 L6 26 L14 42 L24 36 L24 92 L76 92 L76 36 L86 42 L94 26 L78 12 L66 8 C62 16 38 16 34 8 Z";
const BODY_PATH = "M24 30 L24 92 L76 92 L76 30 L66 8 C62 16 38 16 34 8 Z";
const CHEST_MIN = 36; // px: below this the code goes under the shirt

export function UclShirt({
  code,
  goalkeeper = false,
  size = 42,
  label = true,
  title,
}: {
  code: string | null | undefined;
  goalkeeper?: boolean;
  size?: number;
  label?: boolean;
  title?: string;
}) {
  const k = uclKit(code);
  const id = useId().replace(/:/g, "");
  const body = goalkeeper ? k.gk ?? "#1FA85A" : k.body;
  const sleeves = goalkeeper ? body : k.sleeves ?? k.body;
  const trim = goalkeeper ? "#111111" : k.trim ?? "rgba(10,15,46,0.5)";
  const pat = goalkeeper ? "plain" : k.pattern;
  const second = k.second ?? k.body;
  let fill = body;
  let defs = null;
  if (pat === "stripes") {
    defs = (
      <pattern id={`p${id}`} width="13" height="100" patternUnits="userSpaceOnUse" x="24">
        <rect width="13" height="100" fill={body} />
        <rect x="6.5" width="6.5" height="100" fill={second} />
      </pattern>
    );
    fill = `url(#p${id})`;
  } else if (pat === "hoops") {
    defs = (
      <pattern id={`p${id}`} width="100" height="14" patternUnits="userSpaceOnUse" y="30">
        <rect width="100" height="14" fill={body} />
        <rect y="7" width="100" height="7" fill={second} />
      </pattern>
    );
    fill = `url(#p${id})`;
  }
  const chest = label && size >= CHEST_MIN && code;
  const chestBg = pat === "plain" || pat === "band" ? body : "#FFFFFF";
  return (
    <span className="ushirt" style={{ width: size }} title={title ?? code ?? undefined}>
      <svg width={size} height={size} viewBox="0 0 100 100" role="img" aria-label={title ?? `${code ?? ""} shirt`}>
        {defs && <defs>{defs}</defs>}
        <path d={SHIRT_PATH} fill={sleeves} />
        <path d={BODY_PATH} fill={fill} />
        {pat === "centre" && (
          <>
            <rect x="40" y="12" width="20" height="80" fill={k.edge ?? "#FFFFFF"} />
            <rect x="44" y="12" width="12" height="80" fill={second} />
          </>
        )}
        {pat === "halves" && <path d="M50 14 L66 8 L76 30 L76 92 L50 92 Z" fill={second} />}
        {pat === "band" && <rect x="24" y="40" width="52" height="14" fill={second} />}
        <path d={SHIRT_PATH} fill="none" stroke="rgba(10,15,46,0.45)" strokeWidth="2.5" strokeLinejoin="round" />
        <path d="M34 8 C38 16 62 16 66 8" fill="none" stroke={trim} strokeWidth="4" />
        <path d="M8 27 L14 40 M92 27 L86 40" stroke={trim} strokeWidth="4" />
        {chest && (
          <>
            <rect x="29" y="56" width="42" height="21" rx="5" fill={chestBg} opacity={pat === "plain" || pat === "band" ? 0 : 0.92} />
            <text
              x="50"
              y="71.5"
              textAnchor="middle"
              fontSize={code!.length > 3 ? 13 : 16}
              fontWeight="800"
              fontFamily="Inter, sans-serif"
              fill={pat === "plain" || pat === "band" ? inkOn(body) : "#0A0F2E"}
            >
              {code}
            </text>
          </>
        )}
      </svg>
      {label && !chest && code && <span className="ushirt-cap">{code}</span>}
    </span>
  );
}
