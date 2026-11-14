import { useEffect, useId, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { SHIRT_STYLE } from "../config";
import { VIEWER } from "../platform";
import { kitFor } from "../theme/clubColours";

// Desktop: one request per shirt per app session; the backend caches the
// files on disk, so after the first download they also work offline.
// Viewer (web / iPhone): the image straight from FPL's server (the browser
// caches it); if it can't load, the drawn shirt is shown instead.
const fplShirtUrl = (teamCode: number, goalkeeper: boolean) =>
  `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${teamCode}${goalkeeper ? "_1" : ""}-110.png`;
const imageCache = new Map<string, Promise<string>>();

function loadShirt(teamCode: number, goalkeeper: boolean): Promise<string> {
  const key = `${teamCode}-${goalkeeper}`;
  let p = imageCache.get(key);
  if (!p) {
    p = invoke<string>("shirt_image", { teamCode, goalkeeper });
    p.catch((e) => {
      imageCache.delete(key); // retry next time (e.g. back online)
      invoke("ui_log", { message: `shirt ${key}: using drawn club colours (${e})` }).catch(() => {});
    });
    imageCache.set(key, p);
  }
  return p;
}

export interface ShirtProps {
  teamCode?: number | null;
  teamShort?: string | null;
  goalkeeper?: boolean;
  size?: number; // px width
  dimmed?: boolean; // e.g. the player going out in a transfer
  title?: string;
}

/** A club shirt: the real image when available, else a drawn club-colours shirt. */
export function Shirt({ teamCode, teamShort, goalkeeper = false, size = 56, dimmed = false, title }: ShirtProps) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    setSrc(null);
    if (SHIRT_STYLE !== "real" || teamCode == null) return;
    if (VIEWER) {
      setSrc(fplShirtUrl(teamCode, goalkeeper));
      return;
    }
    let live = true;
    loadShirt(teamCode, goalkeeper).then(
      (url) => live && setSrc(url),
      () => {}, // keep the drawn fallback
    );
    return () => {
      live = false;
    };
  }, [teamCode, goalkeeper]);

  const style = { width: size, height: size };
  const cls = `shirt${dimmed ? " dimmed" : ""}`;
  if (src)
    return (
      <img
        className={cls}
        style={style}
        src={src}
        alt={title ?? "shirt"}
        title={title}
        draggable={false}
        onError={() => setSrc(null)} // viewer: offline or blocked -> drawn shirt
      />
    );
  return <DrawnShirt className={cls} size={size} teamShort={teamShort} goalkeeper={goalkeeper} title={title} />;
}

const SHIRT_PATH =
  "M34 8 L22 12 L6 26 L14 42 L24 36 L24 92 L76 92 L76 36 L86 42 L94 26 L78 12 L66 8 C62 16 38 16 34 8 Z";
const BODY_PATH = "M24 30 L24 92 L76 92 L76 30 L66 8 C62 16 38 16 34 8 Z";

function DrawnShirt({
  size,
  teamShort,
  goalkeeper,
  className,
  title,
}: {
  size: number;
  teamShort?: string | null;
  goalkeeper: boolean;
  className: string;
  title?: string;
}) {
  const kit = kitFor(teamShort, goalkeeper);
  const id = useId().replace(/:/g, "");
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 100 100" role="img" aria-label={title ?? "shirt"}>
      {title && <title>{title}</title>}
      {kit.stripes && (
        <defs>
          <pattern id={`s${id}`} width="16" height="100" patternUnits="userSpaceOnUse">
            <rect width="16" height="100" fill={kit.body} />
            <rect width="8" height="100" fill={kit.stripes} />
          </pattern>
        </defs>
      )}
      <path d={SHIRT_PATH} fill={kit.sleeves} stroke="rgba(55,0,60,0.35)" strokeWidth="2" strokeLinejoin="round" />
      <path d={BODY_PATH} fill={kit.stripes ? `url(#s${id})` : kit.body} />
      <path d={SHIRT_PATH} fill="none" stroke="rgba(55,0,60,0.35)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M34 8 C38 16 62 16 66 8" fill="none" stroke="rgba(55,0,60,0.45)" strokeWidth="3" />
    </svg>
  );
}
