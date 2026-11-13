import type { ReactNode } from "react";

/** Lilac rounded card with a bold title bar -- the building block of every page. */
export function Card({
  title,
  aside,
  children,
  className = "",
}: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-head">
          {title && <h2>{title}</h2>}
          {aside && <span className="card-aside">{aside}</span>}
        </header>
      )}
      {children}
    </section>
  );
}

export type PillTone = "purple" | "neon" | "pink" | "grey" | "cyan";

export function Pill({ tone = "purple", children, title }: { tone?: PillTone; children: ReactNode; title?: string }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {children}
    </span>
  );
}
