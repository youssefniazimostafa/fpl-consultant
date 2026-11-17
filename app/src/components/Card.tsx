import { useState, type ReactNode } from "react";
import { helpFor } from "../help";
import { CrashCard, ErrorBoundary } from "./ErrorBoundary";

/** The small "?" next to a card title: tap for two or three plain sentences. */
export function HelpButton({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="help-wrap">
      <button
        className={`help-btn${open ? " open" : ""}`}
        aria-label="What is this?"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        ?
      </button>
      {open && (
        <span className="help-pop" role="note">
          {text}
          <button className="help-close" aria-label="Close" onClick={() => setOpen(false)}>×</button>
        </span>
      )}
    </span>
  );
}

/** Lilac rounded card with a bold title bar -- the building block of every page.
 *  A crash inside one card only replaces that card (see ErrorBoundary). */
export function Card(props: CardProps) {
  return (
    <ErrorBoundary where={`card ${typeof props.title === "string" ? props.title : ""}`} fallback={<CrashCard className={`card ${props.className ?? ""}`} />}>
      <CardInner {...props} />
    </ErrorBoundary>
  );
}

interface CardProps {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  help?: string; // defaults to the HELP entry for the title (src/help.ts)
}

function CardInner({ title, aside, children, className = "", help }: CardProps) {
  const helpText = help ?? helpFor(title);
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-head">
          {title && (
            <h2>
              {title}
              {helpText && <HelpButton text={helpText} />}
            </h2>
          )}
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
