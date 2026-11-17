import { Component, createContext, type ErrorInfo, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { inTauri } from "../platform";

// Without a boundary, one exception while rendering unmounts the whole React
// root and the window goes blank. Every page and every card is wrapped in
// one of these, so a crash only replaces that page or card with a short
// message and the rest of the app keeps working.
//
// A boundary clears itself when the data changes (a Refresh), via
// DataVersion, or when its resetKey changes (e.g. another tab).

/** The loaded latest.json (or anything that changes on Refresh). */
export const DataVersion = createContext<unknown>(null);

export const CRASH_TEXT = "Something went wrong here. Press Refresh.";

interface Props {
  children?: ReactNode;
  /** What to show instead; defaults to the message in a plain paragraph. */
  fallback?: ReactNode;
  resetKey?: unknown;
  /** For the log: which page or card this is. */
  where?: string;
}
interface State {
  failed: boolean;
  version: unknown;
  resetKey: unknown;
}

export class ErrorBoundary extends Component<Props, State> {
  static contextType = DataVersion;
  declare context: unknown;
  state: State = { failed: false, version: undefined, resetKey: undefined };

  static getDerivedStateFromError(): Partial<State> {
    return { failed: true };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.resetKey) return { resetKey: props.resetKey, failed: false };
    return null;
  }

  componentDidMount() {
    this.setState({ version: this.context });
  }

  componentDidUpdate() {
    if (this.context !== this.state.version) this.setState({ version: this.context, failed: false });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const message = `UI crash in ${this.props.where ?? "a card"}: ${error?.stack ?? error}${info.componentStack ?? ""}`;
    console.error(message);
    if (inTauri()) invoke("ui_log", { message }).catch(() => {});
  }

  render() {
    if (this.state.failed) return this.props.fallback ?? <p className="muted crash-msg">{CRASH_TEXT}</p>;
    return this.props.children;
  }
}

/** Card-shaped fallback, for cards that would otherwise vanish. */
export function CrashCard({ className = "card" }: { className?: string }) {
  return (
    <section className={className}>
      <p className="muted crash-msg">{CRASH_TEXT}</p>
    </section>
  );
}
