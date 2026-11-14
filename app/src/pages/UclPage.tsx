import { XpMark } from "../components/Brand";

// UCL mode, for now: a quiet "to be continued" page. Nothing runs here (no
// data, pipeline or chat); a later UCL version only fills in pages, the
// theme already lives under :root[data-game="ucl"] in theme.css.
export function UclPage() {
  return (
    <div className="ucl-page">
      <div className="ucl-card">
        <span className="ucl-mark"><XpMark size={88} variant="full" /></span>
        <h1>Champions League</h1>
        <p className="ucl-tbc">To be continued</p>
        <p className="ucl-line">XPert for the Champions League fantasy game is coming.</p>
      </div>
    </div>
  );
}
