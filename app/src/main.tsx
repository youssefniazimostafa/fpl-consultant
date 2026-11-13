import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/inter"; // bundled locally, works offline
import App from "./App";
import { applyTheme } from "./theme/theme";
import "./theme/theme.css";

applyTheme();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
