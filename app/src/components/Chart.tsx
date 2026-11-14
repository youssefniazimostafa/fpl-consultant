import { useEffect, useRef } from "react";
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  type ChartConfiguration,
} from "chart.js";
import { colors, font } from "../theme/theme";

// Chart.js, bundled into the app by Vite (works offline, no CDN), styled
// with the theme. One small wrapper so every page draws charts the same way.
ChartJS.register(LineController, BarController, LineElement, BarElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);
ChartJS.defaults.font.family = font.family;
ChartJS.defaults.color = colors.textMuted;
ChartJS.defaults.borderColor = colors.lilacDeep;

export const SERIES = [colors.purple, colors.neon, colors.pink, colors.cyan, colors.grey];

export function Chart({ config, height = 240 }: { config: ChartConfiguration; height?: number }) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (!canvas.current) return;
    const chart = new ChartJS(canvas.current, {
      ...config,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, boxHeight: 12 } }, ...(config.options?.plugins ?? {}) },
        ...config.options,
      },
    } as ChartConfiguration);
    return () => chart.destroy();
  }, [config]);
  return (
    <div className="chart-box" style={{ height }}>
      <canvas ref={canvas} />
    </div>
  );
}
