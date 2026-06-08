// Shared dark theme for all nivo charts so they blend into the dark UI.
import type { PartialTheme } from "@nivo/theming"

export const nivoDark: PartialTheme = {
  background: "transparent",
  text: { fill: "#cbd5e1", fontSize: 10 },
  axis: {
    ticks: {
      text: { fill: "#94a3b8", fontSize: 9 },
      line: { stroke: "#334155" },
    },
    legend: { text: { fill: "#cbd5e1", fontSize: 10 } },
    domain: { line: { stroke: "#334155" } },
  },
  grid: { line: { stroke: "#1e293b", strokeWidth: 1 } },
  legends: { text: { fill: "#cbd5e1", fontSize: 10 } },
  tooltip: {
    container: { background: "#0f172a", color: "#e2e8f0", fontSize: 11, borderRadius: 6 },
  },
  crosshair: { line: { stroke: "#64748b", strokeDasharray: "3 3" } },
}
