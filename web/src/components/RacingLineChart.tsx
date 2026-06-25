import type { LineSeries } from "@nivo/line"
import type { RacingLineMsg } from "@/lib/types"
import { ZoomableLine } from "@/components/ZoomableLine"

interface Props {
  line: RacingLineMsg
}

/** Für die aktuell beste Runde: Tempo (km/h) und Gas/Bremse über die Zeit.
 *  Beide Charts sind zoombar (Mausrad) und verschiebbar (Ziehen). */
export function RacingLineChart({ line }: Props) {
  const pts = line.points
  if (pts.length < 2) {
    return <p className="text-xs text-muted-foreground">Noch keine beste Runde.</p>
  }

  // Herunterrechnen, damit der Chart leicht bleibt. x = Zeit in Sekunden (Aufnahme mit 60 fps).
  const step = Math.max(1, Math.ceil(pts.length / 400))
  const speed: { x: number; y: number }[] = []
  const throttle: { x: number; y: number }[] = []
  for (let i = 0; i < pts.length; i += step) {
    const t = +(i / 60).toFixed(2)
    speed.push({ x: t, y: Math.round(pts[i][2] * 3.6) }) // m/s in km/h
    throttle.push({ x: t, y: +pts[i][3].toFixed(2) }) // -1 (Vollbremsung) bis 1 (Vollgas)
  }

  const speedData: LineSeries[] = [{ id: "Tempo", data: speed }]
  const throttleData: LineSeries[] = [{ id: "Gas/Bremse", data: throttle }]

  return (
    <div className="space-y-1">
      <ZoomableLine
        data={speedData}
        colors={["#22c55e"]}
        yMin={0}
        axisLeftLegend="km/h"
        height={110}
      />
      <ZoomableLine
        data={throttleData}
        colors={["#f59e0b"]}
        yMin={-1}
        yMax={1}
        yTicks={[-1, 0, 1]}
        axisLeftLegend="Gas"
        axisBottomLegend="Zeit (s)"
        markers={[{ axis: "y", value: 0 }]}
        height={110}
      />
    </div>
  )
}
