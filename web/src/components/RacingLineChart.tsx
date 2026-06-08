import type { LineSeries } from "@nivo/line"
import type { RacingLineMsg } from "@/lib/types"
import { ZoomableLine } from "@/components/ZoomableLine"

interface Props {
  line: RacingLineMsg
}

/** For the current best lap: speed (km/h) and throttle (accel vs brake) over time.
 *  Both charts are zoomable (wheel) and pannable (drag). */
export function RacingLineChart({ line }: Props) {
  const pts = line.points
  if (pts.length < 2) {
    return <p className="text-xs text-muted-foreground">Noch keine beste Runde.</p>
  }

  // Downsample to keep the chart light; x = time in seconds (60 fps recording).
  const step = Math.max(1, Math.ceil(pts.length / 400))
  const speed: { x: number; y: number }[] = []
  const throttle: { x: number; y: number }[] = []
  for (let i = 0; i < pts.length; i += step) {
    const t = +(i / 60).toFixed(2)
    speed.push({ x: t, y: Math.round(pts[i][2] * 3.6) }) // m/s → km/h
    throttle.push({ x: t, y: +pts[i][3].toFixed(2) }) // −1 (Vollbremsung) … 1 (Vollgas)
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
