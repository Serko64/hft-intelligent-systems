import type { LineSeries } from "@nivo/line"
import type { StatsMsg } from "@/lib/types"
import { ZoomableLine } from "@/components/ZoomableLine"

interface Props {
  history: StatsMsg[]
}

/** Performance over the evolution: best & mean fitness per generation (zoomable). */
export function FitnessChart({ history }: Props) {
  if (history.length < 2) {
    return <p className="text-xs text-muted-foreground">Warte auf Generationen…</p>
  }

  const data: LineSeries[] = [
    { id: "Best", data: history.map((s) => ({ x: s.generation, y: Math.round(s.best_fitness) })) },
    { id: "Ø", data: history.map((s) => ({ x: s.generation, y: Math.round(s.mean_fitness) })) },
  ]

  return (
    <ZoomableLine
      data={data}
      colors={["#facc15", "#38bdf8"]}
      axisBottomLegend="Generation"
      axisLeftLegend="Fitness"
      height={160}
      showLegend
    />
  )
}
