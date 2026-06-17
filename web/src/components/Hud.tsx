import { memo, useEffect, useState, type MutableRefObject } from "react"
import type { Car, StatsMsg } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Props {
  carsRef: MutableRefObject<Car[]>
  stats: StatsMsg | null
}

/**
 * Telemetry overlay. Reads the lead car from the frame ref at ~10 Hz (so it does
 * not re-render at the full 60 fps) and shows per-generation training stats.
 */
function HudImpl({ carsRef, stats }: Props) {
  const [lead, setLead] = useState<Car | null>(null)
  const [count, setCount] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      const cars = carsRef.current
      setLead(cars[0] ?? null)
      setCount(cars.length)
    }, 100)
    return () => clearInterval(id)
  }, [carsRef])

  return (
    <Card className="w-[260px] shrink-0">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          Telemetrie
          <Badge variant="secondary">{count} Autos</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <Row label="Tempo" value={lead ? `${(lead.speed * 3.6).toFixed(0)} km/h` : "—"} />
        <Row label="Runde" value={lead ? String(lead.lap + 1) : "—"} />
        <Row label="Checkpoint" value={lead ? `${lead.checkpoint}/40` : "—"} />
        <Row label="Score (Spitze)" value={lead ? lead.score.toFixed(0) : "—"} />

        {stats && (
          <>
            <div className="h-px bg-border my-1" />
            <Row label="Generation" value={String(stats.generation)} />
            <Row label="Best Fitness" value={stats.best_fitness.toFixed(0)} />
            <Row label="Ø Fitness" value={stats.mean_fitness.toFixed(0)} />
            <Row label="Epsilon" value={stats.epsilon.toFixed(3)} />
          </>
        )}

        {stats && stats.top_scores && stats.top_scores.length > 0 && (
          <div className="mt-2">
            <div className="h-px bg-border my-1" />
            <div className="mb-1 text-xs font-semibold text-muted-foreground">Top 10</div>
            <ol className="space-y-0.5 text-xs">
              {stats.top_scores.slice(0, 10).map(([score, gen], i) => (
                <li key={`${gen}-${i}`} className="flex justify-between tabular-nums">
                  <span className={i === 0 ? "font-semibold text-yellow-400" : "text-muted-foreground"}>
                    {i + 1}. Gen {gen}
                  </span>
                  <span className={i === 0 ? "font-semibold text-yellow-400" : "font-medium"}>
                    {score.toFixed(0)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export const Hud = memo(HudImpl)

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums font-medium">{value}</span>
    </div>
  )
}
