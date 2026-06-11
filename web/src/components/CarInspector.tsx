import { useEffect, useState, type MutableRefObject } from "react"
import { ResponsiveBar } from "@nivo/bar"
import type { Car, InspectMsg, QTableMsg } from "@/lib/types"
import { nivoDark } from "@/lib/nivoTheme"
import { NetView, QValueChart } from "@/components/NetView"
import { QTableHeatmap } from "@/components/QTableHeatmap"

interface Props {
  carsRef: MutableRefObject<Car[]>
  index: number
  inspect: InspectMsg | null
  qtable: QTableMsg | null
  onClose: () => void
}

const G = 9.81

/** Inspector for one clicked car: live stats, score breakdown and the forces
 *  currently acting on it (a small top-down view). Reads the car from the frame
 *  ref at ~10 Hz so it doesn't re-render every frame. */
export function CarInspector({ carsRef, index, inspect, qtable, onClose }: Props) {
  const [car, setCar] = useState<Car | null>(null)
  const [showNet, setShowNet] = useState(false)

  useEffect(() => {
    const id = setInterval(() => setCar(carsRef.current[index] ?? null), 100)
    return () => clearInterval(id)
  }, [carsRef, index])

  if (!car) {
    return (
      <div className="text-xs text-muted-foreground">
        Auto #{index + 1} nicht mehr aktiv.{" "}
        <button className="underline" onClick={onClose}>schließen</button>
      </div>
    )
  }

  // Q-table message for *this* car (q-table backend); null in DQN mode.
  const qtableForThis = qtable && qtable.index === index ? qtable : null
  const isQtable = qtableForThis !== null

  const parts = car.reward_parts ?? {}
  const barData = Object.entries(parts)
    .filter(([, v]) => Math.abs(v) > 0.01)
    .map(([part, v]) => ({ part, value: Math.round(v) }))

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Auto #{index + 1}</span>
        <button className="text-xs text-muted-foreground hover:text-foreground" onClick={onClose}>
          ✕
        </button>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs tabular-nums">
        <Stat label="Generation" value={String(car.generation)} />
        <Stat label="Score" value={car.score.toFixed(0)} />
        <Stat label="Tempo" value={`${(car.speed * 3.6).toFixed(0)} km/h`} />
        <Stat label="Runde" value={String(car.lap + 1)} />
        <Stat label="Checkpoint" value={`${car.checkpoint}/40`} />
        <Stat label="Fortschritt" value={`${(car.progress * 100).toFixed(0)}%`} />
      </div>

      <Forces aLong={car.a_long} aLat={car.a_lat} />

      <Sensors rays={car.rays} />

      <div>
        <div className="mb-0.5 text-xs font-semibold text-muted-foreground">Score-Zusammensetzung</div>
        {barData.length === 0 ? (
          <p className="text-xs text-muted-foreground">Noch keine Belohnung.</p>
        ) : (
          <div className="h-[150px] w-full">
            <ResponsiveBar
              data={barData}
              theme={nivoDark}
              keys={["value"]}
              indexBy="part"
              layout="horizontal"
              margin={{ top: 4, right: 10, bottom: 20, left: 64 }}
              padding={0.25}
              colors={({ data }) => (data.value >= 0 ? "#22c55e" : "#ef4444")}
              enableGridY={false}
              axisBottom={{ tickValues: 4 }}
              labelSkipWidth={16}
              valueScale={{ type: "linear" }}
            />
          </div>
        )}
      </div>

      {/* Net/Q (DQN) or Q-table heatmap (q-table mode) — streamed only while inspected. */}
      <div>
        <button
          className="flex w-full items-center justify-between text-xs font-semibold text-muted-foreground hover:text-foreground"
          onClick={() => setShowNet((s) => !s)}
        >
          <span>{isQtable ? "Q-Tabelle & Q-Werte" : "Neuronales Netz & Q-Werte"}</span>
          <span>{showNet ? "▲" : "▼"}</span>
        </button>
        {showNet && (
          <div className="mt-2 space-y-3">
            {isQtable ? (
              <>
                {inspect && inspect.index === index && <QValueChart insp={inspect} />}
                <QTableHeatmap table={qtableForThis} />
              </>
            ) : inspect && inspect.index === index ? (
              <>
                <QValueChart insp={inspect} />
                <NetView insp={inspect} />
              </>
            ) : (
              <p className="text-xs text-muted-foreground">Lade Netz-Daten…</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

const MAX_RAY_M = 80 // spiegelt environment.MAX_RAY_M
const RAY_LABELS = ["−75°", "−45°", "−20°", "0°", "+20°", "+45°", "+75°"]

/** Was das Auto „sieht": die 7 Lidar-Strahlen (Abstand zur nächsten Wand) als
 *  Balken, rot = Wand nah, grün = frei. Gleiche Daten wie die 3D-Strahlen. */
function Sensors({ rays }: { rays: number[] }) {
  if (!rays || rays.length === 0) return null
  return (
    <div>
      <div className="mb-0.5 text-xs font-semibold text-muted-foreground">Sensoren (Sicht)</div>
      <div className="space-y-0.5">
        {rays.map((r, i) => {
          const v = Math.min(1, Math.max(0, r))
          const hue = v * 120 // 0=rot (nah) … 120=grün (frei)
          return (
            <div key={i} className="flex items-center gap-2 text-xs tabular-nums">
              <span className="w-9 shrink-0 text-muted-foreground">{RAY_LABELS[i] ?? i}</span>
              <div className="h-2 flex-1 overflow-hidden rounded bg-black/40">
                <div className="h-full rounded" style={{ width: `${v * 100}%`, backgroundColor: `hsl(${hue} 90% 50%)` }} />
              </div>
              <span className="w-10 shrink-0 text-right">{(v * MAX_RAY_M).toFixed(0)} m</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

/** Top-down mini view of the forces acting on the car (car points up). */
function Forces({ aLong, aLat }: { aLong: number; aLat: number }) {
  const S = 110
  const cx = S / 2
  const cy = S / 2
  const scale = 1.0 // px per m/s²  (GRIP_MAX≈45 → ~45px)
  // Up = forward accel; down = braking. Left = turning left (a_lat > 0).
  const longY = cy - aLong * scale
  const latX = cx - aLat * scale
  const gLong = (aLong / G).toFixed(1)
  const gLat = (Math.abs(aLat) / G).toFixed(1)
  const gTot = (Math.hypot(aLong, aLat) / G).toFixed(1)

  return (
    <div className="flex items-center gap-3">
      <svg width={S} height={S} className="shrink-0 rounded bg-black/30">
        {/* grip-circle reference (≈4.6 g) */}
        <circle cx={cx} cy={cy} r={45} fill="none" stroke="#334155" strokeDasharray="3 3" />
        <line x1={cx} y1={4} x2={cx} y2={S - 4} stroke="#1e293b" />
        <line x1={4} y1={cy} x2={S - 4} y2={cy} stroke="#1e293b" />
        {/* car body, pointing up */}
        <rect x={cx - 5} y={cy - 10} width={10} height={20} rx={2} fill="#e2e8f0" />
        {/* longitudinal force (accel green up / brake red down) */}
        <line x1={cx} y1={cy} x2={cx} y2={longY} stroke={aLong >= 0 ? "#22c55e" : "#ef4444"} strokeWidth={2.5} />
        {/* lateral force */}
        <line x1={cx} y1={cy} x2={latX} y2={cy} stroke="#38bdf8" strokeWidth={2.5} />
      </svg>
      <div className="space-y-0.5 text-xs tabular-nums">
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Längs</span>
          <span className={aLong >= 0 ? "text-emerald-400" : "text-red-400"}>{gLong} g</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Quer</span>
          <span className="text-sky-400">{gLat} g</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Gesamt</span>
          <span className="font-semibold">{gTot} g</span>
        </div>
      </div>
    </div>
  )
}
