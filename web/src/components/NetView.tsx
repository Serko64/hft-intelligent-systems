import { ResponsiveBar } from "@nivo/bar"
import type { InspectMsg } from "@/lib/types"
import { nivoDark } from "@/lib/nivoTheme"
import { actionLabel } from "@/lib/actions"

const OBS_LABELS = [
  "x", "y", "hdg", "spd", "prog", "distL", "distR",
  "r-75", "r-45", "r-20", "r0", "r+20", "r+45", "r+75",
]

/** Divergierende Farbe: negativ blau, um 0 grau, positiv rot. */
function diverging(v: number, scale: number): string {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)))
  if (t >= 0) {
    const g = Math.round(120 - 80 * t)
    return `rgb(${Math.round(120 + 135 * t)},${g},${g})`
  }
  const a = -t
  return `rgb(${Math.round(120 - 70 * a)},${Math.round(120 - 20 * a)},${Math.round(120 + 135 * a)})`
}

/** Betragsfarbe für ReLU-Aktivierungen (≥0): von dunklem Schiefer zu hellem Cyan. */
function magnitude(v: number, max: number): string {
  const t = Math.max(0, Math.min(1, v / (max || 1)))
  return `rgb(${Math.round(20 + 36 * t)},${Math.round(30 + 178 * t)},${Math.round(40 + 195 * t)})`
}

function maxAbs(a: number[]): number {
  return a.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6)
}

/** Eine Reihe Zellen (eine je Neuron), eingefärbt nach dem Aktivierungswert. */
function Layer({ title, values, color, labels, highlight }: {
  title: string
  values: number[]
  color: (v: number) => string
  labels?: string[]
  highlight?: number
}) {
  const n = values.length
  const W = 280
  const cw = W / n
  const h = labels ? 16 : 12
  return (
    <div>
      <div className="mb-0.5 text-[10px] text-muted-foreground">{title} ({n})</div>
      <svg width={W} height={h + (labels ? 10 : 0)} className="block">
        {values.map((v, i) => (
          <rect
            key={i}
            x={i * cw}
            y={0}
            width={Math.max(1, cw - 0.5)}
            height={h}
            fill={color(v)}
            stroke={i === highlight ? "#39d0ff" : undefined}
            strokeWidth={i === highlight ? 2 : 0}
          >
            <title>{labels ? `${labels[i]}: ${v.toFixed(2)}` : `#${i}: ${v.toFixed(2)}`}</title>
          </rect>
        ))}
        {labels &&
          n <= 24 &&
          labels.map((l, i) => (
            <text key={i} x={i * cw + cw / 2} y={h + 8} fontSize={6} fill="#94a3b8" textAnchor="middle">
              {l}
            </text>
          ))}
      </svg>
    </div>
  )
}

/** Aktivierungskarte des ganzen Netzes für den aktuellen Zustand des inspizierten Autos. */
export function NetView({ insp }: { insp: InspectMsg }) {
  const obsScale = maxAbs(insp.obs)
  const qScale = maxAbs(insp.q)
  const h1 = insp.hidden[0] ?? []
  const h2 = insp.hidden[1] ?? []
  const h1max = maxAbs(h1)
  const h2max = maxAbs(h2)

  return (
    <div className="space-y-1.5">
      <Layer title="Eingabe" values={insp.obs} color={(v) => diverging(v, obsScale)} labels={OBS_LABELS} />
      <Layer title="Hidden 1" values={h1} color={(v) => magnitude(v, h1max)} />
      <Layer title="Hidden 2" values={h2} color={(v) => magnitude(v, h2max)} />
      <Layer
        title="Ausgabe (Q je Aktion)"
        values={insp.q}
        color={(v) => diverging(v, qScale)}
        highlight={insp.action}
      />
      <p className="text-[10px] text-muted-foreground">
        Eingabe/Q: blau −, rot +. Hidden: hell = aktiv. Blau umrandet = gewählte Aktion.
      </p>
    </div>
  )
}

/** Q-Werte als Balkendiagramm, die gewählte (greedy) Aktion ist hervorgehoben. */
export function QValueChart({ insp }: { insp: InspectMsg }) {
  const data = insp.q.map((v, i) => ({ action: String(i), q: +v.toFixed(2) }))
  return (
    <div>
      <div className="mb-0.5 text-xs font-semibold text-muted-foreground">
        Q-Werte — gewählt: {actionLabel(insp.action)}
      </div>
      <div className="h-[150px] w-full">
        <ResponsiveBar
          data={data}
          theme={nivoDark}
          keys={["q"]}
          indexBy="action"
          margin={{ top: 4, right: 8, bottom: 22, left: 36 }}
          padding={0.2}
          colors={({ data }) =>
            Number(data.action) === insp.action ? "#facc15" : Number(data.q) >= 0 ? "#38bdf8" : "#64748b"
          }
          enableLabel={false}
          axisBottom={{ tickValues: 5, legend: "Aktion", legendOffset: 18, legendPosition: "middle" }}
          axisLeft={{ tickValues: 4 }}
          valueScale={{ type: "linear" }}
        />
      </div>
    </div>
  )
}
