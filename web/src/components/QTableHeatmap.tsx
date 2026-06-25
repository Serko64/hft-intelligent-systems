import { useEffect, useRef } from "react"
import type { QTableMsg } from "@/lib/types"

/** Divergierende Farbe: negativ blau, um 0 grau, positiv rot. (Spiegelt NetView.) */
function diverging(v: number, scale: number): string {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)))
  if (t >= 0) {
    const g = Math.round(120 - 80 * t)
    return `rgb(${Math.round(120 + 135 * t)},${g},${g})`
  }
  const a = -t
  return `rgb(${Math.round(120 - 70 * a)},${Math.round(120 - 20 * a)},${Math.round(120 + 135 * a)})`
}

const CANVAS_W = 280
const CANVAS_H = 320

/** Die komplette gelernte Q-Tabelle des inspizierten Autos als Heatmap.
 *  Eine Zeile je Zustand, eine Spalte je Aktion, Farbe = Q-Wert. Auf ein <canvas>
 *  gezeichnet, damit es auch bei tausenden Zuständen flott bleibt (die Tabelle wächst,
 *  während der Agent erkundet). Die Zeilen füllen die feste Höhe anteilig, die Heatmap
 *  „füllt sich" also über die Generationen sichtbar auf. */
export function QTableHeatmap({ table }: { table: QTableMsg }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { values } = table
  const nStates = table.n_states          // echte Zahl gelernter Zustände (für die Beschriftung)
  const nRows = values.length             // tatsächlich gesendete Zeilen (gedeckelt bzw. heruntergerechnet)
  const nActions = values[0]?.length ?? 20

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext("2d")
    if (!ctx) return
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
    if (nRows === 0) return

    // Globaler Betrag, damit die divergierende Skala über alle Zellen vergleichbar ist.
    let scale = 1e-6
    for (const row of values) for (const v of row) scale = Math.max(scale, Math.abs(v))

    const cellW = CANVAS_W / nActions
    const rowH = Math.max(1, CANVAS_H / nRows)
    for (let i = 0; i < nRows; i++) {
      const row = values[i]
      const y = (i * CANVAS_H) / nRows
      for (let j = 0; j < nActions; j++) {
        ctx.fillStyle = diverging(row[j], scale)
        ctx.fillRect(j * cellW, y, Math.max(1, cellW), Math.max(1, rowH))
      }
    }
  }, [values, nRows, nActions])

  return (
    <div>
      <div className="mb-0.5 text-xs font-semibold text-muted-foreground">
        Volle Q-Tabelle — {nStates.toLocaleString()} Zustände × {nActions} Aktionen
        {nRows < nStates ? ` (Anzeige: ${nRows})` : ""}
      </div>
      {nRows === 0 ? (
        <p className="text-[10px] text-muted-foreground">Noch keine Zustände gelernt…</p>
      ) : (
        <>
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            className="block w-full rounded bg-black/30"
          />
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            Zeile = Zustand, Spalte = Aktion (0–{nActions - 1}). Farbe: blau −, grau 0, rot + (Q-Wert).
          </p>
        </>
      )}
    </div>
  )
}
