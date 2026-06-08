import { useMemo, useRef, useState } from "react"
import { ResponsiveLine, type LineSeries } from "@nivo/line"
import { nivoDark } from "@/lib/nivoTheme"

interface Marker {
  axis: "x" | "y"
  value: number
}

interface Props {
  data: LineSeries[]
  colors: string[] | { datum: string }
  yMin?: number | "auto"
  yMax?: number | "auto"
  axisLeftLegend?: string
  axisBottomLegend?: string
  yTicks?: number | number[]
  markers?: Marker[]
  height?: number
  showLegend?: boolean
}

const MARGIN = { top: 8, right: 12, bottom: 28, left: 44 }

/** A nivo line chart with wheel-zoom + drag-pan on the X axis. Double-click resets.
 *  Works for any linearly-scaled x data (generation, time in seconds, …). */
export function ZoomableLine({
  data, colors, yMin = "auto", yMax = "auto", axisLeftLegend, axisBottomLegend,
  yTicks = 4, markers, height = 160, showLegend,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const drag = useRef<{ px: number; x0: number; x1: number } | null>(null)

  // Full extent of the data on X — the zoom window is clamped to this.
  const [full] = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const s of data) for (const p of s.data) {
      const x = Number(p.x)
      if (x < lo) lo = x
      if (x > hi) hi = x
    }
    if (!isFinite(lo)) { lo = 0; hi = 1 }
    if (lo === hi) hi = lo + 1
    return [{ lo, hi }]
  }, [data])

  const [win, setWin] = useState<{ lo: number; hi: number } | null>(null)
  const view = win ?? full

  // px → data-x, accounting for the plot margins.
  const plotW = () => (wrap.current?.clientWidth ?? 1) - MARGIN.left - MARGIN.right
  const dataXAt = (clientX: number) => {
    const rect = wrap.current!.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (clientX - rect.left - MARGIN.left) / plotW()))
    return view.lo + frac * (view.hi - view.lo)
  }

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const span = view.hi - view.lo
    const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2 // out / in
    const center = dataXAt(e.clientX)
    let lo = center - (center - view.lo) * factor
    let hi = center + (view.hi - center) * factor
    // Clamp to the full extent and a sane minimum span.
    lo = Math.max(full.lo, lo)
    hi = Math.min(full.hi, hi)
    if (hi - lo < (full.hi - full.lo) * 0.01) return
    if (hi - lo >= (full.hi - full.lo) - 1e-9) { setWin(null); return }
    if (span === hi - lo) return
    setWin({ lo, hi })
  }

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { px: e.clientX, x0: view.lo, x1: view.hi }
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const span = d.x1 - d.x0
    const dxData = ((e.clientX - d.px) / plotW()) * span
    let lo = d.x0 - dxData
    let hi = d.x1 - dxData
    if (lo < full.lo) { hi += full.lo - lo; lo = full.lo }
    if (hi > full.hi) { lo -= hi - full.hi; hi = full.hi }
    setWin({ lo, hi })
  }
  const onPointerUp = () => { drag.current = null }

  return (
    <div
      ref={wrap}
      style={{ height }}
      className="w-full cursor-ew-resize touch-none"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={() => setWin(null)}
      title="Mausrad: zoomen · ziehen: verschieben · Doppelklick: zurücksetzen"
    >
      <ResponsiveLine
        data={data}
        theme={nivoDark}
        colors={colors}
        margin={MARGIN}
        xScale={{ type: "linear", min: view.lo, max: view.hi }}
        yScale={{ type: "linear", min: yMin, max: yMax }}
        axisBottom={{ legend: axisBottomLegend, legendOffset: 22, legendPosition: "middle", tickValues: 5 }}
        axisLeft={{ legend: axisLeftLegend, legendOffset: -38, legendPosition: "middle", tickValues: yTicks }}
        enablePoints={false}
        enableGridX={false}
        curve="monotoneX"
        lineWidth={2}
        useMesh
        markers={markers?.map((m) => ({
          axis: m.axis,
          value: m.value,
          lineStyle: { stroke: "#64748b", strokeWidth: 1, strokeDasharray: "3 3" },
        }))}
        legends={
          showLegend
            ? [{ anchor: "top-left", direction: "row", translateY: -2, itemWidth: 50, itemHeight: 12, symbolSize: 8 }]
            : []
        }
      />
    </div>
  )
}
