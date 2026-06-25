import { useCallback, useEffect, useState } from "react"
import { ControlPanel } from "@/components/ControlPanel"
import { Hud } from "@/components/Hud"
import { SceneCanvas } from "@/components/SceneCanvas"
import { FitnessChart } from "@/components/FitnessChart"
import { RacingLineChart } from "@/components/RacingLineChart"
import { CarInspector } from "@/components/CarInspector"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useSimBridge } from "@/lib/useSimBridge"
import { api } from "@/lib/bridge"
import type { TrackMsg } from "@/lib/types"
import type { ColorMode } from "@/three/TrackScene"

export default function App() {
  const { connected, status, statusMessage, track, stats, statsHistory, racingLine, inspect, qtable, carsRef, send } =
    useSimBridge()

  // Autos nach Rang einfärben (bester bis schlechtester) oder nach ihrer Generation.
  const [colorMode, setColorMode] = useState<ColorMode>("rank")
  const [showCharts, setShowCharts] = useState(true)
  const [selectedCar, setSelectedCar] = useState<number | null>(null)

  // Dem Backend sagen, für welches Auto es Netz- und Q-Wert-Details streamen soll.
  useEffect(() => {
    send({ type: "inspect_car", index: selectedCar })
  }, [selectedCar, send])

  // Vorschau der gewählten Strecke, bevor das Training startet. Die Live-Strecke
  // (kommt mit Trainings- oder Fahrstart) hat Vorrang vor der Vorschau.
  const [preview, setPreview] = useState<TrackMsg | null>(null)
  const onCircuitChange = useCallback((name: string) => {
    api()
      .then((a) => a.get_track(name))
      .then((d) => setPreview({ type: "track", ...d }))
      .catch(() => undefined)
  }, [])

  return (
    <div className="dark flex h-screen w-screen gap-3 bg-background p-3 text-foreground">
      <ControlPanel
        connected={connected}
        status={status}
        statusMessage={statusMessage}
        onSend={send}
        onCircuitChange={onCircuitChange}
        colorByGen={colorMode === "generation"}
        onColorByGenChange={(g) => setColorMode(g ? "generation" : "rank")}
        showCharts={showCharts}
        onShowChartsChange={setShowCharts}
      />

      <main className="relative flex-1 overflow-hidden rounded-xl border border-border bg-black/40">
        <SceneCanvas
          track={track ?? preview}
          carsRef={carsRef}
          racingLine={racingLine}
          colorMode={colorMode}
          selectedCar={selectedCar}
          onSelectCar={setSelectedCar}
        />

        {/* Chart-Overlays: pointer-events nur auf den Karten, damit das Drehen weiter funktioniert. */}
        {showCharts && (
          <div className="pointer-events-none absolute inset-0 p-2">
            {statsHistory.length >= 2 && (
              <Card className="pointer-events-auto absolute bottom-2 left-2 w-[320px] bg-black/70 backdrop-blur">
                <CardHeader className="p-2 pb-0">
                  <CardTitle className="text-xs">Performance über Generationen</CardTitle>
                </CardHeader>
                <CardContent className="p-2">
                  <FitnessChart history={statsHistory} />
                </CardContent>
              </Card>
            )}

            {racingLine && (
              <Card className="pointer-events-auto absolute bottom-2 right-2 w-[340px] bg-black/70 backdrop-blur">
                <CardHeader className="p-2 pb-0">
                  <CardTitle className="text-xs">Beste Runde — Tempo & Gas/Bremse</CardTitle>
                </CardHeader>
                <CardContent className="p-2">
                  <RacingLineChart line={racingLine} />
                </CardContent>
              </Card>
            )}

            {selectedCar !== null && (
              <Card className="pointer-events-auto absolute right-2 top-2 max-h-[calc(100%-1rem)] w-[300px] overflow-y-auto bg-black/70 backdrop-blur">
                <CardContent className="p-3">
                  <CarInspector
                    carsRef={carsRef}
                    index={selectedCar}
                    inspect={inspect}
                    qtable={qtable}
                    onClose={() => setSelectedCar(null)}
                  />
                </CardContent>
              </Card>
            )}
          </div>
        )}

        <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 rounded-md bg-black/50 px-3 py-1 text-xs text-white/80">
          Linksklick: Auto wählen · Ziehen: drehen · Rechtsklick ziehen: verschieben · Mausrad: zoomen
        </div>
      </main>

      <Hud carsRef={carsRef} stats={stats} />
    </div>
  )
}
