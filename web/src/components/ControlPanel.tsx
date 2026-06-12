import { useEffect, useState } from "react"
import type { ClientMsg } from "@/lib/types"
import { api } from "@/lib/bridge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"

const STEPS_PRESETS = [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 50000, 100000, 250000, 500000, 1000000]
const GENS_PRESETS = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

interface Props {
  connected: boolean
  status: string
  statusMessage?: string
  onSend: (cmd: ClientMsg) => void
  onCircuitChange?: (name: string) => void
  colorByGen: boolean
  onColorByGenChange: (g: boolean) => void
  showCharts: boolean
  onShowChartsChange: (s: boolean) => void
}

export function ControlPanel({
  connected, status, statusMessage, onSend, onCircuitChange,
  colorByGen, onColorByGenChange, showCharts, onShowChartsChange,
}: Props) {
  const [circuits, setCircuits] = useState<string[]>([])
  const [circuit, setCircuit] = useState<string>("")
  const [stepsIdx, setStepsIdx] = useState(4) // 5000
  const [gensIdx, setGensIdx] = useState(5) // 200
  const [qtable, setQtable] = useState(false) // tabular Q-learning instead of the DQN
  const [useRays, setUseRays] = useState(true)
  const [speed, setSpeed] = useState(3) // live-view sub-steps per frame (matches SIM_SPEED_DEFAULT)
  const [autoSpeed, setAutoSpeed] = useState(true) // sync animation to generation compute time
  const [driveBackend, setDriveBackend] = useState<"dqn" | "qtable">("dqn") // welches Modell „Laden & Fahren" lädt

  const selectCircuit = (name: string) => {
    setCircuit(name)
    onCircuitChange?.(name)
  }

  useEffect(() => {
    api()
      .then((a) => a.list_circuits())
      .then((list: string[]) => {
        setCircuits(list)
        if (list.length) selectCircuit(list[0])
      })
      .catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const steps = STEPS_PRESETS[stepsIdx]
  const gens = GENS_PRESETS[gensIdx]
  const running = status === "training" || status === "driving"

  const startTraining = (resume: boolean) =>
    onSend({
      type: "start_training",
      circuit,
      steps_per_gen: steps,
      total_gens: gens,
      evolution_mode: qtable ? "qtable" : "classic",
      resume,
      use_rays: useRays,
      auto_speed: autoSpeed,
    })

  return (
    <Card className="w-[340px] shrink-0">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span>F1 RL Simulator</span>
          <span
            className={`text-xs font-normal rounded px-2 py-0.5 ${
              connected ? "bg-emerald-500/15 text-emerald-400" : "bg-red-500/15 text-red-400"
            }`}
          >
            {connected ? "verbunden" : "getrennt"}
          </span>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Circuit */}
        <div className="space-y-2">
          <Label>Strecke</Label>
          <Select value={circuit} onValueChange={selectCircuit}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Strecke wählen" />
            </SelectTrigger>
            <SelectContent>
              {circuits.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Steps / generations */}
        <div className="space-y-2">
          <Label className="flex justify-between">
            <span>Steps / Generation</span>
            <span className="text-muted-foreground tabular-nums">{steps.toLocaleString()}</span>
          </Label>
          <Slider
            min={0}
            max={STEPS_PRESETS.length - 1}
            step={1}
            value={[stepsIdx]}
            onValueChange={(v) => setStepsIdx(v[0])}
          />
        </div>
        <div className="space-y-2">
          <Label className="flex justify-between">
            <span>Generationen</span>
            <span className="text-muted-foreground tabular-nums">{gens.toLocaleString()}</span>
          </Label>
          <Slider
            min={0}
            max={GENS_PRESETS.length - 1}
            step={1}
            value={[gensIdx]}
            onValueChange={(v) => setGensIdx(v[0])}
          />
        </div>

        {/* Auto speed: sync the animation to how long a generation takes to compute */}
        <div className="flex items-center gap-2">
          <Checkbox id="autospeed" checked={autoSpeed} onCheckedChange={(v) => setAutoSpeed(Boolean(v))} />
          <Label htmlFor="autospeed" className="cursor-pointer">
            Auto-Geschwindigkeit (Generation synchron)
          </Label>
        </div>

        {/* Live-view speed (sub-steps per frame) — manual, disabled while auto is on */}
        <div className="space-y-2">
          <Label className="flex justify-between">
            <span className={autoSpeed ? "text-muted-foreground/50" : ""}>Geschwindigkeit (live)</span>
            <span className="text-muted-foreground tabular-nums">{autoSpeed ? "auto" : `${speed}×`}</span>
          </Label>
          <Slider
            min={1}
            max={20}
            step={1}
            value={[speed]}
            disabled={autoSpeed}
            onValueChange={(v) => {
              setSpeed(v[0])
              onSend({ type: "set_speed", value: v[0] })
            }}
          />
        </div>

        {/* Colour cars by generation instead of rank */}
        <div className="flex items-center gap-2">
          <Checkbox id="colorgen" checked={colorByGen} onCheckedChange={(v) => onColorByGenChange(Boolean(v))} />
          <Label htmlFor="colorgen" className="cursor-pointer">
            Nach Generation einfärben
          </Label>
        </div>

        {/* Show / hide the chart overlays */}
        <div className="flex items-center gap-2">
          <Checkbox id="charts" checked={showCharts} onCheckedChange={(v) => onShowChartsChange(Boolean(v))} />
          <Label htmlFor="charts" className="cursor-pointer">
            Charts anzeigen
          </Label>
        </div>

        {/* Q-table backend: classic tabular Q-learning, no neural network */}
        <div className="flex items-center gap-2">
          <Checkbox id="qtable" checked={qtable} onCheckedChange={(v) => setQtable(Boolean(v))} />
          <Label htmlFor="qtable" className="cursor-pointer">
            Q-Table (kein neuronales Netz)
          </Label>
        </div>

        {/* Sensor rays */}
        <div className="flex items-center gap-2">
          <Checkbox id="rays" checked={useRays} onCheckedChange={(v) => setUseRays(Boolean(v))} />
          <Label htmlFor="rays" className="cursor-pointer">
            Sensor-Strahlen (Rays) nutzen
          </Label>
        </div>

        <Separator />

        {/* Fahrmodell: welches gespeicherte Modell „Laden & Fahren" fährt */}
        <div className="space-y-2">
          <Label>Fahrmodell (Laden &amp; Fahren)</Label>
          <Select value={driveBackend} onValueChange={(v) => setDriveBackend(v as "dqn" | "qtable")}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="dqn">DQN (neuronales Netz)</SelectItem>
              <SelectItem value="qtable">Q-Table</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Actions */}
        <div className="grid grid-cols-2 gap-2">
          <Button className="col-span-2" disabled={!connected || !circuit} onClick={() => startTraining(false)}>
            ▶ Training starten
          </Button>
          <Button variant="secondary" disabled={!connected || !circuit} onClick={() => startTraining(true)}>
            Fortsetzen
          </Button>
          <Button variant="secondary" disabled={!connected || !circuit} onClick={() => onSend({ type: "load_and_drive", circuit, use_rays: useRays, backend: driveBackend })}>
            Laden & Fahren
          </Button>
          <Button className="col-span-2" variant="destructive" disabled={!running} onClick={() => onSend({ type: "stop" })}>
            Stopp
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">
          Status: <span className="text-foreground">{status}</span>
          {statusMessage ? ` — ${statusMessage}` : ""}
        </p>
      </CardContent>
    </Card>
  )
}
