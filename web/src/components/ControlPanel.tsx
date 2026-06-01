import { useEffect, useState } from "react"
import type { ClientMsg } from "@/lib/types"
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
}

export function ControlPanel({ connected, status, statusMessage, onSend, onCircuitChange }: Props) {
  const [circuits, setCircuits] = useState<string[]>([])
  const [circuit, setCircuit] = useState<string>("")
  const [stepsIdx, setStepsIdx] = useState(4) // 5000
  const [gensIdx, setGensIdx] = useState(5) // 200
  const [pack, setPack] = useState(false)

  const selectCircuit = (name: string) => {
    setCircuit(name)
    onCircuitChange?.(name)
  }

  useEffect(() => {
    fetch("/api/circuits")
      .then((r) => r.json())
      .then((d: { circuits: string[] }) => {
        setCircuits(d.circuits)
        if (d.circuits.length) selectCircuit(d.circuits[0])
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
      evolution_mode: pack ? "pack" : "classic",
      resume,
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

        {/* Pack mode */}
        <div className="flex items-center gap-2">
          <Checkbox id="pack" checked={pack} onCheckedChange={(v) => setPack(Boolean(v))} />
          <Label htmlFor="pack" className="cursor-pointer">
            Rudel-Evolution (Pack-Modus)
          </Label>
        </div>

        <Separator />

        {/* Actions */}
        <div className="grid grid-cols-2 gap-2">
          <Button className="col-span-2" disabled={!connected || !circuit} onClick={() => startTraining(false)}>
            ▶ Training starten
          </Button>
          <Button variant="secondary" disabled={!connected || !circuit} onClick={() => startTraining(true)}>
            Fortsetzen
          </Button>
          <Button variant="secondary" disabled={!connected || !circuit} onClick={() => onSend({ type: "load_and_drive", circuit })}>
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
