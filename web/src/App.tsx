import { useCallback, useState } from "react"
import { ControlPanel } from "@/components/ControlPanel"
import { Hud } from "@/components/Hud"
import { SceneCanvas } from "@/components/SceneCanvas"
import { useSimSocket } from "@/lib/useSimSocket"
import type { TrackMsg } from "@/lib/types"

export default function App() {
  const { connected, status, statusMessage, track, stats, carsRef, send } = useSimSocket()

  // Preview the selected circuit before training starts. The live WS track
  // (sent when training/driving begins) takes precedence over the preview.
  const [preview, setPreview] = useState<TrackMsg | null>(null)
  const onCircuitChange = useCallback((name: string) => {
    fetch(`/api/track?name=${encodeURIComponent(name)}`)
      .then((r) => r.json())
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
      />

      <main className="flex-1 overflow-hidden rounded-xl border border-border bg-black/40">
        <SceneCanvas track={track ?? preview} carsRef={carsRef} />
      </main>

      <Hud carsRef={carsRef} stats={stats} />
    </div>
  )
}
