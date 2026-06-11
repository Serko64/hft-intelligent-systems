import { useCallback, useEffect, useRef, useState } from "react"
import type { Car, ClientMsg, InspectMsg, QTableMsg, RacingLineMsg, StatsMsg, TrackMsg } from "./types"
import { api } from "./bridge"

/**
 * Verbindet das UI mit dem Python-Backend über die pywebview-Bridge — der
 * Ersatz für useSimSocket (WebSocket). Gleiche Rückgabe-Signatur, damit App.tsx
 * und alle Komponenten unverändert bleiben.
 *
 * Frames kommen ~60×/s per poll(). Bei jedem Frame React neu zu rendern wäre
 * viel zu teuer, deshalb landet das aktuelle Auto-Array in einem *ref*
 * (`carsRef`), das die three.js-Renderschleife direkt liest. Nur
 * niederfrequente Daten (Status, Strecke, Statistik) liegen in React-State.
 */
export function useSimBridge() {
  const carsRef = useRef<Car[]>([])

  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState<string>("idle")
  const [statusMessage, setStatusMessage] = useState<string | undefined>()
  const [track, setTrack] = useState<TrackMsg | null>(null)
  const [stats, setStats] = useState<StatsMsg | null>(null)
  const [statsHistory, setStatsHistory] = useState<StatsMsg[]>([])
  const [racingLine, setRacingLine] = useState<RacingLineMsg | null>(null)
  const [inspect, setInspect] = useState<InspectMsg | null>(null)
  const [qtable, setQtable] = useState<QTableMsg | null>(null)

  // Letzter bekannter Status, ohne die Poll-Schleife an React-State zu koppeln.
  const statusRef = useRef(status)
  statusRef.current = status

  // Backend ~60×/s pollen. Ein neuer Tick wird erst nach dem vorigen geplant
  // (kein überlappendes Polling), und Frames gehen direkt in den ref.
  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>

    const loop = async (a: Awaited<ReturnType<typeof api>>) => {
      if (stopped) return
      try {
        const msg = await a.poll()
        if (msg.frame !== undefined) carsRef.current = msg.frame
        if (msg.stats !== undefined) {
          const s = msg.stats
          setStats(s)
          // Ein Punkt pro Generation für den Verlaufs-Chart.
          setStatsHistory((h) => {
            const last = h[h.length - 1]
            if (last && last.generation === s.generation) return h
            const next = [...h, s]
            return next.length > 2000 ? next.slice(next.length - 2000) : next
          })
        }
        if (msg.racing_line !== undefined) setRacingLine(msg.racing_line)
        if (msg.inspect !== undefined) setInspect(msg.inspect)
        if (msg.qtable !== undefined) setQtable(msg.qtable)
        if (msg.status !== statusRef.current) {
          statusRef.current = msg.status
          setStatus(msg.status)
          // Nichts läuft -> Autos leeren, damit die Szene nicht auf dem letzten
          // Frame einfriert (nach Stopp / Trainingsende).
          if (msg.status === "idle") {
            carsRef.current = []
            setInspect(null)
            setQtable(null)
          }
        }
      } catch {
        // Ein verworfener Poll ist harmlos — beim nächsten Tick erneut versuchen.
      }
      timer = setTimeout(() => loop(a), 1000 / 60)
    }

    api().then((a) => {
      setConnected(true)
      loop(a)
    })

    return () => {
      stopped = true
      clearTimeout(timer)
    }
  }, [])

  const send = useCallback((cmd: ClientMsg) => {
    api().then(async (a) => {
      switch (cmd.type) {
        case "start_training": {
          const t = await a.start_training(cmd)
          if (t && !t.error) {
            setTrack({ type: "track", ...t })
            setRacingLine(null) // neue Strecke -> alte Racing-Line + Historie ungültig
            setStatsHistory([])
            setStatusMessage(undefined)
          }
          break
        }
        case "load_and_drive": {
          const t = await a.load_and_drive(cmd.circuit, cmd.use_rays, cmd.backend)
          if (t?.error) {
            setStatus("idle")
            setStatusMessage(t.error)
          } else if (t) {
            setTrack({ type: "track", ...t })
            setRacingLine(null)
            setStatsHistory([])
            setStatusMessage(undefined)
          }
          break
        }
        case "set_speed":
          a.set_speed(cmd.value)
          break
        case "inspect_car":
          a.inspect_car(cmd.index)
          break
        case "stop":
          a.stop()
          break
      }
    })
  }, [])

  return { connected, status, statusMessage, track, stats, statsHistory, racingLine, inspect, qtable, carsRef, send }
}
