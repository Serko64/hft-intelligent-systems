import { useCallback, useEffect, useRef, useState } from "react"
import type { Car, ClientMsg, InspectMsg, QTableMsg, RacingLineMsg, ServerMsg, StatsMsg, TrackMsg } from "./types"

/**
 * Verwaltet die WebSocket-Verbindung zum Backend. Inzwischen durch useSimBridge
 * (pywebview) abgelöst, hier nur noch als Referenz.
 *
 * Frames kommen ~60-mal pro Sekunde. Bei jedem Frame React neu zu rendern wäre
 * viel zu teuer, deshalb liegt das aktuelle Auto-Array in einem *ref* (`carsRef`),
 * den die three.js-Renderschleife direkt liest. Nur niederfrequente Daten
 * (Verbindungsstatus, Streckengeometrie, Statistik je Generation) liegen in React-State.
 */
export function useSimSocket() {
  const wsRef = useRef<WebSocket | null>(null)
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

  useEffect(() => {
    let closed = false
    let retry: ReturnType<typeof setTimeout>

    const connect = () => {
      const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as ServerMsg
        switch (msg.type) {
          case "frame":
            carsRef.current = msg.cars
            break
          case "track":
            setTrack(msg)
            setRacingLine(null) // neue Strecke macht die alte Racing-Line ungültig
            setStatsHistory([]) // und ebenso die alte Performance-Historie
            break
          case "stats":
            setStats(msg)
            // Einen Punkt je Generation für den Performance-Verlauf behalten.
            setStatsHistory((h) => {
              const last = h[h.length - 1]
              if (last && last.generation === msg.generation) return h
              const next = [...h, msg]
              return next.length > 2000 ? next.slice(next.length - 2000) : next
            })
            break
          case "racing_line":
            setRacingLine(msg)
            break
          case "inspect":
            setInspect(msg)
            break
          case "qtable":
            setQtable(msg)
            break
          case "status":
            setStatus(msg.state)
            setStatusMessage(msg.message)
            // Nichts läuft, also Autos leeren, damit die Szene nicht auf dem letzten
            // Frame einfriert (nach Stopp oder Trainingsende).
            if (msg.state === "idle") {
              carsRef.current = []
              setInspect(null)
              setQtable(null)
            }
            break
        }
      }
    }

    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      wsRef.current?.close()
    }
  }, [])

  const send = useCallback((cmd: ClientMsg) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
  }, [])

  return { connected, status, statusMessage, track, stats, statsHistory, racingLine, inspect, qtable, carsRef, send }
}
