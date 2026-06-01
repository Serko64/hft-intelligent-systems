import { useCallback, useEffect, useRef, useState } from "react"
import type { Car, ClientMsg, ServerMsg, StatsMsg, TrackMsg } from "./types"

/**
 * Manages the WebSocket connection to the backend.
 *
 * Frames arrive ~60×/s. Re-rendering React on every frame would be far too
 * expensive, so the latest car array is kept in a *ref* (`carsRef`) that the
 * three.js render loop reads directly. Only low-frequency data (connection
 * status, track geometry, per-generation stats) lives in React state.
 */
export function useSimSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const carsRef = useRef<Car[]>([])

  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState<string>("idle")
  const [statusMessage, setStatusMessage] = useState<string | undefined>()
  const [track, setTrack] = useState<TrackMsg | null>(null)
  const [stats, setStats] = useState<StatsMsg | null>(null)

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
            break
          case "stats":
            setStats(msg)
            break
          case "status":
            setStatus(msg.state)
            setStatusMessage(msg.message)
            // Nothing is running -> clear the cars so the scene doesn't freeze
            // on the last frame after a stop / when training finishes.
            if (msg.state === "idle") carsRef.current = []
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

  return { connected, status, statusMessage, track, stats, carsRef, send }
}
