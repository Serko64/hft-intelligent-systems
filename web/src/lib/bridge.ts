// Direkte BRücke zum Python-Backend (pywebview). Ersetzt den alten WebSocket:
// JS ruft Python-Methoden über window.pywebview.api.* auf und zieht die Frames
// per poll(). Kein Server, kein Netzwerk — alles ein Prozess.

import type { Car, InspectMsg, QTableMsg, RacingLineMsg, StatsMsg, TrackMsg } from "./types"

// Rohform von track_to_dict (ohne das vom Frontend ergänzte `type`-Feld).
type TrackData = Omit<TrackMsg, "type">

// Was poll() pro Tick zurückgibt: `status` immer, alles andere nur bei neuen Daten.
export interface PollMsg {
  status: string
  frame?: Car[]
  stats?: StatsMsg
  racing_line?: RacingLineMsg
  inspect?: InspectMsg
  qtable?: QTableMsg
}

export interface PyApi {
  list_circuits: () => Promise<string[]>
  get_track: (name: string) => Promise<TrackData>
  start_training: (cfg: unknown) => Promise<TrackData & { error?: string }>
  load_and_drive: (circuit: string, use_rays: boolean, backend: string) => Promise<TrackData & { error?: string }>
  set_speed: (value: number) => Promise<void>
  inspect_car: (index: number | null) => Promise<void>
  stop: () => Promise<void>
  poll: () => Promise<PollMsg>
}

declare global {
  interface Window {
    pywebview?: { api: PyApi }
  }
}

let readyPromise: Promise<PyApi> | null = null

/** Löst auf, sobald pywebview die JS-API in die Seite injiziert hat. */
export function api(): Promise<PyApi> {
  if (readyPromise) return readyPromise
  readyPromise = new Promise((resolve) => {
    if (window.pywebview?.api) return resolve(window.pywebview.api)
    window.addEventListener("pywebviewready", () => resolve(window.pywebview!.api), {
      once: true,
    })
  })
  return readyPromise
}
