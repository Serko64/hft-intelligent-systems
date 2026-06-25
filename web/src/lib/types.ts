// Spiegel der JSON-Nachrichten, die das Backend schickt (siehe f1_rl/server/protocol.py).

export type Vec2 = [number, number]

export interface PolyRings {
  exterior: Vec2[]
  interiors: Vec2[][]
}

export interface TrackZone {
  name: string // "barrier" | "gravel" | "runoff" | "kerb"
  polygons: PolyRings[]
}

export interface TrackMsg {
  type: "track"
  name: string
  total_length_m: number
  half_width_m: number
  centerline: Vec2[]
  corridor_exterior: Vec2[]
  corridor_interiors: Vec2[][]
  zones?: TrackZone[]
  bounds: { minx: number; miny: number; maxx: number; maxy: number }
}

export interface Car {
  x: number
  y: number
  heading: number
  speed: number
  throttle: number
  checkpoint: number
  progress: number
  lap: number
  rays: number[]
  score: number
  reward_parts?: Record<string, number>
  generation: number
  a_long: number // Längsbeschleunigung m/s² (positiv = beschleunigen, negativ = bremsen)
  a_lat: number // Quer- bzw. Kurvenbeschleunigung m/s² (vorzeichenbehaftet)
}

export interface FrameMsg {
  type: "frame"
  cars: Car[]
}

export interface StatsMsg {
  type: "stats"
  timesteps: number
  generation: number
  epsilon: number
  best_fitness: number
  mean_fitness: number
  top_scores: [number, number][] // Scoreboard: [Score, Generation], best first
}

export interface StatusMsg {
  type: "status"
  state: string // "idle" | "training" | "driving"
  message?: string
}

// Der Pfad der besten Runde. points = [x, y, Tempo, Gas] in Metern, m/s und -1..1.
export interface RacingLineMsg {
  type: "racing_line"
  points: [number, number, number, number][]
  vmin: number
  vmax: number
}

// Der Netz-Zustand eines Autos für die Netz- bzw. Q-Wert-Visualisierung.
export interface InspectMsg {
  type: "inspect"
  index: number
  obs: number[] // 14 Eingaben
  q: number[] // 20 Q-Werte (die DQN-„Tabelle" für diesen Zustand)
  hidden: number[][] // Aktivierungen je Schicht nach ReLU
  action: number // Index der greedy gewählten Aktion
}

// Die Q-Tabelle des inspizierten Autos für die Heatmap (nur Q-Table-Backend). Schlanke
// Nutzlast: values[i] sind die N_ACTIONS Q-Werte einer Zeile (serverseitig gedeckelt bzw.
// heruntergerechnet). n_states ist die echte Gesamtzahl der Zustände (für die Beschriftung,
// kann values.length übersteigen).
export interface QTableMsg {
  type: "qtable"
  index: number
  n_states: number
  values: number[][]
}

export type ServerMsg = TrackMsg | FrameMsg | StatsMsg | StatusMsg | RacingLineMsg | InspectMsg | QTableMsg

// Befehle, die der Client zurück ans Backend schickt.
export type ClientMsg =
  | {
      type: "start_training"
      circuit: string
      steps_per_gen: number
      total_gens: number
      evolution_mode: "classic" | "qtable"
      resume: boolean
      use_rays: boolean
      auto_speed: boolean
      multi_start_eval: boolean
      use_crossover: boolean
    }
  | { type: "load_and_drive"; circuit: string; use_rays: boolean; backend: "dqn" | "qtable" }
  | { type: "set_speed"; value: number }
  | { type: "inspect_car"; index: number | null }
  | { type: "stop" }
