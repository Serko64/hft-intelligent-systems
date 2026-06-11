// Mirror of the JSON messages the FastAPI backend sends (see f1_rl/server/protocol.py).

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
  pack: number
  score: number
  reward_parts?: Record<string, number>
  generation: number
  a_long: number // longitudinal accel m/s² (+ = accelerating, − = braking)
  a_lat: number // lateral/cornering accel m/s² (signed)
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
  n_packs: number
}

export interface StatusMsg {
  type: "status"
  state: string // "idle" | "training" | "driving"
  message?: string
}

// The best lap's path. points = [x, y, speed, throttle] in metres / m·s⁻¹ / −1..1.
export interface RacingLineMsg {
  type: "racing_line"
  points: [number, number, number, number][]
  vmin: number
  vmax: number
}

// One car's neural-net state for the net / Q-value visualisation.
export interface InspectMsg {
  type: "inspect"
  index: number
  obs: number[] // 14 inputs
  q: number[] // 20 Q-values (the DQN "table" for this state)
  hidden: number[][] // per-layer post-ReLU activations
  action: number // greedy (chosen) action index
}

// The inspected car's Q-table for the heatmap (q-table backend only). Slim payload:
// values[i] is one row's N_ACTIONS Q-values (capped/down-sampled server-side);
// n_states is the true total state count (for the caption, may exceed values.length).
export interface QTableMsg {
  type: "qtable"
  index: number
  n_states: number
  values: number[][]
}

export type ServerMsg = TrackMsg | FrameMsg | StatsMsg | StatusMsg | RacingLineMsg | InspectMsg | QTableMsg

// Commands the client sends back.
export type ClientMsg =
  | {
      type: "start_training"
      circuit: string
      steps_per_gen: number
      total_gens: number
      evolution_mode: "classic" | "pack" | "qtable" | "qtable_pack"
      resume: boolean
      use_rays: boolean
      auto_speed: boolean
    }
  | { type: "load_and_drive"; circuit: string; use_rays: boolean; backend: "dqn" | "qtable" }
  | { type: "set_speed"; value: number }
  | { type: "inspect_car"; index: number | null }
  | { type: "stop" }
