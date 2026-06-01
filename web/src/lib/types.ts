// Mirror of the JSON messages the FastAPI backend sends (see f1_rl/server/protocol.py).

export type Vec2 = [number, number]

export interface TrackMsg {
  type: "track"
  name: string
  total_length_m: number
  half_width_m: number
  centerline: Vec2[]
  corridor_exterior: Vec2[]
  corridor_interiors: Vec2[][]
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
  top_scores: [number, number][]
}

export interface StatusMsg {
  type: "status"
  state: string // "idle" | "training" | "driving"
  message?: string
}

export type ServerMsg = TrackMsg | FrameMsg | StatsMsg | StatusMsg

// Commands the client sends back.
export type ClientMsg =
  | {
      type: "start_training"
      circuit: string
      steps_per_gen: number
      total_gens: number
      evolution_mode: "classic" | "pack"
      resume: boolean
    }
  | { type: "load_and_drive"; circuit: string }
  | { type: "stop" }
