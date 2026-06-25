// Spiegel von DISCRETE_ACTIONS in f1_rl/simulation/environment.py, die 20 diskreten
// Kombinationen aus (Lenken, Gas), in genau der Reihenfolge, in der das Netz sie ausgibt.
const STEER = [-1.0, -0.5, 0.0, 0.5, 1.0]
const THROTTLE = [-1.0, 0.0, 0.5, 1.0]

export const DISCRETE_ACTIONS: [number, number][] = STEER.flatMap((s) =>
  THROTTLE.map((t) => [s, t] as [number, number]),
)

const steerLabel = (s: number) =>
  s < 0 ? `←${Math.abs(s)}` : s > 0 ? `→${s}` : "geradeaus"
const throttleLabel = (t: number) =>
  t < 0 ? "Bremse" : t === 0 ? "Schub 0" : `Gas ${t}`

/** Kurzes, lesbares Label für einen Aktionsindex, z. B. "→0.5 · Gas 1". */
export function actionLabel(i: number): string {
  const [s, t] = DISCRETE_ACTIONS[i] ?? [0, 0]
  return `${steerLabel(s)} · ${throttleLabel(t)}`
}
