import { useEffect, useRef, type MutableRefObject } from "react"
import type { Car, TrackMsg } from "@/lib/types"
import { TrackScene } from "@/three/TrackScene"

interface Props {
  track: TrackMsg | null
  carsRef: MutableRefObject<Car[]>
}

/** Hosts the three.js TrackScene and feeds it the live car data + track geometry. */
export function SceneCanvas({ track, carsRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<TrackScene | null>(null)

  // Create the scene once.
  useEffect(() => {
    if (!canvasRef.current) return
    const scene = new TrackScene(canvasRef.current, () => carsRef.current)
    sceneRef.current = scene
    return () => scene.dispose()
  }, [carsRef])

  // Push new track geometry whenever it changes.
  useEffect(() => {
    if (track && sceneRef.current) sceneRef.current.setTrack(track)
  }, [track])

  return <canvas ref={canvasRef} className="h-full w-full block" />
}
