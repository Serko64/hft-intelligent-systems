import { memo, useEffect, useRef, type MutableRefObject } from "react"
import type { Car, RacingLineMsg, TrackMsg } from "@/lib/types"
import { TrackScene, type CarStyle, type ColorMode } from "@/three/TrackScene"

interface Props {
  track: TrackMsg | null
  carsRef: MutableRefObject<Car[]>
  racingLine: RacingLineMsg | null
  carStyle: CarStyle
  colorMode: ColorMode
  selectedCar: number | null
  onSelectCar: (i: number | null) => void
}

/** Hosts the three.js TrackScene and feeds it the live car data + track geometry. */
function SceneCanvasImpl({
  track, carsRef, racingLine, carStyle, colorMode, selectedCar, onSelectCar,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<TrackScene | null>(null)

  // Create the scene once.
  useEffect(() => {
    if (!canvasRef.current) return
    const scene = new TrackScene(canvasRef.current, () => carsRef.current)
    scene.setOnCarSelect(onSelectCar)
    sceneRef.current = scene
    return () => scene.dispose()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [carsRef])

  // Keep the selection callback current without recreating the scene.
  useEffect(() => {
    sceneRef.current?.setOnCarSelect(onSelectCar)
  }, [onSelectCar])

  // Colour mode (rank vs generation) and externally-driven selection highlight.
  useEffect(() => {
    sceneRef.current?.setColorMode(colorMode)
  }, [colorMode])

  useEffect(() => {
    sceneRef.current?.setSelected(selectedCar)
  }, [selectedCar])

  // Push new track geometry whenever it changes.
  useEffect(() => {
    if (track && sceneRef.current) sceneRef.current.setTrack(track)
  }, [track])

  // Switch car bodies between full 3D model and a cheap box (performance).
  useEffect(() => {
    sceneRef.current?.setCarStyle(carStyle)
  }, [carStyle])

  // Draw (or clear) the best-lap racing line.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return
    if (racingLine) scene.setRacingLine(racingLine.points, racingLine.vmin, racingLine.vmax)
    else scene.clearRacingLine()
  }, [racingLine])

  return <canvas ref={canvasRef} className="h-full w-full block" />
}

export const SceneCanvas = memo(SceneCanvasImpl)
