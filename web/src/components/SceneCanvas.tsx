import { memo, useEffect, useRef, type MutableRefObject } from "react"
import type { Car, RacingLineMsg, TrackMsg } from "@/lib/types"
import { TrackScene, type ColorMode } from "@/three/TrackScene"

interface Props {
  track: TrackMsg | null
  carsRef: MutableRefObject<Car[]>
  racingLine: RacingLineMsg | null
  colorMode: ColorMode
  selectedCar: number | null
  onSelectCar: (i: number | null) => void
}

/** Beheimatet die three.js-TrackScene und versorgt sie mit den Live-Autodaten und der Streckengeometrie. */
function SceneCanvasImpl({
  track, carsRef, racingLine, colorMode, selectedCar, onSelectCar,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<TrackScene | null>(null)

  // Szene einmalig erzeugen.
  useEffect(() => {
    if (!canvasRef.current) return
    const scene = new TrackScene(canvasRef.current, () => carsRef.current)
    scene.setOnCarSelect(onSelectCar)
    sceneRef.current = scene
    return () => scene.dispose()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [carsRef])

  // Den Auswahl-Callback aktuell halten, ohne die Szene neu zu bauen.
  useEffect(() => {
    sceneRef.current?.setOnCarSelect(onSelectCar)
  }, [onSelectCar])

  // Farbmodus (Rang oder Generation) und die von außen gesteuerte Auswahl-Hervorhebung.
  useEffect(() => {
    sceneRef.current?.setColorMode(colorMode)
  }, [colorMode])

  useEffect(() => {
    sceneRef.current?.setSelected(selectedCar)
  }, [selectedCar])

  // Neue Streckengeometrie übergeben, sobald sie sich ändert.
  useEffect(() => {
    if (track && sceneRef.current) sceneRef.current.setTrack(track)
  }, [track])

  // Die Racing-Line der besten Runde zeichnen oder entfernen.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return
    if (racingLine) scene.setRacingLine(racingLine.points, racingLine.vmin, racingLine.vmax)
    else scene.clearRacingLine()
  }, [racingLine])

  return <canvas ref={canvasRef} className="h-full w-full block" />
}

export const SceneCanvas = memo(SceneCanvasImpl)
