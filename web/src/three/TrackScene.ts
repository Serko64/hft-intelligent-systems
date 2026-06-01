import * as THREE from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import type { Car, TrackMsg } from "@/lib/types"

// ── Tweakables ────────────────────────────────────────────────────────────────
const CAR_LENGTH_M = 8 // models are auto-scaled to roughly this length
export const MODEL_YAW_OFFSET = 0 // rotate the car model if it faces the wrong way (e.g. Math.PI/2)
const MODEL_PITCH = -Math.PI / 2 // lay a typical Y-up GLB flat into the XY ground plane
const MODEL_URL = "/models/f1.glb"

const GRASS = 0x12260c
const ASPHALT = 0x343338
const WALL = 0xe1e1e1
const CENTERLINE = 0xffd200

/** Rank → colour, mirroring the pygame view (best = gold ... worst = dark red). */
function rankColor(rank: number, total: number): THREE.Color {
  const f = total > 1 ? rank / (total - 1) : 0
  if (f < 0.08) return new THREE.Color(0xffd700)
  if (f < 0.2) return new THREE.Color(0x32dc50)
  if (f < 0.5) return new THREE.Color(0xc88232)
  return new THREE.Color(0x8a3a3a)
}

interface CarObj {
  group: THREE.Group
  marker: THREE.Mesh
  hasModel: boolean
}

/**
 * Renders the circuit and the live cars in a top-down orthographic view.
 * All coordinates are in metres (same as the simulation). The render loop reads
 * the latest cars via the `getCars` callback every frame, so React never needs
 * to re-render at 60 fps.
 */
export class TrackScene {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera: THREE.OrthographicCamera
  private trackGroup = new THREE.Group()
  private carPool: CarObj[] = []
  private carScale = 1 // cars are scaled up on big circuits so they stay visible
  private modelTemplate: THREE.Object3D | null = null
  private bounds = { minx: -50, miny: -50, maxx: 50, maxy: 50 }
  private raf = 0
  private resizeObs: ResizeObserver
  private canvas: HTMLCanvasElement
  private getCars: () => Car[]

  constructor(canvas: HTMLCanvasElement, getCars: () => Car[]) {
    this.canvas = canvas
    this.getCars = getCars
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.scene.background = new THREE.Color(GRASS)
    this.scene.add(this.trackGroup)

    this.camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.1, 2000)
    this.camera.position.set(0, 0, 500)
    this.camera.up.set(0, 1, 0)
    this.camera.lookAt(0, 0, 0)

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.1))
    const dir = new THREE.DirectionalLight(0xffffff, 1.4)
    dir.position.set(0.4, 0.8, 1)
    this.scene.add(dir)

    this.resizeObs = new ResizeObserver(() => this.resize())
    this.resizeObs.observe(canvas)
    this.resize()

    new GLTFLoader().load(
      MODEL_URL,
      (gltf) => {
        this.modelTemplate = this.normalizeModel(gltf.scene)
      },
      undefined,
      () => {
        // No model file yet — the cone fallback is used. This is expected.
        console.info("[scene] No car model at", MODEL_URL, "— using placeholder.")
      },
    )

    this.loop = this.loop.bind(this)
    this.raf = requestAnimationFrame(this.loop)
  }

  // ── Track geometry ────────────────────────────────────────────────────────
  setTrack(track: TrackMsg) {
    this.trackGroup.clear()
    this.bounds = track.bounds

    // Asphalt: corridor exterior with the infield rings as holes.
    const shape = new THREE.Shape(track.corridor_exterior.map(([x, y]) => new THREE.Vector2(x, y)))
    for (const ring of track.corridor_interiors) {
      shape.holes.push(new THREE.Path(ring.map(([x, y]) => new THREE.Vector2(x, y))))
    }
    const asphalt = new THREE.Mesh(
      new THREE.ShapeGeometry(shape),
      new THREE.MeshBasicMaterial({ color: ASPHALT }),
    )
    asphalt.position.z = -0.3
    this.trackGroup.add(asphalt)

    // White boundary lines (exterior + interiors).
    const wallMat = new THREE.LineBasicMaterial({ color: WALL })
    const rings = [track.corridor_exterior, ...track.corridor_interiors]
    for (const ring of rings) {
      const geo = new THREE.BufferGeometry().setFromPoints(
        ring.map(([x, y]) => new THREE.Vector3(x, y, -0.2)),
      )
      this.trackGroup.add(new THREE.LineLoop(geo, wallMat))
    }

    // Dashed yellow centerline.
    const clGeo = new THREE.BufferGeometry().setFromPoints(
      track.centerline.map(([x, y]) => new THREE.Vector3(x, y, -0.1)),
    )
    const clMat = new THREE.LineDashedMaterial({ color: CENTERLINE, dashSize: 6, gapSize: 6 })
    const cl = new THREE.Line(clGeo, clMat)
    cl.computeLineDistances()
    this.trackGroup.add(cl)

    // Scale cars relative to the circuit so they stay visible on huge tracks
    // (an 8 m car on a 7 km track would otherwise be sub-pixel).
    const span = Math.max(
      track.bounds.maxx - track.bounds.minx,
      track.bounds.maxy - track.bounds.miny,
    )
    this.carScale = Math.max(8, span * 0.02) / CAR_LENGTH_M
    for (const obj of this.carPool) obj.group.scale.setScalar(this.carScale)

    this.resize()
  }

  // ── Per-frame update ────────────────────────────────────────────────────────
  private loop() {
    const cars = this.getCars()
    this.syncPool(cars.length)
    for (let i = 0; i < this.carPool.length; i++) {
      const obj = this.carPool[i]
      if (i >= cars.length) {
        obj.group.visible = false
        continue
      }
      const c = cars[i]
      obj.group.visible = true
      obj.group.position.set(c.x, c.y, 0)
      obj.group.rotation.z = c.heading + (obj.hasModel ? MODEL_YAW_OFFSET : 0)
      ;(obj.marker.material as THREE.MeshBasicMaterial).color = rankColor(i, cars.length)
    }
    this.renderer.render(this.scene, this.camera)
    this.raf = requestAnimationFrame(this.loop)
  }

  private syncPool(n: number) {
    while (this.carPool.length < n) this.carPool.push(this.makeCar())
    // If the model finished loading after cars were created, upgrade the bodies.
    if (this.modelTemplate) {
      for (const obj of this.carPool) {
        if (!obj.hasModel) this.upgradeToModel(obj)
      }
    }
  }

  private makeCar(): CarObj {
    const group = new THREE.Group()

    // Coloured disc under the car so rank is readable regardless of the model.
    const marker = new THREE.Mesh(
      new THREE.CircleGeometry(3.5, 24),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    marker.position.z = 0.01
    group.add(marker)

    const obj: CarObj = { group, marker, hasModel: false }
    if (this.modelTemplate) {
      this.upgradeToModel(obj)
    } else {
      group.add(this.makeFallbackBody())
    }

    group.scale.setScalar(this.carScale)
    this.scene.add(group)
    return obj
  }

  private makeFallbackBody(): THREE.Mesh {
    // A cone pointing +X (heading 0).
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(2, CAR_LENGTH_M, 12),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.1, roughness: 0.7 }),
    )
    cone.rotation.z = -Math.PI / 2
    cone.position.z = 0.6
    cone.name = "body"
    return cone
  }

  private upgradeToModel(obj: CarObj) {
    if (!this.modelTemplate) return
    const old = obj.group.getObjectByName("body")
    if (old) obj.group.remove(old)
    const body = this.modelTemplate.clone(true)
    body.name = "body"
    obj.group.add(body)
    obj.hasModel = true
  }

  /** Center a loaded model, scale it to ~CAR_LENGTH_M, and lay it flat. */
  private normalizeModel(src: THREE.Object3D): THREE.Object3D {
    const wrapper = new THREE.Group()
    wrapper.add(src)
    const box = new THREE.Box3().setFromObject(src)
    const size = new THREE.Vector3()
    const center = new THREE.Vector3()
    box.getSize(size)
    box.getCenter(center)
    src.position.sub(center)
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const s = CAR_LENGTH_M / maxDim
    wrapper.scale.setScalar(s)
    wrapper.rotation.x = MODEL_PITCH
    return wrapper
  }

  // ── Camera framing ──────────────────────────────────────────────────────────
  private resize() {
    const w = this.canvas.clientWidth || 1
    const h = this.canvas.clientHeight || 1
    this.renderer.setSize(w, h, false)

    const pad = 1.08
    const cx = (this.bounds.minx + this.bounds.maxx) / 2
    const cy = (this.bounds.miny + this.bounds.maxy) / 2
    const worldW = (this.bounds.maxx - this.bounds.minx) * pad || 100
    const worldH = (this.bounds.maxy - this.bounds.miny) * pad || 100
    const aspect = w / h
    let halfW = worldW / 2
    let halfH = worldH / 2
    if (worldW / worldH > aspect) halfH = halfW / aspect
    else halfW = halfH * aspect

    this.camera.left = cx - halfW
    this.camera.right = cx + halfW
    this.camera.top = cy + halfH
    this.camera.bottom = cy - halfH
    this.camera.position.set(cx, cy, 500)
    this.camera.lookAt(cx, cy, 0)
    this.camera.updateProjectionMatrix()
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    this.resizeObs.disconnect()
    this.renderer.dispose()
  }
}
