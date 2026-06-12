import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import type { Car, PolyRings, TrackMsg, Vec2 } from "@/lib/types"

// ── Tweakables ────────────────────────────────────────────────────────────────
const CAR_LENGTH_M = 5 // real F1 length; matches the training bbox (env CAR_LENGTH_M)

// Sensor-Strahlen des angeklickten Autos (spiegelt environment.py: RAY_ANGLES / MAX_RAY_M).
const RAY_ANGLES_RAD = [-75, -45, -20, 0, 20, 45, 75].map((d) => (d * Math.PI) / 180)
const MAX_RAY_M = 80

const GRASS = 0x2f6b22 // lively green so the dark track + red kerb pop
const ASPHALT = 0x37352f // racing surface (FIA cross-section "Strecke")
const WALL = 0xf4f4ee // bright white track-limit line
const CENTERLINE = 0xffd200
const KERB_RED = 0xc0392b
const KERB_WHITE = 0xeeeeee

// Thin grey asphalt verge just outside the white line; the striped kerb is built
// separately on top of it as an extruded strip (see buildEdgeStrip).
const ZONE_STYLE: Record<string, { color: number; z: number }> = {
  runoff: { color: 0x6e6c66, z: -0.4 }, // grey asphalt verge
}

/** Rank → colour (hex), mirroring the pygame view (best = gold ... worst = dark red). */
function rankColorHex(rank: number, total: number): number {
  const f = total > 1 ? rank / (total - 1) : 0
  if (f < 0.08) return 0xffd700
  if (f < 0.2) return 0x32dc50
  if (f < 0.5) return 0xc88232
  return 0x8a3a3a
}

/** Generation → a stable distinct colour (golden-ratio hue hashing). */
function genColorHex(gen: number): number {
  const hue = (gen * 0.6180339887) % 1
  return new THREE.Color().setHSL(hue, 0.7, 0.55).getHex()
}

export type ColorMode = "rank" | "generation"

interface CarObj {
  group: THREE.Group
  bodyMaterial: THREE.MeshStandardMaterial // tinted per-frame by rank
  lastHex: number // zuletzt gesetzte Farbe — vermeidet überflüssige setHex-Aufrufe pro Frame
}

/**
 * Renders the circuit and the live cars in a freely-movable 3D view
 * (PerspectiveCamera + OrbitControls: left-drag orbit, right-drag pan, wheel
 * zoom). All coordinates are in metres (z is up). The render loop reads the
 * latest cars via `getCars` each frame, so React never re-renders at 60 fps.
 */
export class TrackScene {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private trackGroup = new THREE.Group()
  private racingLine: THREE.Mesh | null = null
  private carPool: CarObj[] = []
  private carScale = 1 // cars are scaled up on big circuits so they stay visible
  private colorMode: ColorMode = "rank"
  private selectedIndex: number | null = null
  private onCarSelect: ((i: number | null) => void) | null = null
  private selectionRing: THREE.Mesh
  private raysObj: THREE.LineSegments // Sensor-Strahlen des selektierten Autos
  private raycaster = new THREE.Raycaster()
  private pointerDown: { x: number; y: number } | null = null
  private raf = 0
  private resizeObs: ResizeObserver
  private canvas: HTMLCanvasElement
  private getCars: () => Car[]

  constructor(canvas: HTMLCanvasElement, getCars: () => Car[]) {
    this.canvas = canvas
    this.getCars = getCars
    // logarithmicDepthBuffer keeps depth precision usable across the huge
    // near→far range (a few metres up to kilometre-wide circuits), which is what
    // otherwise causes the stacked track layers to z-fight ("clip").
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.scene.background = new THREE.Color(GRASS)
    this.scene.add(this.trackGroup)

    // z-up perspective camera, controllable with the mouse.
    this.camera = new THREE.PerspectiveCamera(50, 1, 1, 500000)
    this.camera.up.set(0, 0, 1)
    this.camera.position.set(0, -200, 200)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12 // snappier than the old sluggish 0.08
    this.controls.minDistance = 3 // zoom right up to a single car
    this.controls.maxDistance = 200000
    // Zoom toward the mouse cursor instead of the (often far-off) target, so you
    // can dive straight onto a single car without the zoom "sticking" at min
    // distance from the track centre.
    this.controls.zoomToCursor = true
    this.controls.zoomSpeed = 1.6
    this.controls.panSpeed = 1.2
    this.controls.rotateSpeed = 0.9
    this.controls.screenSpacePanning = true // right-drag pans in view plane (intuitive)
    // Keep the camera above the ground and just shy of the exact top-down pole,
    // where orbiting otherwise flips/freezes (gimbal lock with a z-up camera).
    this.controls.minPolarAngle = 0.08
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05
    this.controls.target.set(0, 0, 0)

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x202018, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 1.4)
    dir.position.set(0.4, 0.6, 1)
    this.scene.add(dir)

    // Flat ring drawn under the selected car to highlight it.
    this.selectionRing = new THREE.Mesh(
      new THREE.RingGeometry(3.2, 4.2, 32),
      new THREE.MeshBasicMaterial({ color: 0x39d0ff, transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
    )
    this.selectionRing.visible = false
    this.selectionRing.renderOrder = 11
    this.scene.add(this.selectionRing)

    // Sensor-Strahlen (7 Linien) des angeklickten Autos. Geometrie wird EINMAL
    // angelegt und pro Frame nur befüllt (kein Neu-Erzeugen), siehe updateRays().
    const rayGeo = new THREE.BufferGeometry()
    rayGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(RAY_ANGLES_RAD.length * 2 * 3), 3))
    rayGeo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(RAY_ANGLES_RAD.length * 2 * 3), 3))
    this.raysObj = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true }))
    this.raysObj.visible = false
    this.raysObj.renderOrder = 12
    this.scene.add(this.raysObj)

    // Click a car to select it (only when the pointer barely moved, so orbiting
    // the camera doesn't trigger a selection).
    this.canvas.addEventListener("pointerdown", this.onPointerDown)
    this.canvas.addEventListener("pointerup", this.onPointerUp)

    this.resizeObs = new ResizeObserver(() => this.resize())
    this.resizeObs.observe(canvas)
    this.resize()

    this.loop = this.loop.bind(this)
    this.raf = requestAnimationFrame(this.loop)
  }

  // ── Track geometry ────────────────────────────────────────────────────────
  setTrack(track: TrackMsg) {
    this.trackGroup.clear()

    // Smoothed corridor rings (Chaikin) so jagged OSM polylines read as flowing
    // racetrack curves. The same smoothed rings drive the kerb + white line so
    // every layer lines up exactly.
    const exterior = this.smooth(track.corridor_exterior)
    const interiors = track.corridor_interiors.map((r) => this.smooth(r))
    const rings = [exterior, ...interiors]

    // The track is built as stacked flat layers. Each gets a distinct render
    // order + z so they never z-fight: grey verge (bottom) → striped kerb →
    // asphalt → white line → centerline (top).
    for (const zone of track.zones ?? []) {
      const st = ZONE_STYLE[zone.name]
      if (!st) continue
      for (const poly of zone.polygons) this.trackGroup.add(this.fillPolygon(poly, st.color, st.z, 1))
    }

    // Red/white striped kerb hugging the track edge, then the racing surface on
    // top of it (covering the inner half of the strip), then a bold white line.
    this.trackGroup.add(this.buildEdgeStrip(rings, 1.4, -0.3, { striped: true, stripeLen: 4, order: 2 }))

    // Racing surface: corridor exterior with the infield rings as holes.
    this.trackGroup.add(this.fillPolygon({ exterior, interiors }, ASPHALT, -0.2, 3))

    // Bold white track-limit line, drawn as a thin band so it stays visible when
    // zoomed out (a 1 px line would vanish).
    this.trackGroup.add(this.buildEdgeStrip(rings, 0.6, -0.1, { color: WALL, order: 4 }))

    // Dashed yellow centerline.
    const clGeo = new THREE.BufferGeometry().setFromPoints(
      this.smooth(track.centerline).map(([x, y]) => new THREE.Vector3(x, y, -0.05)),
    )
    const clMat = new THREE.LineDashedMaterial({ color: CENTERLINE, dashSize: 6, gapSize: 6 })
    const cl = new THREE.Line(clGeo, clMat)
    cl.computeLineDistances()
    cl.renderOrder = 5
    this.trackGroup.add(cl)

    // Cars render at their true physical size (≈5 m), so they sit realistically
    // on the 16 m-wide track instead of being inflated wider than the asphalt.
    // On huge circuits a car is small from the framing angle — zoom in to see it.
    const cx = (track.bounds.minx + track.bounds.maxx) / 2
    const cy = (track.bounds.miny + track.bounds.maxy) / 2
    const span = Math.max(
      track.bounds.maxx - track.bounds.minx,
      track.bounds.maxy - track.bounds.miny,
    )

    // Frame the whole circuit from a tilted bird's-eye angle.
    this.controls.target.set(cx, cy, 0)
    this.camera.position.set(cx, cy - span * 0.55, span * 0.55)
    this.controls.update()

    this.resize()
  }

  // ── Racing line ─────────────────────────────────────────────────────────────
  /** Draw the best lap as a flat ribbon coloured by speed (red = slow corner,
   *  green = fast straight). `points` are [x, y, speed] in metres / m·s⁻¹; vmin
   *  and vmax fix the colour scale. Replaces any previously drawn line. */
  setRacingLine(points: [number, number, number, number][], vmin: number, vmax: number) {
    this.clearRacingLine()
    if (points.length < 2) return

    points = TrackScene.simplifyLine(points)
    const span = Math.max(1e-6, vmax - vmin)
    const half = 1.2 // ribbon half-width (m)
    const positions: number[] = []
    const colors: number[] = []
    const colorAt = (speed: number) => {
      const t = Math.min(1, Math.max(0, (speed - vmin) / span))
      return new THREE.Color().setHSL(t * 0.33, 1, 0.5) // hue 0=red → 0.33=green
    }

    for (let i = 0; i < points.length - 1; i++) {
      const [ax, ay, av] = points[i]
      const [bx, by, bv] = points[i + 1]
      let dx = bx - ax
      let dy = by - ay
      const len = Math.hypot(dx, dy) || 1
      dx /= len
      dy /= len
      const nx = -dy * half // segment normal, scaled to half-width
      const ny = dx * half
      const ca = colorAt(av)
      const cb = colorAt(bv)
      // Two triangles forming the quad between point a and point b.
      positions.push(ax + nx, ay + ny, 0, ax - nx, ay - ny, 0, bx - nx, by - ny, 0)
      positions.push(ax + nx, ay + ny, 0, bx - nx, by - ny, 0, bx + nx, by + ny, 0)
      colors.push(ca.r, ca.g, ca.b, ca.r, ca.g, ca.b, cb.r, cb.g, cb.b)
      colors.push(ca.r, ca.g, ca.b, cb.r, cb.g, cb.b, cb.r, cb.g, cb.b)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3))
    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        polygonOffset: true,
        polygonOffsetFactor: -10,
        polygonOffsetUnits: -10,
      }),
    )
    mesh.position.z = 0.08 // just above every track layer (which sit at z < 0)
    mesh.renderOrder = 10
    this.racingLine = mesh
    this.scene.add(mesh)
  }

  /** Thin out the per‑frame racing line before it becomes geometry. A point is
   *  kept only if dropping it would bend the ribbon (perp. distance > posEps
   *  metres) or shift its speed colour (Δspeed > spdEps m/s). Long constant‑speed
   *  straights collapse to two points, cutting the vertex count by an order of
   *  magnitude with no visible change. Speed‑aware Douglas–Peucker (iterative). */
  private static simplifyLine(
    pts: [number, number, number, number][],
    posEps = 0.6,
    spdEps = 1.5,
  ): [number, number, number, number][] {
    if (pts.length < 3) return pts
    const keep = new Uint8Array(pts.length)
    keep[0] = keep[pts.length - 1] = 1
    const spdScale = posEps / spdEps // express Δspeed as a position‑equivalent deviation
    const stack: [number, number][] = [[0, pts.length - 1]]
    while (stack.length) {
      const [lo, hi] = stack.pop()!
      if (hi - lo < 2) continue
      const [ax, ay, av] = pts[lo]
      const [bx, by, bv] = pts[hi]
      let dx = bx - ax
      let dy = by - ay
      const len = Math.hypot(dx, dy) || 1
      dx /= len
      dy /= len
      let worst = 0
      let idx = -1
      for (let i = lo + 1; i < hi; i++) {
        const [px, py, pv] = pts[i]
        const perp = Math.abs((px - ax) * -dy + (py - ay) * dx)
        const t = ((px - ax) * dx + (py - ay) * dy) / len // 0..1 along the segment
        const spd = Math.abs(pv - (av + (bv - av) * t)) * spdScale
        const dev = Math.max(perp, spd)
        if (dev > worst) {
          worst = dev
          idx = i
        }
      }
      if (worst > posEps && idx >= 0) {
        keep[idx] = 1
        stack.push([lo, idx], [idx, hi])
      }
    }
    const out: [number, number, number, number][] = []
    for (let i = 0; i < pts.length; i++) if (keep[i]) out.push(pts[i])
    return out
  }

  clearRacingLine() {
    if (!this.racingLine) return
    this.scene.remove(this.racingLine)
    this.racingLine.geometry.dispose()
    ;(this.racingLine.material as THREE.Material).dispose()
    this.racingLine = null
  }

  /** Build a flat filled polygon (with holes) at height z. `order` fixes the
   *  paint order of the near-coplanar track layers and biases their depth via
   *  polygonOffset, so higher layers always win cleanly instead of z-fighting. */
  private fillPolygon(rings: PolyRings, color: number, z: number, order = 0): THREE.Mesh {
    const shape = new THREE.Shape(rings.exterior.map(([x, y]) => new THREE.Vector2(x, y)))
    for (const hole of rings.interiors) {
      shape.holes.push(new THREE.Path(hole.map(([x, y]) => new THREE.Vector2(x, y))))
    }
    const mesh = new THREE.Mesh(
      new THREE.ShapeGeometry(shape),
      new THREE.MeshStandardMaterial({
        color,
        roughness: 1,
        metalness: 0,
        polygonOffset: true,
        polygonOffsetFactor: -order,
        polygonOffsetUnits: -order,
      }),
    )
    mesh.position.z = z
    mesh.renderOrder = order
    return mesh
  }

  /** Chaikin corner-cutting: rounds off the angular OSM polylines into smooth
   *  racetrack curves. Treated as a closed ring. Skipped for already-dense rings
   *  (would explode the vertex count) or degenerate ones. */
  private smooth(pts: Vec2[]): Vec2[] {
    if (pts.length < 4 || pts.length > 160) return pts
    let cur = pts
    for (let it = 0; it < 2; it++) {
      const out: Vec2[] = []
      for (let i = 0; i < cur.length; i++) {
        const a = cur[i]
        const b = cur[(i + 1) % cur.length]
        out.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25])
        out.push([a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75])
      }
      cur = out
    }
    return cur
  }

  /** Signed area of a closed ring (>0 ⇒ counter-clockwise winding). */
  private static signedArea(ring: Vec2[]): number {
    let a = 0
    for (let i = 0; i < ring.length; i++) {
      const [x1, y1] = ring[i]
      const [x2, y2] = ring[(i + 1) % ring.length]
      a += x1 * y2 - x2 * y1
    }
    return a / 2
  }

  /** Extrude each ring outward into a flat strip (a kerb or a track-limit line).
   *  The offset direction is derived from the ring's winding (signed area), which
   *  is correct for every segment even on twisty circuits — unlike a centroid
   *  test, which flips on the far side of a winding track. The exterior ring
   *  grows outward (away from the surface); interior rings grow into their hole.
   *  With `striped`, quads alternate red/white by arc length (the F1 kerb). */
  private buildEdgeStrip(
    rings: Vec2[][],
    width: number,
    z: number,
    opts: { striped?: boolean; stripeLen?: number; color?: number; order?: number },
  ): THREE.Mesh {
    const order = opts.order ?? 0
    const positions: number[] = []
    const colors: number[] = []
    const red = new THREE.Color(KERB_RED)
    const white = new THREE.Color(KERB_WHITE)
    const solid = new THREE.Color(opts.color ?? KERB_WHITE)
    const stripeLen = opts.stripeLen ?? 4

    rings.forEach((ring, idx) => {
      const isHole = idx > 0 // interior rings extrude into their hole
      // For a CCW ring (area>0) the rotate-right normal (dy,-dx) points away from
      // the enclosed area. Holes want the opposite (into the hole).
      const sign = Math.sign(TrackScene.signedArea(ring)) * (isHole ? -1 : 1)
      let arc = 0
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i]
        const b = ring[(i + 1) % ring.length]
        const dx = b[0] - a[0]
        const dy = b[1] - a[1]
        const len = Math.hypot(dx, dy) || 1
        const nx = (sign * dy) / len
        const ny = (sign * -dx) / len
        const ax2 = a[0] + nx * width
        const ay2 = a[1] + ny * width
        const bx2 = b[0] + nx * width
        const by2 = b[1] + ny * width

        // Two triangles: a, b, b2 and a, b2, a2.
        positions.push(a[0], a[1], 0, b[0], b[1], 0, bx2, by2, 0)
        positions.push(a[0], a[1], 0, bx2, by2, 0, ax2, ay2, 0)

        const col = opts.striped
          ? Math.floor(arc / stripeLen) % 2 === 0
            ? red
            : white
          : solid
        for (let v = 0; v < 6; v++) colors.push(col.r, col.g, col.b)
        arc += len
      }
    })

    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3))
    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        polygonOffset: true,
        polygonOffsetFactor: -order,
        polygonOffsetUnits: -order,
      }),
    )
    mesh.position.z = z
    mesh.renderOrder = order
    return mesh
  }

  // ── Per-frame update ────────────────────────────────────────────────────────
  private loop() {
    this.raf = requestAnimationFrame(this.loop)
    // Bei verstecktem Tab nicht rendern/rechnen — spart GPU/CPU im Hintergrund.
    if (document.hidden) return

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
      obj.group.rotation.z = c.heading
      const hex = this.colorMode === "generation"
        ? genColorHex(c.generation)
        : rankColorHex(i, cars.length)
      // Nur umfärben, wenn sich die Farbe wirklich geändert hat.
      if (hex !== obj.lastHex) {
        obj.bodyMaterial.color.setHex(hex)
        obj.lastHex = hex
      }
    }

    // Park the selection ring under the selected car (if any & still present),
    // and draw the rays that car's sensors currently see.
    const sel = this.selectedIndex
    if (sel !== null && sel < cars.length) {
      this.selectionRing.visible = true
      this.selectionRing.position.set(cars[sel].x, cars[sel].y, 0.1)
      this.updateRays(cars[sel])
    } else {
      this.selectionRing.visible = false
      this.raysObj.visible = false
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  /** Zeichnet die 7 Sensor-Strahlen des angeklickten Autos: Ursprung = Auto,
   *  Länge = gemessener Abstand (`ray·MAX_RAY_M`), Farbe rot (Wand nah) → grün
   *  (frei). Aktualisiert nur die vorhandenen Attribute, baut keine Geometrie neu. */
  private updateRays(c: Car) {
    const rays = c.rays
    if (!rays || rays.length < RAY_ANGLES_RAD.length) {
      this.raysObj.visible = false
      return
    }
    const geo = this.raysObj.geometry
    const pos = geo.attributes.position as THREE.BufferAttribute
    const col = geo.attributes.color as THREE.BufferAttribute
    const z = 0.7
    const color = new THREE.Color()
    for (let i = 0; i < RAY_ANGLES_RAD.length; i++) {
      const r = Math.min(1, Math.max(0, rays[i]))
      const ang = c.heading + RAY_ANGLES_RAD[i]
      const dist = r * MAX_RAY_M
      const ex = c.x + Math.cos(ang) * dist
      const ey = c.y + Math.sin(ang) * dist
      pos.setXYZ(i * 2, c.x, c.y, z)
      pos.setXYZ(i * 2 + 1, ex, ey, z)
      color.setHSL(r * 0.33, 1, 0.5) // 0=rot (nah) … 0.33=grün (frei)
      col.setXYZ(i * 2, color.r, color.g, color.b)
      col.setXYZ(i * 2 + 1, color.r, color.g, color.b)
    }
    pos.needsUpdate = true
    col.needsUpdate = true
    this.raysObj.visible = true
  }

  setColorMode(mode: ColorMode) {
    this.colorMode = mode
  }

  /** Register a callback fired when the user clicks a car (or empty space → null). */
  setOnCarSelect(cb: (i: number | null) => void) {
    this.onCarSelect = cb
  }

  /** Highlight a car by index (null clears the highlight). */
  setSelected(i: number | null) {
    this.selectedIndex = i
  }

  private onPointerDown = (e: PointerEvent) => {
    this.pointerDown = { x: e.clientX, y: e.clientY }
  }

  private onPointerUp = (e: PointerEvent) => {
    const down = this.pointerDown
    this.pointerDown = null
    if (!down) return
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return // a drag, not a click

    const rect = this.canvas.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    )
    this.raycaster.setFromCamera(ndc, this.camera)
    const groups = this.carPool.map((o) => o.group)
    const groupIndex = new Map(groups.map((g, i) => [g, i]))
    const hits = this.raycaster.intersectObjects(groups, true)
    let picked: number | null = null
    if (hits.length) {
      // Vom getroffenen Mesh die Eltern hoch bis zur Auto-Gruppe laufen.
      let o: THREE.Object3D | null = hits[0].object
      while (o && !groupIndex.has(o as THREE.Group)) o = o.parent
      if (o) picked = groupIndex.get(o as THREE.Group) ?? null
    }
    this.selectedIndex = picked
    this.onCarSelect?.(picked)
  }

  private syncPool(n: number) {
    while (this.carPool.length < n) this.carPool.push(this.makeCar())
  }

  private makeCar(): CarObj {
    // A flat box roughly the size of an F1 car, length along +X (heading 0).
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.1, roughness: 0.6 })
    const box = new THREE.Mesh(new THREE.BoxGeometry(CAR_LENGTH_M, 2, 1), material)
    box.position.z = 0.5
    box.name = "body"

    const group = new THREE.Group()
    group.add(box)
    group.scale.setScalar(this.carScale)
    this.scene.add(group)
    return { group, bodyMaterial: material, lastHex: -1 }
  }

  // ── Camera / resize ─────────────────────────────────────────────────────────
  private resize() {
    const w = this.canvas.clientWidth || 1
    const h = this.canvas.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  dispose() {
    this.clearRacingLine()
    cancelAnimationFrame(this.raf)
    this.canvas.removeEventListener("pointerdown", this.onPointerDown)
    this.canvas.removeEventListener("pointerup", this.onPointerUp)
    this.selectionRing.geometry.dispose()
    ;(this.selectionRing.material as THREE.Material).dispose()
    this.raysObj.geometry.dispose()
    ;(this.raysObj.material as THREE.Material).dispose()
    this.resizeObs.disconnect()
    this.controls.dispose()
    this.renderer.dispose()
  }
}
