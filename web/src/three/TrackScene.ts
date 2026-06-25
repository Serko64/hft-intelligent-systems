import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import type { Car, PolyRings, TrackMsg, Vec2 } from "@/lib/types"

// ── Einstellbare Werte ──────────────────────────────────────────────────────────
const CAR_LENGTH_M = 5 // echte F1-Länge, passt zur Trainings-Bounding-Box (env CAR_LENGTH_M)

// Sensor-Strahlen des angeklickten Autos (spiegelt environment.py: RAY_ANGLES / MAX_RAY_M).
const RAY_ANGLES_RAD = [-75, -45, -20, 0, 20, 45, 75].map((d) => (d * Math.PI) / 180)
const MAX_RAY_M = 80

const GRASS = 0x2f6b22 // kräftiges Grün, damit die dunkle Strecke und der rote Randstein hervorstechen
const ASPHALT = 0x37352f // Fahrbahn (FIA-Querschnitt "Strecke")
const WALL = 0xf4f4ee // helle weiße Streckenbegrenzungslinie
const CENTERLINE = 0xffd200
const KERB_RED = 0xc0392b
const KERB_WHITE = 0xeeeeee

// Schmaler grauer Asphaltstreifen direkt außerhalb der weißen Linie. Der gestreifte
// Randstein wird separat als extrudierter Streifen darübergelegt (siehe buildEdgeStrip).
const ZONE_STYLE: Record<string, { color: number; z: number }> = {
  runoff: { color: 0x6e6c66, z: -0.4 }, // grauer Asphaltstreifen
}

/** Rang zu Farbe (hex), wie die pygame-Ansicht (bester = Gold bis schlechtester = Dunkelrot). */
function rankColorHex(rank: number, total: number): number {
  const f = total > 1 ? rank / (total - 1) : 0
  if (f < 0.08) return 0xffd700
  if (f < 0.2) return 0x32dc50
  if (f < 0.5) return 0xc88232
  return 0x8a3a3a
}

/** Generation zu einer stabilen, gut unterscheidbaren Farbe (Hue über goldenen Schnitt). */
function genColorHex(gen: number): number {
  const hue = (gen * 0.6180339887) % 1
  return new THREE.Color().setHSL(hue, 0.7, 0.55).getHex()
}

export type ColorMode = "rank" | "generation"

interface CarObj {
  group: THREE.Group
  bodyMaterial: THREE.MeshStandardMaterial // je Frame nach Rang eingefärbt
  lastHex: number // zuletzt gesetzte Farbe, vermeidet überflüssige setHex-Aufrufe pro Frame
}

/**
 * Rendert die Strecke und die Live-Autos in einer frei beweglichen 3D-Ansicht
 * (PerspectiveCamera und OrbitControls: Linksziehen dreht, Rechtsziehen verschiebt,
 * Mausrad zoomt). Alle Koordinaten sind in Metern (z zeigt nach oben). Die Render-
 * schleife liest die aktuellen Autos je Frame über `getCars`, React rendert also
 * nie mit 60 fps neu.
 */
export class TrackScene {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private trackGroup = new THREE.Group()
  private racingLine: THREE.Mesh | null = null
  private carPool: CarObj[] = []
  private carScale = 1 // auf großen Strecken werden die Autos vergrößert, damit sie sichtbar bleiben
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
    // logarithmicDepthBuffer hält die Tiefengenauigkeit über den riesigen Bereich von
    // nah bis fern brauchbar (wenige Meter bis kilometerbreite Strecken). Sonst würden
    // die gestapelten Streckenschichten ums Z konkurrieren ("clippen").
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.scene.background = new THREE.Color(GRASS)
    this.scene.add(this.trackGroup)

    // Perspektivkamera mit z nach oben, per Maus steuerbar.
    this.camera = new THREE.PerspectiveCamera(50, 1, 1, 500000)
    this.camera.up.set(0, 0, 1)
    this.camera.position.set(0, -200, 200)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12 // direkter als das alte träge 0.08
    this.controls.minDistance = 3 // bis ganz an ein einzelnes Auto heranzoomen
    this.controls.maxDistance = 200000
    // Zum Mauszeiger zoomen statt zum (oft weit entfernten) Ziel, damit man direkt
    // auf ein einzelnes Auto zugehen kann, ohne dass der Zoom an der Mindestdistanz
    // zur Streckenmitte „hängenbleibt".
    this.controls.zoomToCursor = true
    this.controls.zoomSpeed = 1.6
    this.controls.panSpeed = 1.2
    this.controls.rotateSpeed = 0.9
    this.controls.screenSpacePanning = true // Rechtsziehen verschiebt in der Bildebene (intuitiv)
    // Kamera über dem Boden halten und knapp vor dem exakten Senkrecht-von-oben-Pol,
    // wo das Drehen sonst kippt oder einfriert (Gimbal Lock bei z-nach-oben-Kamera).
    this.controls.minPolarAngle = 0.08
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05
    this.controls.target.set(0, 0, 0)

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x202018, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 1.4)
    dir.position.set(0.4, 0.6, 1)
    this.scene.add(dir)

    // Flacher Ring unter dem ausgewählten Auto, um es hervorzuheben.
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

    // Klick auf ein Auto wählt es aus (nur wenn der Zeiger kaum bewegt wurde, damit
    // das Drehen der Kamera keine Auswahl auslöst).
    this.canvas.addEventListener("pointerdown", this.onPointerDown)
    this.canvas.addEventListener("pointerup", this.onPointerUp)

    this.resizeObs = new ResizeObserver(() => this.resize())
    this.resizeObs.observe(canvas)
    this.resize()

    this.loop = this.loop.bind(this)
    this.raf = requestAnimationFrame(this.loop)
  }

  // ── Streckengeometrie ───────────────────────────────────────────────────────
  setTrack(track: TrackMsg) {
    this.trackGroup.clear()

    // Geglättete Korridorringe (Chaikin), damit eckige OSM-Polylinien als fließende
    // Streckenkurven wirken. Dieselben geglätteten Ringe treiben Randstein und weiße
    // Linie, damit jede Schicht exakt übereinanderliegt.
    const exterior = this.smooth(track.corridor_exterior)
    const interiors = track.corridor_interiors.map((r) => this.smooth(r))
    const rings = [exterior, ...interiors]

    // Die Strecke entsteht aus gestapelten flachen Schichten. Jede bekommt eine eigene
    // Render-Reihenfolge und ein eigenes z, damit sie nie ums Z konkurrieren: grauer
    // Streifen (unten), gestreifter Randstein, Asphalt, weiße Linie, Mittellinie (oben).
    for (const zone of track.zones ?? []) {
      const st = ZONE_STYLE[zone.name]
      if (!st) continue
      for (const poly of zone.polygons) this.trackGroup.add(this.fillPolygon(poly, st.color, st.z, 1))
    }

    // Rot-weiß gestreifter Randstein direkt an der Streckenkante, darüber die Fahrbahn
    // (deckt die innere Hälfte des Streifens ab), darüber eine kräftige weiße Linie.
    this.trackGroup.add(this.buildEdgeStrip(rings, 1.4, -0.3, { striped: true, stripeLen: 4, order: 2 }))

    // Fahrbahn: Korridor-Außenring mit den Infield-Ringen als Löchern.
    this.trackGroup.add(this.fillPolygon({ exterior, interiors }, ASPHALT, -0.2, 3))

    // Kräftige weiße Begrenzungslinie, als schmales Band gezeichnet, damit sie auch
    // herausgezoomt sichtbar bleibt (eine 1-Pixel-Linie würde verschwinden).
    this.trackGroup.add(this.buildEdgeStrip(rings, 0.6, -0.1, { color: WALL, order: 4 }))

    // Gelb gestrichelte Mittellinie.
    const clGeo = new THREE.BufferGeometry().setFromPoints(
      this.smooth(track.centerline).map(([x, y]) => new THREE.Vector3(x, y, -0.05)),
    )
    const clMat = new THREE.LineDashedMaterial({ color: CENTERLINE, dashSize: 6, gapSize: 6 })
    const cl = new THREE.Line(clGeo, clMat)
    cl.computeLineDistances()
    cl.renderOrder = 5
    this.trackGroup.add(cl)

    // Autos werden in ihrer echten physischen Größe gerendert (≈5 m), sie sitzen also
    // realistisch auf der 16 m breiten Strecke, statt breiter als der Asphalt zu wirken.
    // Auf riesigen Strecken ist ein Auto aus der Rahmung klein, zum Sehen hineinzoomen.
    const cx = (track.bounds.minx + track.bounds.maxx) / 2
    const cy = (track.bounds.miny + track.bounds.maxy) / 2
    const span = Math.max(
      track.bounds.maxx - track.bounds.minx,
      track.bounds.maxy - track.bounds.miny,
    )

    // Die ganze Strecke aus einem geneigten Vogelperspektiven-Winkel einrahmen.
    this.controls.target.set(cx, cy, 0)
    this.camera.position.set(cx, cy - span * 0.55, span * 0.55)
    this.controls.update()

    this.resize()
  }

  // ── Racing-Line ───────────────────────────────────────────────────────────────
  /** Zeichnet die beste Runde als flaches Band, eingefärbt nach Tempo (rot = langsame
   *  Kurve, grün = schnelle Gerade). `points` sind [x, y, Tempo] in Metern und m/s,
   *  vmin und vmax legen die Farbskala fest. Ersetzt eine zuvor gezeichnete Linie. */
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
      return new THREE.Color().setHSL(t * 0.33, 1, 0.5) // Hue 0 = rot, 0.33 = grün
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
      // Zwei Dreiecke, die das Viereck zwischen Punkt a und Punkt b bilden.
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
    mesh.position.z = 0.08 // knapp über allen Streckenschichten (die bei z < 0 liegen)
    mesh.renderOrder = 10
    this.racingLine = mesh
    this.scene.add(mesh)
  }

  /** Dünnt die je Frame aufgezeichnete Racing-Line aus, bevor sie zu Geometrie wird.
   *  Ein Punkt bleibt nur, wenn sein Weglassen das Band biegen würde (Abstand > posEps
   *  Meter) oder seine Tempo-Farbe verschiebt (ΔTempo > spdEps m/s). Lange Geraden mit
   *  konstantem Tempo schrumpfen auf zwei Punkte, das senkt die Vertex-Zahl um eine
   *  Größenordnung ohne sichtbaren Unterschied. Tempo-bewusster Douglas-Peucker (iterativ). */
  private static simplifyLine(
    pts: [number, number, number, number][],
    posEps = 0.6,
    spdEps = 1.5,
  ): [number, number, number, number][] {
    if (pts.length < 3) return pts
    const keep = new Uint8Array(pts.length)
    keep[0] = keep[pts.length - 1] = 1
    const spdScale = posEps / spdEps // ΔTempo als positions-äquivalente Abweichung ausdrücken
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
        const t = ((px - ax) * dx + (py - ay) * dy) / len // 0..1 entlang des Segments
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

  /** Baut ein flaches gefülltes Polygon (mit Löchern) auf Höhe z. `order` legt die
   *  Zeichenreihenfolge der fast koplanaren Streckenschichten fest und beeinflusst ihre
   *  Tiefe über polygonOffset, sodass höhere Schichten sauber gewinnen statt zu z-fighten. */
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

  /** Chaikin-Eckenschneiden: rundet die eckigen OSM-Polylinien zu glatten Strecken-
   *  kurven. Wird als geschlossener Ring behandelt. Übersprungen bei bereits dichten
   *  Ringen (würde die Vertex-Zahl explodieren lassen) oder entarteten Ringen. */
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

  /** Vorzeichenbehaftete Fläche eines geschlossenen Rings (>0 bedeutet Drehsinn gegen den Uhrzeigersinn). */
  private static signedArea(ring: Vec2[]): number {
    let a = 0
    for (let i = 0; i < ring.length; i++) {
      const [x1, y1] = ring[i]
      const [x2, y2] = ring[(i + 1) % ring.length]
      a += x1 * y2 - x2 * y1
    }
    return a / 2
  }

  /** Extrudiert jeden Ring nach außen zu einem flachen Streifen (Randstein oder
   *  Begrenzungslinie). Die Versatzrichtung kommt aus dem Drehsinn des Rings (Vorzeichen
   *  der Fläche) und stimmt für jedes Segment auch auf kurvigen Strecken, anders als ein
   *  Schwerpunkt-Test, der auf der Gegenseite einer Schleife kippt. Der Außenring wächst
   *  nach außen (weg von der Fläche), Innenringe wachsen in ihr Loch hinein. Mit `striped`
   *  wechseln die Vierecke nach Bogenlänge rot und weiß (der F1-Randstein). */
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
      const isHole = idx > 0 // Innenringe extrudieren in ihr Loch hinein
      // Bei einem Ring gegen den Uhrzeigersinn (Fläche>0) zeigt die nach rechts gedrehte
      // Normale (dy,-dx) weg von der eingeschlossenen Fläche. Löcher wollen das Gegenteil.
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

        // Zwei Dreiecke: a, b, b2 und a, b2, a2.
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

  // ── Update je Frame ───────────────────────────────────────────────────────────
  private loop() {
    this.raf = requestAnimationFrame(this.loop)
    // Bei verstecktem Tab nicht rendern oder rechnen, das spart GPU und CPU im Hintergrund.
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

    // Den Auswahlring unter dem gewählten Auto parken (falls es eins gibt und noch da
    // ist) und die Strahlen zeichnen, die dessen Sensoren gerade sehen.
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
   *  Länge = gemessener Abstand (`ray·MAX_RAY_M`), Farbe rot (Wand nah) bis grün
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

  /** Registriert einen Callback, der bei Klick auf ein Auto feuert (leerer Raum gibt null). */
  setOnCarSelect(cb: (i: number | null) => void) {
    this.onCarSelect = cb
  }

  /** Hebt ein Auto per Index hervor (null entfernt die Hervorhebung). */
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
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return // ein Ziehen, kein Klick

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
    // Eine flache Box etwa in F1-Auto-Größe, Länge entlang +X (Fahrtrichtung 0).
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

  // ── Kamera und Größenänderung ─────────────────────────────────────────────────
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
