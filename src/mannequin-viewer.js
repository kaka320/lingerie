/**
 * MannequinViewer — shows a body model and reshapes it from measurements:
 * height, weight (BMI), bust, underbust and waist.
 *
 * Each model is a GLB with four morph targets ("bust", "underbust", "waist",
 * "fat") plus a calibration JSON measured on the mesh (see tools/). Height is
 * handled by scaling the whole model.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const INCH = 2.54;

/** Available models. `skin` replaces the material with a plain colour; omit it to keep the model's textures. */
export const MODELS = [
  { id: 'model', name: 'Model', url: 'models/bikini.glb', calibration: 'models/bikini.json' },
  { id: 'mannequin', name: 'Mannequin', url: 'models/mannequin.glb', calibration: 'models/mannequin.json', skin: 0xe8c4a8 },
];

const KEYS = ['underbust', 'bust', 'waist'];   // measured circumferences
const SHAPES = ['bust', 'underbust', 'waist']; // morph targets solved from them
const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

/** Solve a small linear system A·w = b (Gaussian elimination). */
function linsolve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-9) return null;
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

/**
 * @param {object} m  { heightCm, weightKg, bustCm, underbustCm, waistCm }
 * @param {object} c  calibration JSON of the model
 * @returns {{ weights:{bust,underbust,waist,fat}, scale:number, bmi:number }}
 */
export function solve(m, c) {
  const ratio = m.heightCm / c.statureCm;              // real cm per calibration cm
  const scale = (m.heightCm / 100) / c.statureWorld;    // GLB world scale
  const bmi = m.weightKg / Math.pow(m.heightCm / 100, 2);
  const fat = clamp((bmi - c.baseBmi) / c.bmiPerFatUnit, c.range.fat);

  // Remaining circumference change each shape must produce, in calibration cm.
  const want = { underbust: m.underbustCm, bust: m.bustCm, waist: m.waistCm };
  const need = KEYS.map((k) => want[k] / ratio - c.base[k] - c.fatT[k] * fat);

  // Solve all three shapes together; clamp any that go out of range and re-solve the rest.
  const w = { bust: 0, underbust: 0, waist: 0 };
  let free = [...SHAPES];
  for (let pass = 0; pass < 3 && free.length; pass++) {
    const fixed = SHAPES.filter((s) => !free.includes(s));
    const rows = KEYS.map((k, i) => need[i] - fixed.reduce((a, s) => a + c[s + 'T'][k] * w[s], 0));
    // least squares when fewer unknowns than equations: (AᵀA) w = Aᵀ b
    const A = KEYS.map((k) => free.map((s) => c[s + 'T'][k]));
    const AtA = free.map((_, i) => free.map((_, j) => A.reduce((a, r) => a + r[i] * r[j], 0)));
    const Atb = free.map((_, i) => A.reduce((a, r, k) => a + r[i] * rows[k], 0));
    const sol = linsolve(AtA, Atb);
    if (!sol) break;
    let clamped = false;
    free.forEach((s, i) => {
      const v = clamp(sol[i], c.range[s]);
      if (v !== sol[i]) clamped = true;
      w[s] = v;
    });
    if (!clamped) break;
    free = free.filter((s) => w[s] > c.range[s][0] && w[s] < c.range[s][1]);
  }
  return { weights: { ...w, fat }, scale, bmi };
}

/** Bra size from inches: band = underbust rounded to an even number, cup from the difference. */
export function braSize(underbustIn, bustIn) {
  let band = Math.round(underbustIn);
  if (band % 2) band += 1;
  const cups = ['AA', 'A', 'B', 'C', 'D', 'DD', 'DDD', 'G', 'H', 'I', 'J', 'K'];
  const i = Math.max(0, Math.min(cups.length - 1, Math.round(bustIn - band)));
  return `${band}${cups[i]}`;
}

export function bmiLabel(bmi) {
  if (bmi < 18.5) return 'Underweight';
  if (bmi < 25) return 'Healthy';
  if (bmi < 30) return 'Overweight';
  return 'Obese';
}

export class MannequinViewer {
  constructor(container, { model = MODELS[0].id } = {}) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf6f2ee);

    const w = container.clientWidth || 600, h = container.clientHeight || 800;
    this.camera = new THREE.PerspectiveCamera(32, w / h, 0.05, 50);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 7;
    this.controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0xb0a090, 1.2);
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(2, 4, 3);
    const fill = new THREE.DirectionalLight(0xfff0e0, 0.9); fill.position.set(-3, 2, 1);
    const rim = new THREE.DirectionalLight(0xffffff, 1.2); rim.position.set(0, 3, -4);
    this.scene.add(hemi, key, fill, rim);

    const floor = new THREE.Mesh(new THREE.CircleGeometry(0.6, 64),
      new THREE.MeshStandardMaterial({ color: 0xe6dfd8, roughness: 1 }));
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);

    // Height reference: a faint scale bar in feet beside the model
    const mat = new THREE.LineBasicMaterial({ color: 0xc9bdb5 });
    const bar = new THREE.Group();
    for (let ft = 0; ft <= 7; ft++) {
      const y = ft * 12 * INCH / 100;
      bar.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(0.48, y, -0.45), new THREE.Vector3(0.55, y, -0.45)]), mat));
    }
    bar.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0.55, 0, -0.45), new THREE.Vector3(0.55, 7 * 12 * INCH / 100, -0.45)]), mat));
    this.scene.add(bar);

    this.mesh = null; this.root = null; this.calibration = null; this.body = null;
    this.view = 'front'; this.heightM = 1.65;
    this.cache = new Map();
    this.ready = this.setModel(model);
    this.setView('front');

    window.addEventListener('resize', () => this.resize());
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }

  async _load(def) {
    if (this.cache.has(def.id)) return this.cache.get(def.id);
    const [gltf, calibration] = await Promise.all([
      new GLTFLoader().loadAsync(def.url),
      fetch(def.calibration).then((r) => r.json()),
    ]);
    let mesh = null;
    gltf.scene.traverse((o) => {
      if (o.isMesh && o.morphTargetDictionary) {
        mesh = o;
        if (def.skin !== undefined) o.material = new THREE.MeshStandardMaterial({ color: def.skin, roughness: 0.65 });
      }
    });
    // Wrap the scene so it can be scaled about the feet: centre it on x/z and put the soles at y = 0.
    const root = new THREE.Group();
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const c = box.getCenter(new THREE.Vector3());
    gltf.scene.position.set(-c.x, -box.min.y, -c.z);
    root.add(gltf.scene);
    const entry = { root, mesh, calibration };
    this.cache.set(def.id, entry);
    return entry;
  }

  /** Switch the displayed model by id (see MODELS). Keeps the current measurements. */
  async setModel(id) {
    const def = MODELS.find((m) => m.id === id) || MODELS[0];
    this.loading = def.id;
    const entry = await this._load(def);
    if (this.loading !== def.id) return this;          // a newer switch won
    if (this.root) this.scene.remove(this.root);
    this.root = entry.root; this.mesh = entry.mesh; this.calibration = entry.calibration;
    this.modelId = def.id;
    this.scene.add(this.root);
    if (this.body) this.setBody(this.body);
    return this;
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Camera preset: 'front' | 'side' | 'back'. Frames the whole body. */
  setView(view) {
    this.view = view;
    const h = this.heightM;
    const d = Math.max(2.4, h * 2.0), ty = h * 0.52;
    const dir = { front: [0, 1], side: [1, 0], back: [0, -1] }[view] || [0, 1];
    this.controls.target.set(0, ty, 0);
    this.camera.position.set(dir[0] * d, ty + 0.1, dir[1] * d);
    this.controls.update();
  }

  /**
   * Set the body from measurements.
   * @param {object} m { heightCm, weightKg, bustCm, underbustCm, waistCm }
   */
  setBody(m) {
    this.body = m;
    if (!this.calibration) return null;
    const r = solve(m, this.calibration);
    const d = this.mesh.morphTargetDictionary, inf = this.mesh.morphTargetInfluences;
    for (const k of Object.keys(r.weights)) if (k in d) inf[d[k]] = r.weights[k];
    this.root.scale.setScalar(r.scale);
    const newH = m.heightCm / 100;
    if (Math.abs(newH - this.heightM) > 1e-3) {
      const dy = (newH - this.heightM) * 0.52;   // keep the body framed when height changes
      this.heightM = newH;
      this.controls.target.y += dy; this.camera.position.y += dy;
    }
    return r;
  }

  /** Convenience: inches / feet / kg → setBody. */
  setBodyImperial({ feet, inches, weightKg, bustIn, underbustIn, waistIn }) {
    return this.setBody({
      heightCm: (feet * 12 + inches) * INCH,
      weightKg,
      bustCm: bustIn * INCH,
      underbustCm: underbustIn * INCH,
      waistCm: waistIn * INCH,
    });
  }
}
