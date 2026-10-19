/**
 * MannequinViewer — shows the mannequin GLB and reshapes it from body
 * measurements: height, weight (BMI), bust, underbust and waist.
 *
 * The GLB carries four morph targets baked by tools/bake_morphs.py:
 *   "bust"      – fuller breasts
 *   "underbust" – wider ribcage
 *   "waist"     – wider waist
 *   "fat"       – whole-body fullness (belly, hips, thighs, arms, face)
 * Height is handled by scaling the whole mannequin.
 *
 * Each target changes several circumferences, so weights are solved in order:
 * fat (from BMI) → waist → bust/underbust (2x2 system), all against the
 * calibration measured with tools/measure.py.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const INCH = 2.54;

// Measured on the mesh with tools/measure.py (cm, mesh is 180 cm tall).
export const CALIBRATION = {
  meshHeightCm: 180,
  baseBmi: 18.5,            // the untouched mesh is the slimmest body (fat weight never < 0)
  bmiPerFatUnit: 7,         // fat weight 1.0 ≈ +7 BMI
  base:      { underbust: 75.1, bust: 89.7, waist: 59.4 },
  // circumference gained (cm) at weight = 1.0 for each target
  bustT:      { underbust: 4.6,  bust: 9.1,  waist: 0 },
  underbustT: { underbust: 13.4, bust: 3.7,  waist: 0 },
  waistT:     { underbust: 0,    bust: 0,    waist: 17.4 },
  fatT:       { underbust: 13.6, bust: 10.3, waist: 22.7 },
  range: { bust: [-0.6, 3.0], underbust: [-0.6, 2.5], waist: [-0.6, 3.0], fat: [0, 2.5] },
};

const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

/**
 * @param {object} m  { heightCm, weightKg, bustCm, underbustCm, waistCm }
 * @returns {{ weights:{bust,underbust,waist,fat}, scale:number, bmi:number }}
 */
export function solve(m, c = CALIBRATION) {
  const scale = m.heightCm / c.meshHeightCm;          // whole-model scale
  const bmi = m.weightKg / Math.pow(m.heightCm / 100, 2);
  const fat = clamp((bmi - c.baseBmi) / c.bmiPerFatUnit, c.range.fat);

  // Work in mesh-space cm: real circumference / scale.
  const ub = m.underbustCm / scale, b = m.bustCm / scale, w = m.waistCm / scale;

  const waist = clamp((w - c.base.waist - c.fatT.waist * fat) / c.waistT.waist, c.range.waist);

  const du = ub - c.base.underbust - c.fatT.underbust * fat - c.waistT.underbust * waist;
  const db = b - c.base.bust - c.fatT.bust * fat - c.waistT.bust * waist;
  // [du] = [a b] [wb]
  // [db]   [cc d] [wu]
  const a = c.bustT.underbust, bb = c.underbustT.underbust;
  const cc = c.bustT.bust, d = c.underbustT.bust;
  const det = a * d - bb * cc;
  const bust = clamp((du * d - bb * db) / det, c.range.bust);
  const underbust = clamp((a * db - cc * du) / det, c.range.underbust);

  return { weights: { bust, underbust, waist, fat }, scale, bmi };
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
  constructor(container, { modelUrl = 'models/mannequin.glb', skin = 0xe8c4a8 } = {}) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf6f2ee);

    const w = container.clientWidth || 600, h = container.clientHeight || 800;
    this.camera = new THREE.PerspectiveCamera(32, w / h, 0.05, 50);
    this.camera.position.set(0, 1.25, 2.6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1.15, 0);
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 5;
    this.controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0xb0a090, 1.2);
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(2, 4, 3);
    const fill = new THREE.DirectionalLight(0xfff0e0, 0.9); fill.position.set(-3, 2, 1);
    const rim = new THREE.DirectionalLight(0xffffff, 1.2); rim.position.set(0, 3, -4);
    this.scene.add(hemi, key, fill, rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(0.6, 64),
      new THREE.MeshStandardMaterial({ color: 0xe6dfd8, roughness: 1 }));
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);

    // Height reference: a faint scale bar in feet beside the mannequin
    this.scaleBar = new THREE.Group();
    const mat = new THREE.LineBasicMaterial({ color: 0xc9bdb5 });
    for (let ft = 0; ft <= 7; ft++) {
      const y = ft * 12 * INCH / 100;
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0.55, y, 0), new THREE.Vector3(0.62, y, 0)]);
      this.scaleBar.add(new THREE.Line(g, mat));
    }
    const axis = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0.62, 0, 0), new THREE.Vector3(0.62, 7 * 12 * INCH / 100, 0)]);
    this.scaleBar.add(new THREE.Line(axis, mat));
    this.scene.add(this.scaleBar);

    this.mesh = null; this.root = null;
    this.ready = new Promise((resolve, reject) => {
      new GLTFLoader().load(modelUrl, (gltf) => {
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.morphTargetDictionary) {
            this.mesh = o;
            o.material = new THREE.MeshStandardMaterial({ color: skin, roughness: 0.65, metalness: 0 });
          }
        });
        this.root = gltf.scene;
        this.scene.add(gltf.scene);
        resolve(this);
      }, undefined, reject);
    });

    window.addEventListener('resize', () => this.resize());
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /**
   * Set the body from measurements.
   * @param {object} m { heightCm, weightKg, bustCm, underbustCm, waistCm }
   */
  setBody(m) {
    const r = solve(m);
    if (this.mesh) {
      const d = this.mesh.morphTargetDictionary, inf = this.mesh.morphTargetInfluences;
      for (const k of Object.keys(r.weights)) if (k in d) inf[d[k]] = r.weights[k];
      this.root.scale.setScalar(r.scale);
      this.controls.target.set(0, 1.15 * r.scale, 0);
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

  /** Set raw morph weights directly (0 = base mesh). */
  setWeights(weights) {
    if (!this.mesh) return;
    const d = this.mesh.morphTargetDictionary;
    for (const k in weights) if (k in d) this.mesh.morphTargetInfluences[d[k]] = weights[k];
  }
}
