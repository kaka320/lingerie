/**
 * MannequinViewer — shows the mannequin GLB and reshapes its bust / underbust
 * from measurements in centimetres.
 *
 * The GLB carries two morph targets baked by tools/bake_morphs.py:
 *   "bust"      – pushes both breasts forward
 *   "underbust" – inflates the ribcage
 * Each target changes BOTH circumferences a little, so the two weights are
 * found by solving a 2x2 linear system against the measured calibration.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Measured on the base mesh with tools/measure.py (cm).
export const CALIBRATION = {
  base: { underbust: 75.1, bust: 89.7 },
  // circumference gained at weight = 1.0 for each target
  bustTarget: { underbust: 2.6, bust: 14.6 },
  underbustTarget: { underbust: 13.4, bust: 3.7 },
  weightRange: [-0.6, 2.2],
};

export function weightsFor(underbustCm, bustCm, c = CALIBRATION) {
  const du = underbustCm - c.base.underbust;
  const db = bustCm - c.base.bust;
  // [du] = [a b] [wb]      a = bustTarget.underbust   b = underbustTarget.underbust
  // [db]   [c d] [wu]      c = bustTarget.bust        d = underbustTarget.bust
  const a = c.bustTarget.underbust, b = c.underbustTarget.underbust;
  const cc = c.bustTarget.bust, d = c.underbustTarget.bust;
  const det = a * d - b * cc;
  let wb = (du * d - b * db) / det;
  let wu = (a * db - cc * du) / det;
  const [lo, hi] = c.weightRange;
  wb = Math.min(hi, Math.max(lo, wb));
  wu = Math.min(hi, Math.max(lo, wu));
  return { bust: wb, underbust: wu };
}

/** Rough UK/EU cup label from the bust/underbust difference. */
export function cupSize(underbustCm, bustCm) {
  const diff = bustCm - underbustCm;
  const cups = ['AA', 'A', 'B', 'C', 'D', 'DD', 'E', 'F', 'FF', 'G', 'GG', 'H'];
  const i = Math.max(0, Math.min(cups.length - 1, Math.round((diff - 10) / 2.5)));
  const band = Math.round(underbustCm / 5) * 5;
  return `${band}${cups[i]}`;
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

    this.mesh = null;
    this.ready = new Promise((resolve, reject) => {
      new GLTFLoader().load(modelUrl, (gltf) => {
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.morphTargetDictionary) {
            this.mesh = o;
            o.material = new THREE.MeshStandardMaterial({ color: skin, roughness: 0.65, metalness: 0 });
          }
        });
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

  /** Set the shape from measurements in cm. Returns the morph weights used. */
  setMeasurements(underbustCm, bustCm) {
    const w = weightsFor(underbustCm, bustCm);
    if (this.mesh) {
      const d = this.mesh.morphTargetDictionary;
      this.mesh.morphTargetInfluences[d.bust] = w.bust;
      this.mesh.morphTargetInfluences[d.underbust] = w.underbust;
    }
    return w;
  }

  /** Set raw morph weights directly (0 = base mesh). */
  setWeights(bust, underbust) {
    if (!this.mesh) return;
    const d = this.mesh.morphTargetDictionary;
    this.mesh.morphTargetInfluences[d.bust] = bust;
    this.mesh.morphTargetInfluences[d.underbust] = underbust;
  }
}
