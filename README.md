# Lingerie fit mannequin

A 3D mannequin that reshapes itself from the customer's height, weight,
bust, underbust and waist, for use on a lingerie e-commerce site.

## Try it

Any static server works (modules need http, not `file://`):

```
python3 -m http.server 8000
# open http://localhost:8000
```

## How it works

- `models/mannequin.glb` is the base mesh with four **morph targets** baked in:
  `bust`, `underbust`, `waist` and `fat` (whole-body fullness driven by BMI).
- `src/mannequin-viewer.js` loads it with Three.js. Height scales the whole
  model; weight sets the `fat` target from BMI; the waist, bust and underbust
  weights are then solved against the calibration measured on the mesh
  (`CALIBRATION`) so the entered circumferences are reproduced.
- `index.html` is a demo page with height (ft/in), weight (kg) and bust /
  underbust / waist (inches), plus a bra-size and BMI readout.

## Embedding in your shop

```html
<div id="viewer" style="height:600px"></div>
<script type="importmap">{ "imports": {
  "three": "./vendor/three/build/three.module.js",
  "three/addons/": "./vendor/three/addons/" } }</script>
<script type="module">
  import { MannequinViewer } from './src/mannequin-viewer.js';
  const v = new MannequinViewer(document.getElementById('viewer'), { modelUrl: 'models/mannequin.glb' });
  v.ready.then(() => v.setBodyImperial({
    feet: 5, inches: 5, weightKg: 58, bustIn: 35, underbustIn: 30, waistIn: 26 }));
  // or metric: v.setBody({ heightCm, weightKg, bustCm, underbustCm, waistCm })
</script>
```

## Regenerating the morph targets

`tools/bake_morphs.py` computes the deformations and writes them into the GLB;
`tools/measure.py` reports the resulting circumferences used for calibration.

```
pip install numpy trimesh scipy shapely networkx
python3 tools/bake_morphs.py original.glb models/mannequin.glb
python3 tools/measure.py models/mannequin.glb   # copy numbers into CALIBRATION
```

The base mesh is a free female mannequin base mesh (Sketchfab); check its
licence before commercial use.
