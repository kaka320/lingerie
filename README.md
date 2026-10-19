# Lingerie fit mannequin

A 3D body preview that reshapes itself from the customer's height, weight,
bust, underbust and waist, for use on a lingerie e-commerce site. Customers
can switch between a textured model and a plain mannequin, and between
front, side and back views.

## Try it

Any static server works (modules need http, not `file://`):

```
python3 -m http.server 8000
# open http://localhost:8000
```

## How it works

- `models/bikini.glb` (default "Model") and `models/mannequin.glb` each carry
  four **morph targets**: `bust`, `underbust`, `waist` and `fat` (whole-body
  fullness driven by BMI). Each has a calibration file next to it
  (`models/*.json`) with the circumferences measured on the mesh.
- Models are listed in `MODELS` in `src/mannequin-viewer.js`; add an entry
  there to offer another preview.
- `src/mannequin-viewer.js` loads it with Three.js. Height scales the whole
  model; weight sets the `fat` target from BMI; the waist, bust and underbust
  weights are then solved against the calibration measured on the mesh
  so the entered circumferences are reproduced.
- `index.html` is the page: height, weight (kg) and bust / underbust / waist
  (inches) with −/+ steppers and sliders, plus a bra-size readout. On phones
  the 3D preview stays pinned at the top while the controls scroll beneath,
  and the bra size floats on the preview.
- Bra size uses the "+2" method common in Bangladesh: band = underbust + 2
  inches, rounded up to an even number; cup = bust − band (1 = A, 2 = B, …).

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

```
pip install numpy trimesh scipy shapely networkx

# textured model: bakes the GLB and writes its calibration JSON
python3 tools/bake_bikini.py bikini_girl.glb models/bikini.glb models/bikini.json

# plain mannequin: bake, then copy the measured numbers into models/mannequin.json
python3 tools/bake_morphs.py free_female_mannequin_base_mesh.glb models/mannequin.glb
python3 tools/measure.py models/mannequin.glb
```

Both base meshes come from Sketchfab; check each licence before commercial
use.
