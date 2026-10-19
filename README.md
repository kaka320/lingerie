# Lingerie fit mannequin

A 3D mannequin that reshapes its bust and underbust from the customer's
measurements, for use on a lingerie e-commerce site.

## Try it

Any static server works (modules need http, not `file://`):

```
python3 -m http.server 8000
# open http://localhost:8000
```

## How it works

- `models/mannequin.glb` is the base mesh with two **morph targets** baked in:
  `bust` (pushes the breasts forward) and `underbust` (inflates the ribcage).
- `src/mannequin-viewer.js` loads it with Three.js and converts centimetre
  measurements into morph weights by solving a small 2×2 system against the
  calibration measured on the mesh (`CALIBRATION`).
- `index.html` is a demo page with inputs, sliders and a cup-size estimate.

## Embedding in your shop

```html
<div id="viewer" style="height:600px"></div>
<script type="importmap">{ "imports": {
  "three": "./vendor/three/build/three.module.js",
  "three/addons/": "./vendor/three/addons/" } }</script>
<script type="module">
  import { MannequinViewer } from './src/mannequin-viewer.js';
  const v = new MannequinViewer(document.getElementById('viewer'), { modelUrl: 'models/mannequin.glb' });
  v.ready.then(() => v.setMeasurements(75, 90)); // underbust cm, bust cm
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
