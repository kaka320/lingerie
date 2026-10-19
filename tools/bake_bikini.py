"""Bake 'bust', 'underbust', 'waist' and 'fat' morph targets into the bikini model GLB
and write its calibration (circumferences in cm) next to it.

Usage: python bake_bikini.py input.glb models/bikini.glb models/bikini.json

The source mesh is a single watertight shell, Y-up, 1 unit tall, +Z front, centred
on the origin. We work in a world frame scaled to 1.70 m with the feet at y = 0.
"""
import json, struct, sys
import numpy as np
import scipy.sparse as sp
import trimesh

src, dst, calib_out = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(src, 'rb').read()
jlen = struct.unpack('<I', data[12:16])[0]
gltf = json.loads(data[20:20 + jlen])
blen = struct.unpack('<I', data[20 + jlen:24 + jlen])[0]
bin_ = bytearray(data[28 + jlen:28 + jlen + blen])
mesh = gltf['meshes'][0]
prim = mesh['primitives'][0]

def read_acc(idx):
    a = gltf['accessors'][idx]; bv = gltf['bufferViews'][a['bufferView']]
    n = {'VEC4': 4, 'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[a['componentType']]
    stride = bv.get('byteStride', 0)
    assert stride in (0, n * np.dtype(dt).itemsize), 'interleaved buffers not supported'
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(bin_, dtype=dt, count=a['count'] * n, offset=off).reshape(a['count'], n).astype(np.float64)

Praw = read_acc(prim['attributes']['POSITION'])
Nraw = read_acc(prim['attributes']['NORMAL'])
F = read_acc(prim['indices']).astype(np.int64).reshape(-1, 3)

H = 1.70
P = np.stack([Praw[:, 0] * H, (Praw[:, 1] + 0.5) * H, Praw[:, 2] * H], 1)
x, y, z = P[:, 0], P[:, 1], P[:, 2]
to_raw = lambda D: D / H                     # same axes, uniform scale

def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
def unit(V): return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
def band(lo, hi, feather=0.05):
    return smooth((y - lo) / feather + 0.5) * smooth((hi - y) / feather + 0.5)

# ---------- welded topology (UV seams duplicate vertices) ----------
key = np.round(P * 1e5).astype(np.int64)
_, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
G = inv.max() + 1
Fg = inv[F]
rows = np.concatenate([Fg[:, 0], Fg[:, 1], Fg[:, 2], Fg[:, 1], Fg[:, 2], Fg[:, 0]])
cols = np.concatenate([Fg[:, 1], Fg[:, 2], Fg[:, 0], Fg[:, 0], Fg[:, 1], Fg[:, 2]])
A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(G, G)).tocsr()
A.data[:] = 1.0
A = A + sp.identity(G, format='csr')
A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A          # row-normalised neighbour average

def group_mean(V):
    s = np.zeros((G,) + V.shape[1:]); np.add.at(s, inv, V)
    c = np.bincount(inv, minlength=G).astype(float)
    return s / c.reshape((-1,) + (1,) * (V.ndim - 1))
def laplace(V, iters):
    """Smooth a per-vertex field over the welded mesh surface (geodesic, so the arms
    never mix with the head or the legs with each other)."""
    Vg = group_mean(V)
    for _ in range(iters): Vg = A @ Vg
    return Vg[inv]

def welded_normals(V):
    Vg = group_mean(V)
    fn = np.cross(Vg[Fg[:, 1]] - Vg[Fg[:, 0]], Vg[Fg[:, 2]] - Vg[Fg[:, 0]])
    n = np.zeros((G, 3))
    for i in range(3): np.add.at(n, Fg[:, i], fn)
    return unit(n)[inv]

N0 = welded_normals(P)
NS = unit(laplace(N0, 25))                    # very smooth outward normals

# ---------- landmarks (metres, world frame) ----------
Y_CROTCH, Y_WAIST, Y_UNDER, Y_BUST, Y_ARMPIT = 0.86, 1.13, 1.20, 1.265, 1.31
STATURE = 1.676                               # head top

# torso axis: centroid of each height slice between crotch and armpit (no arms there)
step = 0.01
ys = np.arange(Y_CROTCH, Y_ARMPIT + step, step)
ax_x = np.zeros(len(ys)); ax_z = np.zeros(len(ys))
for i, yy in enumerate(ys):
    s = (np.abs(y - yy) < step) & (np.abs(x) < 0.22)
    ax_x[i], ax_z[i] = x[s].mean(), z[s].mean()
k = np.ones(7) / 7
ax_x = np.convolve(np.pad(ax_x, 3, mode='edge'), k, 'valid'); ax_z = np.convolve(np.pad(ax_z, 3, mode='edge'), k, 'valid')
AX = np.stack([np.interp(y, ys, ax_x), np.interp(y, ys, ax_z)], 1)
RAD_T = unit(np.stack([x - AX[:, 0], np.zeros_like(y), z - AX[:, 1]], 1))
in_torso = smooth((y - Y_CROTCH + 0.02) / 0.04) * smooth((0.20 - np.abs(x)) / 0.03)
frontness = smooth((z - AX[:, 1]) / 0.05)
backness = smooth((AX[:, 1] - z) / 0.05)

# breast centres from the front-most bulge on each side
BREAST = {}
for sx in (-1, 1):
    s = (sx * x > 0.02) & (sx * x < 0.17) & (y > 1.19) & (y < 1.35)
    zt = np.percentile(z[s], 80)
    s2 = s & (z > zt)
    BREAST[sx] = np.array([x[s2].mean(), y[s2].mean(), z[s2].mean() - 0.085])

# ---------- BUST: fuller breasts ----------
D_bust = np.zeros_like(P)
for sx, c in BREAST.items():
    ry = np.where(y > c[1], 0.11, 0.07)
    dxy = np.sqrt(((x - c[0]) / 0.10) ** 2 + ((y - c[1]) / ry) ** 2)      # distance across the front
    w = smooth(1 - dxy) ** 1.2 * smooth((z - c[2]) / 0.04) * (sx * x > 0.0)
    rad = unit(P - c)
    rad[:, 0] = sx * np.maximum(sx * rad[:, 0], 0.0)                     # never into the cleavage
    fwd = unit(np.tile([0.0, -0.3, 1.0], (len(P), 1)))
    a = w[:, None]
    D_bust += unit(rad * (1 - 0.5 * a) + fwd * (0.6 * a)) * (w * 0.07)[:, None]
D_bust = laplace(D_bust, 4)

# ---------- UNDERBUST: wider ribcage ----------
w_u = np.where(y > Y_UNDER, smooth(1 - (y - Y_UNDER) / 0.06), smooth(1 - (Y_UNDER - y) / 0.08)) * in_torso
D_under = laplace(RAD_T * (w_u * 0.03)[:, None], 6)

# ---------- WAIST: wider natural waist ----------
w_w = smooth(1 - np.abs(y - Y_WAIST) / 0.10) * in_torso
D_waist = laplace(RAD_T * (w_w * (1.0 + 0.5 * frontness) * 0.03)[:, None], 6)

# ---------- FAT: realistic weight distribution along smooth normals ----------
is_arm = smooth((y - Y_ARMPIT + 0.01) / 0.03) * smooth((np.abs(x) - 0.10) / 0.03)
is_head = smooth((y - 1.45) / 0.04) * (1 - is_arm)
prof_torso = (1.25 * band(0.84, 1.02)        # hips / buttocks
            + 1.10 * band(1.02, 1.17)        # waist / belly
            + 0.85 * band(1.17, 1.23)        # ribs
            + 0.60 * band(1.23, 1.34)        # bust
            + 0.50 * band(1.34, 1.45)        # shoulders / upper chest
            + 0.30 * band(1.45, 1.52))       # neck
prof_torso *= (1 + 0.7 * frontness * band(1.00, 1.20))      # belly
prof_torso *= (1 + 0.4 * backness * band(0.86, 1.04))       # buttocks
prof_leg = (1.15 * band(0.62, 0.88)          # thighs
          + 0.55 * band(0.42, 0.62)          # knees
          + 0.50 * band(0.12, 0.42)          # calves
          + 0.08 * band(-0.1, 0.12))         # ankles / feet
dist_shoulder = np.linalg.norm(np.stack([np.abs(x) - 0.16, y - 1.38], 1), axis=1)
prof_arm = 0.9 * smooth(1 - dist_shoulder / 0.32) + 0.15         # upper arm -> forearm
legs = y < Y_CROTCH
profile = np.where(legs, prof_leg, prof_torso)
profile = profile * (1 - is_arm) + prof_arm * is_arm
profile = profile * (1 - is_head) + 0.12 * is_head                  # head and hair barely change
profile = laplace(profile[:, None], 15)[:, 0]

dirv = NS.copy()
inner = legs & (np.sign(x) * dirv[:, 0] < 0)                       # inner thighs
dirv[inner, 0] *= 0.2
D_fat = laplace(unit(dirv) * (profile * 0.03)[:, None], 10)        # 3 cm at profile 1.0

# ---------- write GLB ----------
TARGETS = [('bust', D_bust), ('underbust', D_under), ('waist', D_waist), ('fat', D_fat)]

def add_vec3(arr):
    while len(bin_) % 4: bin_.append(0)
    off = len(bin_); raw = arr.astype(np.float32).tobytes(); bin_.extend(raw)
    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': off, 'byteLength': len(raw)})
    gltf['accessors'].append({'bufferView': len(gltf['bufferViews']) - 1, 'componentType': 5126,
                              'count': len(arr), 'type': 'VEC3',
                              'min': arr.min(0).astype(float).tolist(), 'max': arr.max(0).astype(float).tolist()})
    return len(gltf['accessors']) - 1

prim['targets'] = []
mesh['extras'] = {'targetNames': []}
for name, D in TARGETS:
    dN = welded_normals(P + D) - N0
    prim['targets'].append({'POSITION': add_vec3(to_raw(D)), 'NORMAL': add_vec3(dN)})
    mesh['extras']['targetNames'].append(name)
mesh['weights'] = [0.0] * len(TARGETS)
gltf['buffers'][0]['byteLength'] = len(bin_)
jb = json.dumps(gltf, separators=(',', ':')).encode()
while len(jb) % 4: jb += b' '
while len(bin_) % 4: bin_.append(0)
out = b'glTF' + struct.pack('<II', 2, 12 + 8 + len(jb) + 8 + len(bin_))
out += struct.pack('<I', len(jb)) + b'JSON' + jb + struct.pack('<I', len(bin_)) + b'BIN\x00' + bytes(bin_)
open(dst, 'wb').write(out)
print('wrote', dst, round(len(out) / 1e6, 2), 'MB')

# ---------- calibration ----------
def circ(V, yy):
    m = trimesh.Trimesh(V, F, process=False)
    sec = m.section(plane_origin=[0, yy, 0], plane_normal=[0, 1, 0]); p2, _ = sec.to_2D()
    polys = [q for q in p2.polygons_closed if q is not None]
    return max(polys, key=lambda q: q.area).length * 100
planes = {'underbust': Y_UNDER, 'bust': Y_BUST, 'waist': Y_WAIST}
def measure(V): return {k: round(circ(V, v), 1) for k, v in planes.items()}
base = measure(P)
calib = {
    'statureCm': round(STATURE * 100, 1),     # height of the figure in calibration cm
    'statureWorld': round(STATURE / H, 4),    # same height in GLB world units
    'baseBmi': 17.5,
    'bmiPerFatUnit': 7,
    'base': base,
    'range': {'bust': [-0.5, 3.0], 'underbust': [-0.8, 3.0], 'waist': [-1.0, 3.5], 'fat': [0, 3.0]},
}
print('base', base)
for name, D in TARGETS:
    m1 = measure(P + D)
    calib[name + 'T'] = {k: round(m1[k] - base[k], 2) for k in base}
    print(name, calib[name + 'T'])
json.dump(calib, open(calib_out, 'w'), indent=2)
print('wrote', calib_out)
