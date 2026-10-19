"""Bake 'bust', 'underbust', 'waist' and 'fat' morph targets into the mannequin GLB.

Usage: python bake_morphs.py input.glb output.glb
"""
import json, struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
data = open(src, 'rb').read()
assert data[:4] == b'glTF'
jlen = struct.unpack('<I', data[12:16])[0]
gltf = json.loads(data[20:20 + jlen])
blen = struct.unpack('<I', data[20 + jlen:24 + jlen])[0]
bin_ = bytearray(data[28 + jlen:28 + jlen + blen])

mesh = gltf['meshes'][0]
prim = mesh['primitives'][0]

def read_acc(idx):
    a = gltf['accessors'][idx]; bv = gltf['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    n = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[a['componentType']]
    return np.frombuffer(bin_, dtype=dt, count=a['count'] * n, offset=off).reshape(a['count'], n).astype(np.float64)

P = read_acc(prim['attributes']['POSITION'])
N = read_acc(prim['attributes']['NORMAL'])
F = read_acc(prim['indices']).astype(np.int64).reshape(-1, 3)
print('verts', len(P), 'bounds', P.min(0).round(3), P.max(0).round(3))
from scipy.spatial import cKDTree
_tree = cKDTree(P)
_nbrs = _tree.query_ball_point(P, r=0.009)        # ~1.6 cm neighbourhoods (raw units, mesh is 1.0 tall)
def smooth_field(V):
    """Average a per-vertex vector field over spatial neighbours (position based, so
    the mesh's overlapping patches get identical values)."""
    out = np.zeros_like(V)
    for i, nb in enumerate(_nbrs): out[i] = V[nb].mean(0)
    return out
def unit(V): return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


# The raw mesh is Z-up and 1 unit tall; the node chain scales it by 1.8 and
# rotates it to Y-up with +Z facing front.  Work in that world frame (metres):
#   world_x = 1.8*raw_x,  world_y(up) = 1.8*raw_z,  world_z(front) = -1.8*raw_y
S = 1.8
x, y, z = S * P[:, 0], S * P[:, 2], -S * P[:, 1]
def to_raw(D):  # world-frame displacement -> raw mesh frame
    return np.stack([D[:, 0] / S, -D[:, 2] / S, D[:, 1] / S], 1)

def outward_dir():
    """Rough outward direction for every vertex: away from the torso, arm or leg axis.
    Used only to orient normals, because the mesh is a double shell (inner surface has
    inward-pointing normals)."""
    cx = np.zeros_like(x); cz = np.zeros_like(z)
    legs = (y < 0.98) & (np.abs(x) < 0.17)
    arms = (np.abs(x) > 0.15) & (y > 0.85) & (y < 1.52) & ~legs
    cx[legs] = np.sign(x[legs]) * 0.09
    cx[arms] = np.sign(x[arms]) * 0.215; cz[arms] = -0.02
    return unit(np.stack([x - cx, np.zeros_like(y), z - cz], 1))
OUT = outward_dir()
def orient(Nv):
    """Flip normals that point inward, then smooth spatially."""
    flip = (Nv * OUT).sum(1) < 0
    Nv = Nv.copy(); Nv[flip] *= -1
    return unit(smooth_field(Nv))
def to_world_dir(V): return np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)
def to_raw_dir(V):   return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

# ---------- torso mask (exclude arms) ----------
# arms sit at |x| > ~0.13 and z < 0.05 at chest height; torso front is z > -0.12
torso = (np.abs(x) < 0.135) | (z > 0.06)
torso &= (y > 1.15) & (y < 1.50)

# ---------- BUST target: push each breast outward from its own centre ----------
D_bust = np.zeros_like(P)
front = smooth((z - 0.03) / 0.05)                 # only the front of the chest moves
lateral = smooth((0.175 - np.abs(x)) / 0.045)     # fade out smoothly before the arms
for sx in (-1, 1):
    # breast centre sits inside the chest, a little below the apex
    cx, cy, cz = sx * 0.085, 1.345, 0.045
    # wide ellipsoidal region of influence
    ry = np.where(y > cy, 0.14, 0.075)             # reaches further up than down
    d = np.sqrt(((x - cx) / 0.11) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / 0.17) ** 2)
    w = smooth(1 - d) ** 1.5 * front * lateral * (y > 1.15) * (y < 1.55) * (sx * x > 0.005)
    # inflate radially from the centre (makes the whole breast fuller) ...
    rad = np.stack([x - cx, y - cy, z - cz], 1)
    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-6)
    rad[:, 0] = sx * np.maximum(sx * rad[:, 0], 0.0)   # never push toward the sternum
    # ... blended with forward projection + slight sag near the apex only
    extra = np.stack([np.zeros_like(x), -0.3 * np.ones_like(y), np.ones_like(z)], 1)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    a = w[:, None]                                    # 0 at the edge, ~1 at the apex
    dirv = rad * (1 - 0.6 * a) + extra * (0.7 * a)
    dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
    D_bust += dirv * (w * 0.085)[:, None]           # ~8.5 cm growth at full weight

# ---------- UNDERBUST target: inflate the ribcage radially ----------
cy_u = 1.295
w_u = smooth(1 - np.abs(y - cy_u) / 0.085) * torso
# radial direction from the torso axis in the x-z plane (axis roughly at z=0.01)
rad = np.stack([x, np.zeros_like(y), z - 0.01], 1)
rn = np.linalg.norm(rad, axis=1, keepdims=True); rad = rad / np.maximum(rn, 1e-6)
D_under = rad * (w_u * 0.022)[:, None]            # 2.2 cm radial growth (~14 cm circumference)

# ---------- limb axes (positional, independent of normals) ----------
# Each vertex grows outward from the axis of its limb (torso, arm or leg). The
# three fields are blended smoothly across the crotch and armpits so there are
# no seams. Being purely positional, overlapping patches of this mesh move
# together and the inner shell stays consistent with the outer one.
step = 0.01
ys = np.arange(y.min() - step, y.max() + 2 * step, step)
sl = np.clip(np.round((y - ys[0]) / step).astype(int), 0, len(ys) - 1)
side = np.sign(x); side[side == 0] = 1

def axis_curve(sel):
    """Smoothed (x, z) centroid curve of the selected vertices by height, evaluated for ALL vertices."""
    cx = np.full(len(ys), np.nan); cz = np.full(len(ys), np.nan)
    for i in np.unique(sl[sel]):
        g = sel & (sl == i); cx[i] = x[g].mean(); cz[i] = z[g].mean()
    ok = ~np.isnan(cx)
    cx = np.interp(ys, ys[ok], cx[ok]); cz = np.interp(ys, ys[ok], cz[ok])
    k = np.ones(9) / 9
    cx = np.convolve(np.pad(cx, 4, mode='edge'), k, 'valid'); cz = np.convolve(np.pad(cz, 4, mode='edge'), k, 'valid')
    return np.stack([np.interp(y, ys, cx), np.interp(y, ys, cz)], 1)

def radial(axis):
    return unit(np.stack([x - axis[:, 0], np.zeros_like(y), z - axis[:, 1]], 1))

# torso / head axis
RAD_T = radial(axis_curve((np.abs(x) < 0.17) & (y >= 0.93) | (y >= 1.42)))
# legs and arms: one curve per side, picked by the vertex's own side
RAD_L = np.zeros((len(P), 3)); RAD_A = np.zeros((len(P), 3))
for sd in (-1, 1):
    m = side == sd
    RAD_L[m] = radial(axis_curve((y < 0.93) & (np.abs(x) < 0.17) & m))[m]
    arm_sel = ((y < 0.93) & (np.abs(x) >= 0.17) | (y >= 0.93) & (y < 1.42) & (np.abs(x) > 0.175)) & m
    RAD_A[m] = radial(axis_curve(arm_sel))[m]
# inner thighs never push into the other leg
inner = np.sign(x) * RAD_L[:, 0] < 0
RAD_L[inner, 0] *= 0.15; RAD_L = unit(RAD_L)

# blend weights
W_L = (1 - smooth((y - 0.88) / 0.10)) * smooth((0.19 - np.abs(x)) / 0.04)
W_A = smooth((np.abs(x) - 0.16) / 0.035) * (1 - smooth((y - 1.40) / 0.06)) * smooth((y - 0.55) / 0.05)
W_A = np.maximum(W_A, (y < 0.93) * smooth((np.abs(x) - 0.16) / 0.035))      # hands
W_L = W_L * (1 - W_A)
W_T = np.clip(1 - W_L - W_A, 0, 1)
RAD = unit(RAD_T * W_T[:, None] + RAD_L * W_L[:, None] + RAD_A * W_A[:, None])

def band(lo, hi, feather=0.06):
    return smooth((y - lo) / feather + 0.5) * smooth((hi - y) / feather + 0.5)
frontness = smooth(z / 0.05)                  # 1 on the front, 0 on the back
backness = smooth(-z / 0.05)

# ---------- WAIST target: widen the waist ----------
cy_w = 1.15                                   # natural waist, just above the navel
w_w = smooth(1 - np.abs(y - cy_w) / 0.11) * W_T
D_waist = RAD * (w_w * (1.0 + 0.5 * frontness) * 0.022)[:, None]

# ---------- BODY FAT target: realistic weight distribution ----------
prof_torso = (0.20 * band(1.60, 1.90)         # head
            + 0.40 * band(1.52, 1.60)         # neck
            + 0.60 * band(1.38, 1.52)         # shoulders / upper chest
            + 0.65 * band(1.28, 1.38)         # bust
            + 0.90 * band(1.18, 1.28)         # ribs
            + 1.10 * band(0.98, 1.18)         # waist / belly
            + 1.25 * band(0.80, 0.98))        # hips / buttocks
prof_torso = prof_torso * (1 + 0.7 * frontness * band(0.95, 1.22))   # belly pooch
prof_torso = prof_torso * (1 + 0.4 * backness * band(0.82, 1.02))    # buttocks
prof_arm = (0.95 * band(1.22, 1.52)           # upper arm
          + 0.55 * band(0.98, 1.22)           # forearm
          + 0.15 * band(0.60, 0.98))          # hands
prof_leg = (1.15 * band(0.78, 1.00)           # upper thigh
          + 0.95 * band(0.60, 0.78)           # thigh
          + 0.55 * band(0.45, 0.60)           # knee
          + 0.55 * band(0.22, 0.45)           # calf
          + 0.20 * band(0.08, 0.22)           # ankle
          + 0.05 * band(-0.1, 0.08))          # feet
profile = prof_torso * W_T + prof_arm * W_A + prof_leg * W_L
profile = smooth_field(profile[:, None])[:, 0]  # soften region borders
D_fat = RAD * (profile * 0.028)[:, None]        # 2.8 cm at profile 1.0

# ---------- normal deltas ----------
def vertex_normals(V):
    """Vertex normals (raw frame) from faces, with the authored normal's orientation."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.zeros_like(V)
    for i in range(3): np.add.at(a, F[:, i], fn)
    a = unit(a)
    flip = (a * N).sum(1) < 0
    a[flip] *= -1
    return a
N0 = vertex_normals(P)
targets = []
for name, D in (('bust', to_raw(D_bust)), ('underbust', to_raw(D_under)), ('waist', to_raw(D_waist)), ('fat', to_raw(D_fat))):
    Nn = vertex_normals(P + D) - N0
    targets.append((name, D.astype(np.float32), Nn.astype(np.float32)))

# ---------- append to GLB ----------
def add_vec3(arr):
    while len(bin_) % 4: bin_.append(0)
    off = len(bin_); raw = arr.astype(np.float32).tobytes(); bin_.extend(raw)
    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': off, 'byteLength': len(raw)})
    gltf['accessors'].append({'bufferView': len(gltf['bufferViews']) - 1, 'componentType': 5126,
                              'count': len(arr), 'type': 'VEC3',
                              'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()})
    return len(gltf['accessors']) - 1

prim['targets'] = []
mesh['extras'] = {'targetNames': []}
for name, D, Nn in targets:
    prim['targets'].append({'POSITION': add_vec3(D), 'NORMAL': add_vec3(Nn)})
    mesh['extras']['targetNames'].append(name)
mesh['weights'] = [0.0] * len(targets)
gltf['buffers'][0]['byteLength'] = len(bin_)

jb = json.dumps(gltf, separators=(',', ':')).encode()
while len(jb) % 4: jb += b' '
while len(bin_) % 4: bin_.append(0)
out = b'glTF' + struct.pack('<II', 2, 12 + 8 + len(jb) + 8 + len(bin_))
out += struct.pack('<I', len(jb)) + b'JSON' + jb
out += struct.pack('<I', len(bin_)) + b'BIN\x00' + bytes(bin_)
open(dst, 'wb').write(out)
print('wrote', dst, len(out) / 1e6, 'MB')
