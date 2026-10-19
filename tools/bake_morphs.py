"""Bake 'bust' and 'underbust' morph targets into the mannequin GLB.

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

# The raw mesh is Z-up and 1 unit tall; the node chain scales it by 1.8 and
# rotates it to Y-up with +Z facing front.  Work in that world frame (metres):
#   world_x = 1.8*raw_x,  world_y(up) = 1.8*raw_z,  world_z(front) = -1.8*raw_y
S = 1.8
x, y, z = S * P[:, 0], S * P[:, 2], -S * P[:, 1]
def to_raw(D):  # world-frame displacement -> raw mesh frame
    return np.stack([D[:, 0] / S, -D[:, 2] / S, D[:, 1] / S], 1)

def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

# ---------- torso mask (exclude arms) ----------
# arms sit at |x| > ~0.13 and z < 0.05 at chest height; torso front is z > -0.12
torso = (np.abs(x) < 0.135) | (z > 0.06)
torso &= (y > 1.15) & (y < 1.50)

# ---------- BUST target: push each breast outward from its own centre ----------
D_bust = np.zeros_like(P)
front = smooth((z - 0.02) / 0.05)                 # only the front of the chest moves
for sx in (-1, 1):
    cx, cy = sx * 0.085, 1.355                     # breast apex (x, height)
    d = np.sqrt(((x - cx) / 0.085) ** 2 + ((y - cy) / 0.085) ** 2)
    w = smooth(1 - d) * front * torso * (sx * x > 0.0)
    dirv = np.stack([(x - cx) * 2.0, (y - cy) * 1.0 - 0.05, np.ones_like(z)], 1)
    dirv /= np.linalg.norm(dirv, axis=1, keepdims=True)
    D_bust += dirv * (w * 0.05)[:, None]           # 5 cm forward at the apex

# ---------- UNDERBUST target: inflate the ribcage radially ----------
cy_u = 1.295
w_u = smooth(1 - np.abs(y - cy_u) / 0.085) * torso
# radial direction from the torso axis in the x-z plane (axis roughly at z=0.01)
rad = np.stack([x, np.zeros_like(y), z - 0.01], 1)
rn = np.linalg.norm(rad, axis=1, keepdims=True); rad = rad / np.maximum(rn, 1e-6)
D_under = rad * (w_u * 0.022)[:, None]            # 2.2 cm radial growth (~14 cm circumference)

# ---------- normal deltas ----------
def vertex_normals(V):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for i in range(3): np.add.at(vn, F[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
N0 = vertex_normals(P)
targets = []
for name, D in (('bust', to_raw(D_bust)), ('underbust', to_raw(D_under))):
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
