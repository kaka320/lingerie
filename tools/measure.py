import trimesh, numpy as np, json, struct, sys
p=sys.argv[1]
d=open(p,'rb').read(); jl=struct.unpack('<I',d[12:16])[0]; g=json.loads(d[20:20+jl])
bl=struct.unpack('<I',d[20+jl:24+jl])[0]; b=d[28+jl:28+jl+bl]
def acc(i,dt=np.float32,n=3):
    a=g['accessors'][i]; bv=g['bufferViews'][a['bufferView']]
    return np.frombuffer(b,dt,a['count']*n,bv.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,n).astype(float)
prim=g['meshes'][0]['primitives'][0]
pos=acc(prim['attributes']['POSITION']); F=acc(prim['indices'],np.uint32,1).astype(int).reshape(-1,3)
T=np.array([[1.8,0,0],[0,0,-1.8],[0,1.8,0]]); world=pos@T
def circ(mesh,y):
    sec=mesh.section(plane_origin=[0,y,0],plane_normal=[0,1,0]); p2,_=sec.to_2D()
    polys=[q for q in p2.polygons_closed if q is not None]; return max(polys,key=lambda q:q.area).length*100
m0=trimesh.Trimesh(world,F,process=False)
print(f"{'target':10s} {'underbust@1.30':>16s} {'bust@1.35':>12s}")
print(f"{'base':10s} {circ(m0,1.30):16.1f} {circ(m0,1.35):12.1f}")
for name,t in zip(g['meshes'][0]['extras']['targetNames'],prim['targets']):
    D=acc(t['POSITION'])@T; m=trimesh.Trimesh(world+D,F,process=False)
    print(f"{name:10s} {circ(m,1.30):16.1f} {circ(m,1.35):12.1f}")
