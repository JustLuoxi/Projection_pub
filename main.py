from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0.01,  exposure=1.2)
scene.set_floor(-64, (1.0,1.0,1.0))
scene.set_background_color((0/255.0,0/255.0,0/255.0))
scene.set_directional_light((0.0, 1.0,0.0), 0.0, (1, 1, 1)) # ￥
# scene.set_directional_light((-1.0, 0.0,0.0), 0.0, (1, 1, 1)) # €
# scene.set_directional_light((0.0, 0.0,1.0), 0.0, (1, 1, 1)) # $
@ti.func
def colortable(id):
    color = vec3(1.0,1.0,1.0)
    if   id==0: color.fill(rgb(228,26,28))
    elif id==1: color.fill(rgb(55,126,184))
    elif id==2: color.fill(rgb(77,175,74))
    elif id==3: color.fill(rgb(152,78,163))
    elif id==4: color.fill(rgb(255,127,0))
    elif id==5: color.fill(rgb(255,255,51))
    elif id==6: color.fill(rgb(166,86,40))
    return color
@ti.func
def rgb(r,g,b): 
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def proj_plane(o, n, t, p): 
    y = dot(p-o,n);xz=p-(o+n*y);bt=cross(t,n);
    return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def elli(rx,ry,rz,p1_unused,p2_unused,p3_unused,p):
    r = p/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def cyli(r1,h,r2,round, cone, hole_unused, p):
    ms=min(r1,min(h,r2));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(h-p.y)*0.5/h);r=vec2(p.x/r1,p.z/r2)
    d=vec2((r.norm()-1.0)*ms+rt,ti.abs(p.y)-h)+rr; return min(max(d.x,d.y),0.0)+max(d,0.0).norm()-rr<0
@ti.func
def box(x, y, z, round, cone, unused, p):
    ms=min(x,min(y,z));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(y-p.y)*0.5/y);q=ti.abs(p)-vec3(x-rt,y,z-rt)+rr
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) - rr< 0
@ti.func
def tri(r1, h, r2, round_unused, cone, vertex, p):
    r = vec3(p.x/r1, p.y, p.z/r2);rt=mix(1.0-cone,1.0,float(h-p.y)*0.5/h);r.z+=(r.x+1)*mix(-0.577, 0.577, vertex)
    q = ti.abs(r); return max(q.y-h,max(q.z*0.866025+r.x*0.5,-r.x)-0.5*rt)< 0
@ti.func
def make(func: ti.template(), p1, p2, p3, p4, p5, p6, pos, dir, up, color, mat, mode,random_c = True, add_noise=vec3(0.0)):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = proj_plane(vec3(0.0,0.0,0.0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4,p5,p6,xyz):
            if random_c:
                color=colortable(ti.round(ti.random()*7))
            if mode == 0: scene.set_voxel(pos + vec3(i,j,k), mat, color + add_noise*ti.random()) # additive
            if mode == 1: scene.set_voxel(pos + vec3(i,j,k), 0, color + add_noise*ti.random()) # subtractive
@ti.kernel
def initialize_voxels():
    # background wall
    make(box,64.0,64.0,1.0,0.1,0.0,0.0,vec3(63,0,0),vec3(-0.0,1.0,-0.0),vec3(-0.0,-0.0,-1.0),rgb(255,255,255),1,0,False)
    make(box,64.0,64.0,1.0,0.1,0.0,0.0,vec3(0,0,-63),vec3(-0.0,1.0,-0.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,0,False)
    make(box,64.0,64.0,1.0,0.1,0.0,0.0,vec3(0,-64,0),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,0,False)
    # 
    make(cyli,27.5,25.6,15.4,0.0,0.0,0.0,vec3(-27,11,23),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0),rgb(255,255,255),1,0)
    make(cyli,27.5,3.2,15.4,0.0,0.0,0.0,vec3(-27,11,53),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0),rgb(255,255,255),1,0)
    make(cyli,27.5,29.9,13.0,0.0,0.0,0.0,vec3(-29,-10,28),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0),rgb(255,255,255),1,0)
    make(box,27.5,31.6,6.8,0.0,0.0,0.0,vec3(-17,11,29),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0),rgb(255,255,255),1,1)
    make(box,23.3,31.4,4.7,0.0,0.0,0.0,vec3(-37,-10,29),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0),rgb(255,255,255),1,1)
    make(box,10.9,31.3,4.6,0.0,0.0,0.0,vec3(-46,-5,29),vec3(0.0,0.0,-1.0),vec3(-0.9,0.5,-0.0),rgb(255,255,255),1,1)
    make(box,9.8,33.2,6.9,0.0,0.0,0.0,vec3(-7,6,27),vec3(-0.0,-0.0,-1.0),vec3(-0.9,0.4,0.0),rgb(255,255,255),1,1)
    make(box,30.2,33.9,4.2,0.0,0.0,0.0,vec3(-29,-1,28),vec3(-0.0,1.0,0.0),vec3(-0.0,-0.0,1.0),rgb(255,255,255),1,0)
    make(box,5.3,33.0,33.3,0.0,0.0,0.0,vec3(-29,38,28),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0),rgb(255,255,255),1,1)
    make(box,5.3,33.0,32.0,0.0,0.0,0.0,vec3(-29,-31,29),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0),rgb(255,255,255),1,1)
    make(box,1.9,33.0,17.7,0.0,0.0,0.0,vec3(-29,29,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,1)
    make(box,1.9,33.0,12.1,0.0,0.0,0.0,vec3(-29,-24,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,1)
    make(box,1.9,33.0,2.0,0.0,0.0,0.0,vec3(-29,-2,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,1)
    make(box,3.7,33.0,12.0,0.0,0.0,0.0,vec3(-29,32,9),vec3(-1.0,-0.0,0.0),vec3(0.0,-0.9,0.4),rgb(255,255,255),1,1)
    make(box,5.3,33.0,26.1,0.0,0.0,0.0,vec3(-29,17,36),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0),rgb(255,255,255),1,1)
    make(box,4.8,33.0,25.9,0.0,0.0,0.0,vec3(-29,-13,35),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0),rgb(255,255,255),1,1)
    make(box,1.9,33.0,26.0,0.0,0.0,0.0,vec3(-29,1,36),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0),rgb(255,255,255),1,1)
    make(box,10.3,33.0,17.0,0.0,0.0,0.0,vec3(-29,5,51),vec3(-1.0,-0.0,0.0),vec3(0.0,-0.0,1.0),rgb(255,255,255),1,1)
    make(box,3.7,33.0,9.8,0.0,0.0,0.0,vec3(-29,-27,6),vec3(-1.0,-0.0,0.0),vec3(0.0,0.8,0.7),rgb(255,255,255),1,1)
    make(box,4.5,33.0,7.5,0.0,0.0,0.0,vec3(-29,28,57),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,1)
    make(box,2.9,33.0,7.5,0.0,0.0,0.0,vec3(-29,-20,59),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0),rgb(255,255,255),1,1)
    make(box,11.7,5.8,35.8,0.0,0.0,0.0,vec3(-29,4,-3),vec3(-0.0,-0.0,-1.0),vec3(1.0,0.0,-0.0),rgb(255,255,255),1,1)
    make(box,9.7,3.1,30.9,0.0,0.0,0.0,vec3(-46,-1,31),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(box,9.7,3.1,30.9,0.0,0.0,0.0,vec3(-11,-1,31),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(box,9.7,3.5,30.9,0.0,0.0,0.0,vec3(-5,-1,37),vec3(-1.0,-0.0,0.0),vec3(-0.0,-0.0,-1.0),rgb(255,255,255),1,1)
    make(box,9.7,3.8,30.9,0.0,0.0,0.0,vec3(-52,-1,37),vec3(-1.0,-0.0,0.0),vec3(-0.0,-0.0,-1.0),rgb(255,255,255),1,1)
    make(box,9.6,5.2,30.9,0.0,0.0,0.0,vec3(-11,-1,46),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(box,9.8,5.2,30.9,0.0,0.0,0.0,vec3(-46,-1,46),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(box,7.4,5.2,30.9,0.0,0.0,0.0,vec3(-49,-1,54),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(box,6.3,5.2,30.9,0.0,0.0,0.0,vec3(-7,-1,54),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(tri,3.0,36.1,15.6,0.0,0.0,0.5,vec3(-29,4,4),vec3(0.0,-1.0,0.0),vec3(0.0,0.0,1.0),rgb(255,255,255),1,1)
    make(tri,10.7,30.8,8.1,0.0,0.0,0.0,vec3(-4,-1,11),vec3(0.0,-1.0,0.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)
    make(tri,5.7,30.8,6.9,0.0,0.0,0.0,vec3(-49,-1,14),vec3(0.0,-1.0,0.0),vec3(-0.0,-0.0,-1.0),rgb(255,255,255),1,1)
    make(box,2.6,30.8,15.4,0.0,0.0,0.0,vec3(0,-1,13),vec3(0.0,-1.0,0.0),vec3(-1.0,-0.0,0.0),rgb(255,255,255),1,1)

initialize_voxels(); 
scene.finish()
