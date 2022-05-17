from cv2 import sqrt
from sqlalchemy import false
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
    if   id==0: color.fill((228/255.0   ,26/255.0,  28/255.0))
    elif id==1: color.fill((55/255.0    ,126/255.0, 184/255.0))
    elif id==2: color.fill((77/255.0    ,175/255.0, 74/255.0))
    elif id==3: color.fill((152/255.0   ,78/255.0,  163/255.0))
    elif id==4: color.fill((255/255.0   ,127/255.0, 0/255.0))
    elif id==5: color.fill((255/255.0   ,255/255.0, 51/255.0))
    elif id==6: color.fill((166/255.0   ,86/255.0,  40/255.0))
    return color
@ti.func
def world2local(lookat,up,p):
    left = cross(up,lookat)
    mat = mat3([[up.x,up.y,up.z],[lookat.x,lookat.y,lookat.z],[left.x,left.y,left.z]])
    p = mat@p
    return p
@ti.func
def box(x, y, z, occupied, p):
    return ti.abs(p.x)<x and ti.abs(p.y)<y and ti.abs(p.z)<z
@ti.func
def cylinder(r1,h,r2,occupied, p):
    r=vec2(p.x/r1,p.z/r2).norm() 
    return r<1 and ti.abs(p.y)<h
@ti.func
def vecproduct(v1,v2):
    return v1.x*v2.y - v1.y*v2.x
@ti.func
def triangle(r1, h, r2, vertex, p):
    o=vec2(p.x,p.z);A=vec2(-r1,-r2);B=vec2(-r1,r2);C=vec2(r1,r2-2.0*vertex*r2)
    return vecproduct(o-A,B-A)>0 and vecproduct(o-B,C-B)>0 and vecproduct(o-C,A-C)>0 and ti.abs(p.y)<h
@ti.func
def render(shape: ti.template(),x1,x2,x3,x4,pos,lookat,up,mat,mode,random_c=True,add_noise=vec3(0.0)):
    lookat = normalize(lookat); up = normalize(cross(cross(lookat, up), lookat))
    radius = 2 * int(max(x3,max(x1, x2)))
    for i,j,k in ti.ndrange((-radius,radius),(-radius,radius),(-radius,radius)): 
        x_local = world2local(lookat, up, vec3(i,j,k))
        if shape(x1,x2,x3,x4,x_local):
            color = vec3(1.0,1.0,1.0)
            if random_c:
                color=colortable(ti.round(ti.random()*7))
            if mode == 0: scene.set_voxel(pos + vec3(i,j,k), mat, color + add_noise*ti.random()) # additive
            if mode == 1: scene.set_voxel(pos + vec3(i,j,k), 0, color + add_noise*ti.random()) # subtractive

@ti.kernel
def initialize_voxels():
    # background wall
    render(box,64.0,64.0,1.0,0.0,vec3(63,0,0),vec3(-0.0,1.0,-0.0),vec3(-0.0,-0.0,-1.0), 1,0,False)
    render(box,64.0,64.0,1.0,0.0,vec3(0,0,-63),vec3(-0.0,1.0,-0.0),vec3(-1.0,-0.0,0.0), 1,0,False)
    render(box,64.0,64.0,1.0,0.0,vec3(0,-64,0),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,0,False)
    # main body
    render(cylinder,27.5,25.6,15.4,0.0,vec3(-27,11,23),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0), 1,0)
    render(cylinder,27.5,3.2,15.4,0.0,vec3(-27,11,53),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0), 1,0)
    render(cylinder,27.5,29.9,13.0,0.0,vec3(-29,-10,28),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0), 1,0)
    render(box,27.5,31.6,6.8,0.0,vec3(-17,11,29),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0), 1,1)
    render(box,23.3,31.4,4.7,0.0,vec3(-37,-10,29),vec3(0.0,0.0,-1.0),vec3(-1.0,0.0,-0.0), 1,1)
    render(box,10.9,31.3,4.6,0.0,vec3(-46,-5,29),vec3(0.0,0.0,-1.0),vec3(-0.9,0.5,-0.0), 1,1)
    render(box,9.8,33.2,6.9,0.0,vec3(-7,6,27),vec3(-0.0,-0.0,-1.0),vec3(-0.9,0.4,0.0), 1,1)
    render(box,30.2,33.9,4.2,0.0,vec3(-29,-1,28),vec3(-0.0,1.0,0.0),vec3(-0.0,-0.0,1.0), 1,0)
    render(box,5.3,33.0,33.3,0.0,vec3(-29,38,28),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0), 1,1)
    render(box,5.3,33.0,32.0,0.0,vec3(-29,-31,29),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0), 1,1)
    render(box,1.9,33.0,17.7,0.0,vec3(-29,29,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,1)
    render(box,1.9,33.0,12.1,0.0,vec3(-29,-24,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,1)
    render(box,1.9,33.0,2.0,0.0,vec3(-29,-2,-1),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,1)
    render(box,3.7,33.0,12.0,0.0,vec3(-29,32,9),vec3(-1.0,-0.0,0.0),vec3(0.0,-0.9,0.4), 1,1)
    render(box,5.3,33.0,26.1,0.0,vec3(-29,17,36),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0), 1,1)
    render(box,4.8,33.0,25.9,0.0,vec3(-29,-13,35),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0), 1,1)
    render(box,1.9,33.0,26.0,0.0,vec3(-29,1,36),vec3(-1.0,-0.0,0.0),vec3(0.0,-1.0,-0.0), 1,1)
    render(box,10.3,33.0,17.0,0.0,vec3(-29,5,51),vec3(-1.0,-0.0,0.0),vec3(0.0,-0.0,1.0), 1,1)
    render(box,3.7,33.0,9.8,0.0,vec3(-29,-27,6),vec3(-1.0,-0.0,0.0),vec3(0.0,0.8,0.7), 1,1)
    render(box,4.5,33.0,7.5,0.0,vec3(-29,28,57),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,1)
    render(box,2.9,33.0,7.5,0.0,vec3(-29,-20,59),vec3(-1.0,-0.0,0.0),vec3(-0.0,0.0,-1.0), 1,1)
    render(box,11.7,5.8,35.8,0.0,vec3(-29,4,-3),vec3(-0.0,-0.0,-1.0),vec3(1.0,0.0,-0.0), 1,1)
    render(box,9.7,3.1,30.9,0.0,vec3(-46,-1,31),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(box,9.7,3.1,30.9,0.0,vec3(-11,-1,31),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(box,9.7,3.5,30.9,0.0,vec3(-5,-1,37),vec3(-1.0,-0.0,0.0),vec3(-0.0,-0.0,-1.0), 1,1)
    render(box,9.7,3.8,30.9,0.0,vec3(-52,-1,37),vec3(-1.0,-0.0,0.0),vec3(-0.0,-0.0,-1.0), 1,1)
    render(box,9.6,5.2,30.9,0.0,vec3(-11,-1,46),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(box,9.8,5.2,30.9,0.0,vec3(-46,-1,46),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(box,7.4,5.2,30.9,0.0,vec3(-49,-1,54),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(box,6.3,5.2,30.9,0.0,vec3(-7,-1,54),vec3(0.0,0.0,1.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(triangle,3.0,36.1,15.6,0.5,vec3(-29,4,4),vec3(0.0,-1.0,0.0),vec3(0.0,0.0,1.0), 1,1)
    render(triangle,10.7,30.8,8.1,0.0,vec3(-4,-1,11),vec3(0.0,-1.0,0.0),vec3(-1.0,-0.0,0.0), 1,1)
    render(triangle,5.7,30.8,6.9,0.0,vec3(-49,-1,14),vec3(0.0,-1.0,0.0),vec3(-0.0,-0.0,-1.0), 1,1)
    render(box,2.6,30.8,15.4,0.0,vec3(0,-1,13),vec3(0.0,-1.0,0.0),vec3(-1.0,-0.0,0.0), 1,1)

initialize_voxels()
scene.finish()
