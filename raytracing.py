import numpy as np
import matplotlib.pyplot as plt
import sys

resolution = 1.5
w = 400*resolution
h = 300*resolution

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect_triangle(O, D, P, N):
    #Si ya el plano no intersecciona, pues nada
    #Le pasamos uno de los puntos del triangulo que esta en el plano
    #inters es el punto de interseccion
    inters = intersect_plane(0,D,np.array(P[0]),N);
    if (inters ==np.inf):
        return inters

    #Los vectores de la esquina del triangulo
    u = np.subtract(P[1],P[0])
    v = np.subtract(P[2],P[0])

    w0 = O - P[0]
    a = -np.dot(N,w0)
    b = np.dot(N,D)

    r = a / b
    if (r < 1e-6):
        return np.inf

    I = O + r*D
    uu = np.dot(u,u)
    uv = np.dot(u,v)
    vv = np.dot(v,v)
    w = I - P[0]
    wu = np.dot(w,u)
    wv = np.dot(w,v)
    DD = uv * uv - uu * vv


    s = (uv * wv - vv * wu) / DD
    if (s < 1e-6 or s > 1.0):
        return np.inf

    t = (uv * wu - uu * wv) / DD
    if (t < 1e-6 or (s+t) > 1.):
        return np.inf

    return 1.

def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])  
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['position'], obj['normal'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    i=0
    col_ray = ambient
    for i in range(len(lights)):
        toL = normalize(lights[i] - M)
        toO = normalize(O - M)
        # Shadow: find if the point is shadowed or not.
        l = [intersect(M + N * .0001, toL, obj_sh) 
                for k, obj_sh in enumerate(scene) if k != obj_idx]
        if l and min(l) < np.inf:
            pass;
        else:
            # Start computing the color.
            #col_ray = ambient
            # Lambert shading (diffuse).
            col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
            # Blinn-Phong shading (specular).
            col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * col_lights[i]
    return obj, M, N, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
        diffuse_c=.75, specular_c=.5, reflection=.25)
    
   
def add_triangle(p1,p2,p3, color):
    #Hay que pintarlos de menor a mayor

    #Calculamos la normal al triangulo
    u = np.subtract(p2,p1)
    v = np.subtract(p3,p1)
    #n = -abs(np.cross(u,v))
    n = (np.cross(u,v))

    #Pongo la normal negativa para que siempre miren hacia delante
    return dict(type='triangle', position=(p1,p2,p3), 
        color=np.array(color), reflection=.5,normal=n)
    
def add_triangle_concatenate(t,p1,where,color):
    #annadimos un nuevo triangulo concatenado al otro
    t1,t2,t3 = t['position'];
    if (where == 1):
        pos = (p1,t2,t3)
    elif (where== 2):
        pos = (t1,p1,t3)
    elif (where == 3):
        pos =(t1,t2,p1)
    elif (where == -1):
        pos = (p1,t3,t2)
    elif (where== -2):
        pos = (t3,p1,t1)
    elif (where == -3):
        pos =(t2,t1,p1)

    u = np.subtract(pos[1],pos[0])
    v = np.subtract(pos[2],pos[1])
    #Pongo la normal negativa para que siempre miren hacia delante
    n = np.cross(u,v)
    return dict(type='triangle', position=pos,
        color=np.array(color), reflection=.5,normal=n)


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)

if (len(sys.argv)!= 5):
    print "Usage: -scene [1-4] -lights [1-3]"
    exit()


if (sys.argv[1] == "-scene" and sys.argv[2] == "1"):
    #Original
    scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
             add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
             add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
             add_plane([0., -.5, 0.], [0., 1., 0.]),
        ]
if (sys.argv[1] == "-scene" and sys.argv[2] == "2"):
    #Triforce
    t2 = add_triangle([-1.,-0.5,1],[-0.5,0.5,1],[0.,-0.5,1],[1.,0.,0.])
    t3 = add_triangle_concatenate(t2,[0.5,0.5,1], -1, [0., 1., 0.])
    t4 = add_triangle_concatenate(t3,[1.,-0.5,1], -3, [0., 0., 1.])
    t5 = add_triangle_concatenate(t3,[0.,1.5,1], -2, [1., 1., 0.])

    scene = [t2,
             t3,
             t4,
             t5,
             add_plane([0., -.5, 0.], [0., 1., 0.]),
        ]

if (sys.argv[1] == "-scene" and sys.argv[2] == "3"):
    #Triangle too close
    scene = [
             add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
             add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
             add_plane([0., -.5, 0.], [0., 1., 0.]),
             add_triangle([-0.5,-0.5,-1.5],[0.,0.5,-1.5],[0.5,-0.5,-1.5],[1.,0.,0.]),
             add_triangle([-1.7,-0.5,1.9],[-1.2,0.5,1.5],[-0.7,-0.5,1.5],[1.,0.,-0.4]),
            add_triangle([0.5,-0.5,0.5],[1.,0.5,0.5],[1.5,-0.5,0.5],[1.,0.,0.]),

        ]
if (sys.argv[1] == "-scene" and sys.argv[2] == "4"):
    t1 = add_triangle([-1.,-0.5,1.5],[-0.5,0.5,1],[-0.2,-0.5,1],[1.,0.,0.])
    t2 = add_triangle_concatenate(t1,[0.,-0.5,1.5], -1, [0., 1., 0.])

    t3 = add_triangle([0.5,-0.5,1.5],[1.,0.9,1],[1.,-0.5,1],[1.,1.,0.])
    t4 = add_triangle_concatenate(t3,[1.5,-0.5,1.], -1, [0., 1., 1.])

    scene = [t1,
            add_plane([0., -.5, 0.], [0., 1., 0.]),
            t2,
            t3,
            t4,

        ]

# Light position and color.
#L = np.array([5., 5., -10.])
#color_light = np.array([0.,1.,0.])


if (sys.argv[3] == "-lights" and sys.argv[4] == "1"):
    #Lights array
    lights = np.array([[7., 3., -10.]])
    #Color lights array
    col_lights = np.array([[0.,1.,0.]])
if (sys.argv[3] == "-lights" and sys.argv[4] == "2"):
    #Lights array
    lights = np.array([[7., 3., -10.],[-8.,7.,-10.,]])
    #Color lights array
    col_lights = np.array([[0.,1.,0.],[1.,0.,0.]])
if (sys.argv[3] == "-lights" and sys.argv[4] == "3"):
    #Lights array
    lights = np.array([[7., 3., -10.],[-8.,7.,-10.,],[1.,15.,-3.,]])
    #Color lights array
    col_lights = np.array([[0.,1.,0.],[1.,0.,0.],[0.5,0.5,3.]])

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print i / float(w) * 100, "%"
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflection: create a new ray.
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('fig1.png', img)
