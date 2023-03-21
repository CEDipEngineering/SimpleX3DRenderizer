
from typing import List, Tuple
import numpy as np
import math

class CustomPoint2D():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def get_pixel(self):
        return (int(self.x), int(self.y))


    # Methods for debug printing
    def __str__(self) -> str:
        return "({:.1f},{:.1f})".format(self.x, self.y)
    
    def __repr__(self) -> str:
        return self.__str__()
    
class CustomPoint3D():
    def __init__(self, x, y, z, w=1) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __sub__(self, other):
        if other is CustomPoint3D:
            return CustomPoint3D(self.x-other.x, self.y-other.y, self.z-other.z)
        raise TypeError("CustomPoint3D only supports operations with other CustomPoint3D obejcts.")
    
    def __add__(self, other):
        if other is CustomPoint3D:
            return CustomPoint3D(self.x+other.x, self.y+other.y, self.z+other.z)
        raise TypeError("CustomPoint3D only supports operations with other CustomPoint3D obejcts.")
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        if i == 2:
            return self.z
        raise IndexError("Invalid index passed, must be 0,1,2 to access x,y,z coordinates.")
    
    def __setitem__(self, i, v):
        if i == 0:
            self.x = v
        if i == 1:
            self.y = v
        if i == 2:
            self.z = v
        raise IndexError("Invalid index passed, must be 0,1,2 to access x,y,z coordinates.")
    
    def get_homo(self):
        return (*self.get_pixel(), 1)

    def get_pixel(self):
        return (int(self.x), int(self.y), int(self.z))
    
    def homogeneous_division(self):
        "Transforms inplace, but also returns self, just in case its used mid-operation"
        self.x /= self.w
        self.y /= self.w
        self.z /= self.w
        self.w /= self.w
        return self

    # Methods for debug printing
    def __str__(self) -> str:
        return "({:.1f},{:.1f},{:.1f})".format(self.x, self.y, self.z)
    
    def __repr__(self) -> str:
        return self.__str__()
    
def get_emissive_rgb(color):
    # Initially using only emissive colors, later will use rest of color information
    c = color["emissiveColor"]
    return [int(255 * i) for i in c]

def reshape_points2D(row) -> List[CustomPoint2D]:
    # Row always has even number of values, each even index corresponds to x, each odd index corresponds to y.
    out = []
    for x, y in zip(row[::2],row[1::2]):
        out.append(CustomPoint2D(x,y))
    return out

def reshape_points3D(row) -> List[CustomPoint3D]:
    # Row always has even number of values, each even index corresponds to x, each odd index corresponds to y.
    # out = []
    # for x, y, z in zip(row[::3], row[1::3], row[2::3]):
    #     out.append(CustomPoint3D(x,y,z))
    # return out
    arr = np.array(row)
    return np.reshape(arr, (-1, 3)).T

def draw_line(p0: CustomPoint2D, p1: CustomPoint2D) -> List[CustomPoint2D]:
    """
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham's%20line%20algorithm%20is%20a,straight%20line%20between%20two%20points.
    Implementation based on Bresenham algorithm using Wikipedia article as basis for solution
    """
    def plotLineLow(x1, y1, x0, y0):
        line = [] # Array de saida dos pontos desejados
        dx = x1-x0 
        dy = y1-y0
        yi = 1
        # Se y0 > y1, vamos desenhar ao contrario, so muda o passo, e inverte a diferenca
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2*dy - dx
        y = y0
        # Passo a passo, mantendo passo de x com tamanho 1, e inferindo se y muda ou nao, e quanto.
        for x in range(x0, x1+1):
            line.append(CustomPoint2D(x, y))
            if D > 0:
                y = y + yi
                D = D + (2 * (dy - dx))
            else:
                D = D + 2*dy
        return line

    # Repetir procedimento para x trocado com y, para pega inclinacoes de sinais diferentes
    def plotLineHigh(x1, y1, x0, y0):
        print(x0,y0)
        print(x1,y1)
        line = []
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x0

        for y in range(y0, y1+1):
            line.append(CustomPoint2D(x, y))
            if D > 0:
                x = x + xi
                D = D + (2 * (dx - dy))
            else:
                D = D + 2*dx
        return line

    x0, y0 = p0.get_pixel()
    x1, y1 = p1.get_pixel()
    
    # Teste para usar High vs. Low
    if abs(y1 - y0) < abs(x1 - x0):
        print("Low")
        if x0 < x1:
            print("reverse")
            line = plotLineLow(x1, y1, x0, y0)
        else:
            print("normal")
            line = plotLineLow(x0, y0, x1, y1)
    else:
        print("High")
        if y0 < y1:
            print("reverse")
            line = plotLineHigh(x1, y1, x0, y0)
        else:
            print("normal")
            line = plotLineHigh(x0, y0, x1, y1)
    
    return line

def get_bounding_box_pixels(*args: List[CustomPoint2D]) -> List[CustomPoint2D]:
    """
    Given a polygon (list of points), determine the bounding box of said polygon, then return a list of every pixel in said box. 
    """
    l = [(p.x, p.y) for p in args]
    x_max = max([math.ceil(p[0]) for p in l])
    x_min = min([math.floor(p[0]) for p in l])
    y_max = max([math.ceil(p[1]) for p in l])
    y_min = min([math.floor(p[1]) for p in l])
    out = []
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            out.append(CustomPoint2D(x,y))
    return out

def draw_triangle(p0: CustomPoint2D, p1: CustomPoint2D, p2: CustomPoint2D) -> List[CustomPoint2D]:
    # Bounding box optimization
    bounding_box = get_bounding_box_pixels(p0,p1,p2)
    out = []
    for p in bounding_box:
        #compute whether inside triangle
        if inside(p, [p0,p1,p2]):
            # print("{}: is inside!".format(p))
            out.append(p)
    return out

def inside(p: CustomPoint2D, tri: Tuple[CustomPoint2D, CustomPoint2D, CustomPoint2D]):
    p0, p1, p2 = tri
    # Lambda function uses scalar product to determine semiplane in regards to line segment.
    L = lambda p, pi, pj: (pi.y - pj.y)*p.x - (pi.x - pj.x)*p.y + pi.y*(pi.x - pj.x) - (pi.y - pj.y)*pi.x
    for side in [(p0, p1), (p1, p2), (p2, p0)]:
        # print(L(p, *side))
        if L(p, *side) > 0:
            return False # Outside correct semiplane -> outside triangle
    return True

def look_at(camera_pos: CustomPoint3D, axis: np.array, angle: float):
    # Usually camera is at (0,0,0), pointing at z=-1, and up is just ([0,1,0])
    # w = at - camera_pos 
    # w /= np.linalg.norm(at-camera_pos)
    # u = np.cross(w, up)
    # u /= np.linalg.norm(u)
    # v = np.cross(u, w)
    # v /= np.linalg.norm(v)

    # rot = [[u[0] , u[1] , u[2] , 0],
    #        [v[0] , v[1] , v[2] , 0],
    #        [-w[0], -w[1], -w[2], 0],
    #        [0    , 0    , 0    , 1]]

    tran = np.array(
        [
            [1, 0, 0, -camera_pos.x],
            [0, 1, 0, -camera_pos.y],
            [0, 0, 1, -camera_pos.z],
            [0, 0, 0,             1]
        ]
    )

    orientation = quaternion_rotation_matrix(CustomPoint3D(axis[0], axis[1], axis[2]), angle)
    # print(orientation)
    mat = np.matmul(orientation.T, tran)

    return mat

def quaternion_rotation_matrix(axis: CustomPoint3D, angle: float):
    sin = np.sin(angle/2)
    cos = np.cos(angle/2)
    mod = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2) # normalize axis
    axis = CustomPoint3D(axis.x/mod, axis.y/mod, axis.z/mod, 1)
    i,j,k,r =   (
                    axis.x * sin,
                    axis.y * sin,
                    axis.z * sin,
                    cos
                )
    return np.array(
        [
            [1 - 2*(j**2 + k**2), 2*(i*j - k*r)      , 2*(i*k + j*r)      , 0],
            [2*(i*j + k*r)      , 1 - 2*(i**2 + k**2), 2*(j*k + i*r)      , 0],
            [2*(i*k - j*r)      , 2*(j*k + i*r)      , 1 - 2*(i**2 + j**2), 0],
            [0                  , 0                  , 0                  , 1]
        ]
    )
def make_transform(translation: CustomPoint3D  = None, scale: CustomPoint3D  = None, rotation: Tuple[CustomPoint3D, float] = None):
    
    translation = CustomPoint3D(0,0,0) if not translation else CustomPoint3D(*translation)
    scale = CustomPoint3D(1,1,1) if not scale else CustomPoint3D(*scale) 
    rotation = (CustomPoint3D(1,0,0), 0) if not rotation else (rotation[0], rotation[1])


    # Translation matrix using homogeneous coordinates 
    T = np.array(
        [
            [1,0,0,translation.x],
            [0,1,0,translation.y],
            [0,0,1,translation.z],
            [0,0,0,            1],
        ]
    )

    # Scale matrix
    S = np.array(
        [
            [scale.x,0      ,0      ,0],
            [0      ,scale.y,0      ,0],
            [0      ,0      ,scale.z,0],
            [0      ,0      ,0      ,1],
        ]
    )

    # Quaternion rotation
    # print(rotation)
    R = quaternion_rotation_matrix(rotation[0], rotation[1])
    # print(T)
    # print(R)
    # print(S)
    # Return combination
    TR = np.matmul(T, R)
    return np.matmul(TR, S)

def make_projection_matrix(near: float, far: float, fovd: float, w: int, h: int) -> np.array:
    fovy = 2*np.arctan(np.tan(fovd/2)*h/(np.sqrt(w**2+h**2)))
    top = near * np.tan(fovy)
    right = top*(w/h)
    P = np.array(
        [  
            [near/right, 0       , 0                     , 0                       ],
            [0         , near/top, 0                     , 0                       ],
            [0         , 0       , -(near+far)/(far-near), -2*(far*near)/(far-near)],
            [0         , 0       , -1                    , 0                       ]
        ]
    )
    E = np.array(
        [  
            [w/2, 0   , 0, w/2],
            [0  , -h/2, 0, h/2],
            [0  , 0   , 1, 0  ],
            [0  , 0   , 0, 1  ]
        ]
    )
    # print("Persp\n",P)
    # print("Screen\n",E)
    return np.matmul(E, P)

def normalize_2d(projected: np.array) -> np.array:
    out = []
    for p in projected.T:
        x,y,z,w = p
        # print(p)
        out.append(x/w)
        out.append(y/w)
    return out

def prepare_points(points: np.array, model: np.array, view: np.array) -> np.array:
    arr = reshape_points3D(points)
    homo = [1] * len(arr[0])
    arr = list(arr)
    arr.append(homo)
    arr = np.array(arr)  
    t = np.matmul(model, arr)
    p = np.matmul(view, t)
    return normalize_2d(p)

if __name__ == "__main__":

    # Teste Linha
    # points = [
    #     CustomPoint(2.2, 3.1),
    #     CustomPoint(5.5, 10.1),
    # ]
    # print(points)
    # print(draw_line(*points))

    # Teste triangulo    
    # points = [ # 5.1, 5.5, 12.2, 17.4, 16.2, 10.1
    #     CustomPoint2D(5.1, 5.5),
    #     CustomPoint2D(12.2, 17.4),
    #     CustomPoint2D(16.2, 10.1),
    # ]
    
    # print(points)
    # print(draw_triangle(*points))

    # points = [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
    # print(reshape_points3D(points))
    # print(reshape_points3D(list(range(15))))

    # Teste quaternion
    # print(quaternion_rotation_matrix(CustomPoint3D(1,0,0), -np.pi/2))

    # Teste transform
    # print(make_transform(rotation=[CustomPoint3D(1,0,0), -np.pi/2]))

    exit(0)
