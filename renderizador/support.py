
from typing import List, Tuple
import numpy as np
import math

PRINT_TRANSFORMS = False

class CustomPoint2D():
    def __init__(self, x, y, color=(1,1,1)) -> None:
        self.x = x
        self.y = y
        self.r, self.g, self.b = color

    def get_pixel(self):
        return (int(self.x), int(self.y))

    def get_colot(self):
        return (self.r, self.g, self.b)

    def __sub__(self, other):
        if type(other) is CustomPoint2D:
            return CustomPoint2D(self.x-other.x, self.y-other.y)
        raise TypeError("CustomPoint2D only supports operations with other CustomPoint2D obejcts.")
    
    def __add__(self, other):
        if type(other) is CustomPoint2D:
            return CustomPoint2D(self.x+other.x, self.y+other.y)
        raise TypeError("CustomPoint2D only supports operations with other CustomPoint2D obejcts.")
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        raise IndexError("Invalid index passed, must be 0,1 to access x,y coordinates.")
    
    def __setitem__(self, i, v):
        if i == 0:
            self.x = v
        if i == 1:
            self.y = v
        raise IndexError("Invalid index passed, must be 0,1 to access x,y coordinates.")
  
    # Methods for debug printing
    def __str__(self) -> str:
        return "({:.1f},{:.1f})".format(self.x, self.y)
    
    def __repr__(self) -> str:
        return self.__str__()
    
class CustomPoint3D():
    def __init__(self, x, y, z, w=1, alpha=0.3, beta=0.3, gamma=0.3, c=np.array([0.,0.,0.]), t=np.array([0.,0.]), n = np.array([0., 0., 1.])) -> None:
        # Homogeneous Coordinates
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # Baricentric coordinates
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Color (array, 0:1, float, rgb)
        self.c = c

        # Texture mapping (array, 0:1, float, uv)
        self.t = t

        # Normal
        self.n = n

    def __sub__(self, other):
        if type(other) is CustomPoint3D:
            return CustomPoint3D(self.x-other.x, self.y-other.y, self.z-other.z)
        raise TypeError("CustomPoint3D only supports operations with other CustomPoint3D obejcts.")
    
    def __rsub__(self, other):
        if type(other) is CustomPoint3D:
            return CustomPoint3D(other.x-self.x, other.y-self.y, other.z-self.z)
        raise TypeError("CustomPoint3D only supports operations with other CustomPoint3D obejcts.")
    
    def __add__(self, other):
        if type(other) is CustomPoint3D:
            return CustomPoint3D(self.x+other.x, self.y+other.y, self.z+other.z)
        raise TypeError("CustomPoint3D only supports operations with other CustomPoint3D obejcts.")
    
    def __mul__(self, other):
        if type(other) not in [int, float, np.float64]:
            raise TypeError("Multiplying points is only available for scalars (int/float). Provided {}".format(type(other)))
        return CustomPoint3D(
            self.x*other,
            self.y*other,
            self.z*other,
            self.w
        )
    
    def __rmul__(self, other):
        if type(other) not in [int, float, np.float64]:
            raise TypeError("Multiplying points is only available for scalars (int/float). Provided {}".format(type(other)))
        return CustomPoint3D(
            self.x*other,
            self.y*other,
            self.z*other,
            self.w
        )

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
    
    def get_flat_pixel(self):
        return (int(self.x), int(self.y))
    
    def homogeneous_division(self):
        "Transforms inplace, but also returns self, just in case its used mid-operation"
        self.x /= self.w
        self.y /= self.w
        # self.z /= self.w
        self.w /= self.w
        return self

    def outside(self):
        outside = lambda n: n<0.0 or n>1.0 # Triangle interior check
        return outside(self.alpha) or outside(self.beta) or outside(self.gamma)

    def normalize(self):
        norm = np.linalg.norm([self.x, self.y, self.z])
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    # Methods for debug printing
    def __str__(self) -> str:
        return "P3D:({},{},{},{})".format(self.x, self.y, self.z, self.w)
    
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
        # print(x0,y0)
        # print(x1,y1)
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
        # print("Low")
        if x0 < x1:
            # print("reverse")
            line = plotLineLow(x1, y1, x0, y0)
        else:
            # print("normal")
            line = plotLineLow(x0, y0, x1, y1)
    else:
        # print("High")
        if y0 < y1:
            # print("reverse")
            line = plotLineHigh(x1, y1, x0, y0)
        else:
            # print("normal")
            line = plotLineHigh(x0, y0, x1, y1)
    
    return line

def get_bounding_box_pixels(*args: List[CustomPoint2D]) -> List[CustomPoint3D]:
    """
    DEPRACATED!! USE GL.tri_bounding_box INSTEAD!!
    Given a polygon (list of points), determine the bounding box of said polygon, then return a list of every pixel in said box. 
    """
    print("Warning! Using deprecated function, use GL.tri_bounding_box instead.")
    l = [(p.x, p.y) for p in args]
    x_max = max([math.ceil(p[0]) for p in l])
    x_min = min([math.floor(p[0]) for p in l])
    y_max = max([math.ceil(p[1]) for p in l])
    y_min = min([math.floor(p[1]) for p in l])
    out = []
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            out.append(CustomPoint3D(x,y,0,1))
    return out

def draw_triangle(p0: CustomPoint2D, p1: CustomPoint2D, p2: CustomPoint2D) -> List[CustomPoint2D]:
    # Bounding box optimization
    bounding_box = get_bounding_box_pixels(p0,p1,p2)
    out = []
    outside = lambda n: n<0.0 or n>1.0
    for p in bounding_box:
        #compute whether inside triangle
        # Baricentric
        bari = construct_baricentric_coordinates(p, [p0,p1,p2])
        # 0 to 1 means inside
        if outside(bari[0]): continue
        if outside(bari[1]): continue
        if outside(bari[2]): continue
        # print("{}: is inside!".format(p))
        out.append(p)
    return out

def construct_baricentric_coordinates(p: CustomPoint2D, tri: List[CustomPoint2D]):
    a,b,c = tri
    
    alpha_num = (-(p[0]-b[0])*(c[1]-b[1]) + (p[1] - b[1]) * (c[0] - b[0]))
    alpha_den = (-(a[0] - b[0])*(c[1] - b[1]) + (a[1] - b[1])*(c[0] - b[0]))
    if alpha_den == 0: alpha_den = 1e-10
    alpha = alpha_num/alpha_den
    
    beta_num = (-(p[0]-c[0])*(a[1]-c[1]) + (p[1] - c[1]) * (a[0] - c[0]))
    beta_den = (-(b[0] - c[0])*(a[1] - c[1]) + (b[1] - c[1])*(a[0] - c[0]))
    if beta_den == 0: beta_den = 1e-10
    beta = beta_num/beta_den
    
    gamma = 1 - alpha - beta
    return np.array([alpha, beta, gamma])

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
    
    translation =  CustomPoint3D(0,0,0)     if translation  is None else CustomPoint3D(*translation)
    scale       =  CustomPoint3D(1,1,1)     if scale        is None else CustomPoint3D(*scale) 
    rotation    = (CustomPoint3D(1,0,0), 0) if rotation     is None else (rotation[0], rotation[1])

    '''
    translation = CustomPoint3D(0,0,0) if not translation else CustomPoint3D(*translation)
    scale = CustomPoint3D(1,1,1) if not scale else CustomPoint3D(*scale) 
    rotation = (CustomPoint3D(1,0,0), 0) if not rotation else (rotation[0], rotation[1])
    '''


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
    TRS = np.matmul(TR, S)
    return TRS

def make_projection_matrix(near: float, far: float, fovd: float, w: int, h: int) -> np.array:
    fovy = 2.0*np.arctan(np.tan(fovd/2.0)*h/(np.sqrt(w**2.0+h**2.0)))
    top = near * np.tan(fovy)
    right = top*(w/h)
    P = np.array(
        [  
            [near/right, 0.0       , 0.0                     , 0.0                       ],
            [0.0         , near/top, 0.0                     , 0.0                       ],
            [0.0         , 0.0       , -(near+far)/(far-near), -2*(far*near)/(far-near)],
            [0.0         , 0.0       , -1                    , 0.0                       ]
        ]
    )
    E = np.array(
        [  
            [w/2.0, 0.0   , 0.0, w/2.0],
            [0.0  , -h/2.0, 0.0, h/2.0],
            [0.0  , 0.0   , 0.5, 0.5  ],
            [0.0  , 0.0   , 0.0, 1.0  ]
        ]
    )
    if PRINT_TRANSFORMS:
        print("Perspective:\n{}".format(P))
        print("Screen:\n{}".format(E))
    return np.matmul(E, P)
    # return P

def normalize_2d(projected: np.array) -> np.array:
    out = []
    for p in projected.T:
        x,y,z,w = p
        # print(p)
        out.append(x/w)
        out.append(y/w)
    return out

def normalize_3d(projected: np.array) -> np.array:
    out = []
    for p in projected.T:
        out.append(CustomPoint3D(*p).homogeneous_division())
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

def prepare_points_3d(points: np.array, model: np.array, view: np.array) -> np.array:
    if PRINT_TRANSFORMS:
        print("Model:\n{}".format(model))
        print("View:\n{}".format(view))
    arr = reshape_points3D(points)
    homo = [1] * len(arr[0])
    arr = list(arr)
    arr.append(homo)
    arr = np.array(arr)  
    t = np.matmul(model, arr)
    # print("Transformed: \n", t)
    # if PRINT_TRANSFORMS: print("Points NDC:\n{}".format(t))
    p = np.matmul(view, t)
    # print("Projected: \n", p)
    return normalize_3d(p)

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
