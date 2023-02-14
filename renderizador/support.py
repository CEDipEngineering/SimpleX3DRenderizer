
from typing import List
import math

class CustomPoint():
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

def get_emissive_rgb(color):
    # Initially using only emissive colors, later will use rest of color information
    c = color["emissiveColor"]
    return [int(255 * i) for i in c]

def reshape_points(row) -> List[CustomPoint]:
    # Row always has even number of values, each even index corresponds to x, each odd index corresponds to y.
    out = []
    for x, y in zip(row[::2],row[1::2]):
        out.append(CustomPoint(x,y))
    return out

def draw_line(p0: CustomPoint, p1: CustomPoint) -> List[CustomPoint]:
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
            line.append(CustomPoint(x, y))
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
            line.append(CustomPoint(x, y))
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

def get_bounding_box_pixels(*args: List[CustomPoint]) -> List[CustomPoint]:
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
            out.append(CustomPoint(x,y))
    return out

def draw_triangle(p0: CustomPoint, p1: CustomPoint, p2: CustomPoint) -> List[CustomPoint]:
    # Bounding box optimization
    bounding_box = get_bounding_box_pixels(p0,p1,p2)
    out = []
    for p in bounding_box:
        #compute whether inside triangle
        if inside(p, [p0,p1,p2]):
            print("{}: is inside!".format(p))
            out.append(p)
    return out

def inside(p: CustomPoint, tri: CustomPoint):
    p0, p1, p2 = tri
    # Lambda function uses scalar product to determine semiplane in regards to line segment.
    L = lambda p, pi, pj: (pi.y - pj.y)*p.x - (pi.x - pj.x)*p.y + pi.y*(pi.x - pj.x) - (pi.y - pj.y)*pi.x
    for side in [(p0, p1), (p1, p2), (p2, p0)]:
        # print(L(p, *side))
        if L(p, *side) > 0:
            return False # Outside correct semiplane -> outside triangle
    return True

if __name__ == "__main__":

    # Teste Linha
    # points = [
    #     CustomPoint(2.2, 3.1),
    #     CustomPoint(5.5, 10.1),
    # ]
    # print(points)
    # print(draw_line(*points))

    # Teste triangulo    
    points = [ # 5.1, 5.5, 12.2, 17.4, 16.2, 10.1
        CustomPoint(5.1, 5.5),
        CustomPoint(12.2, 17.4),
        CustomPoint(16.2, 10.1),
    ]
    
    print(points)
    print(draw_triangle(*points))
    
    exit(0)
