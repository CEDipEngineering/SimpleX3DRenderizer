#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Carlos Dip
Disciplina: Computação Gráfica
Data: Feb 13, 2023
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from collections import deque
from itertools import zip_longest

from support import * # Implementacoes individuais
from TransformStack import TransformStack # Pilha de transforms
from Mipmap import Mipmap # Implementacao de mipmap
from icosphere import icosphere # Ver icosphere.py
from Light import Light, DirectionalLight # Implementacao das fontes de luz

DEBUG_COLORS = False
ENABLE_TRANSPARENCY = True

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    transform_stack = TransformStack()
    projection = transform_stack.peek()
    lights : List[Light] = []

    ## lambdas
    outside = lambda n: n<0.0 or n>1.0 # Triangle interior check
    in_screen = lambda pt2d: (pt2d.x>0 and pt2d.x<(GL.width*GL.SSAA) and pt2d.y>0 and pt2d.y<(GL.height*GL.SSAA)) # Frustum Culling

    @staticmethod
    def setup(width, height, near=0.01, far=1000, SSAA=1, renderer=None):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.SSAA = SSAA
        GL.renderer = renderer

    @staticmethod
    def tri_bounding_box(*args: List[CustomPoint3D]) -> List[CustomPoint3D]:
        """
        Given a triangle (list of 3 points), determine the bounding box of said triangle, 
        then return a matrix of pixels in said box, already interpolated using baricentric coordinates. 
        """
        x_max = max([math.ceil(p[0])  for p in args])+1
        x_min = min([math.floor(p[0]) for p in args])-1
        y_max = max([math.ceil(p[1])  for p in args])+1
        y_min = min([math.floor(p[1]) for p in args])-1
        out = []
        p0, p1, p2 = args
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                
                p = CustomPoint3D(x,y,0,1)
                # Baricentric coordinates (dot product)
                bari = construct_baricentric_coordinates(p, [p0,p1,p2])
                p.alpha, p.beta, p.gamma = bari
                p.z = 1/(p.alpha/p0.z + p.beta/p1.z + p.gamma/p2.z) # Interpolate z coordinate with harmonic weighted mean

                # Store point
                out.append(p)

        return np.reshape(np.array(out), ((y_max - y_min)+1 , (x_max - x_min)+1))

    @staticmethod
    def draw_triangles(tris: List[List[CustomPoint3D]], colors: dict, colorPerVertex: bool = False, current_texture = None, lighting=None):
        """
        Generalized method for drawing triangles.

        Args:
        - tris must be a list of lists, where each inner list has shape 3, such as:
            [[P3D: (10,10,0), P3D: (0,0,0)  , P3D: (10, 0, 0)],
             [P3D: (25,10,0), P3D: (13,12,0), P3D: (10, 0, 0)]]
        
        - colors must be a dict, as supplied by X3D specs.
        - colorPerVertex indicates whether to interpolate colors from each vertex. Points must have attribute c specified.
        - currentTexture indicates the texture being mapped to the surface. Points must have attribute t specified.
        - lighting must be either 'face' or 'interpolate' to indicate whether surface normals used for shading will be considered per face or vertex.
        """
        if current_texture:
            mipmap = Mipmap(gpu.GPU.load_texture(current_texture[0]), maxLevel=3)

        emissive_color  = np.float64(colors["emissiveColor"])
        specular_color  = np.float64(colors["specularColor"])
        diffuse_color   = np.float64(colors["diffuseColor"])
        shininess       = np.float64(colors["shininess"])


        # if lighting:
        #     print("\n".join(map(str, GL.lights)))

        for tri in tris:
            # Unpack vertices
            p0, p1, p2 = tri

            # Random face colors, for debugging
            if DEBUG_COLORS: r,g,b = np.random.randint(0,256, size=(3))
            
            # Bounding box optimization
            bounding_box = GL.tri_bounding_box(p0,p1,p2)

            # Loop over all points in triangle bounding box
            w, h = bounding_box.shape
            for j in range(h):
                for i in range(w):

                    p = bounding_box[i, j]
                    if not GL.in_screen(p): continue # p out of screen
                    if p.outside(): continue # p outside triangle

                    # Z-Buffer
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["DEPTH"])
                    if gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F) < p.z: continue # This point is behind something else we've already drawn
                    gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F, [p.z]) # Closest point in this camera position

                    # Bind buffer
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])
                    if GL.SSAA != 1: gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"]) # SSAA

                    # Colors
                    if not DEBUG_COLORS: 
                        if colorPerVertex: # Solid colors, just use emissive
                            r, g, b = (p.alpha*p0.c*(1/p0.z) + p.beta*p1.c*(1/p1.z) + p.gamma*p2.c*(1/p2.z))*p.z # Compute weighted RGB by z
                            # Convert to 0.255 int
                            r = int(r * 255)
                            g = int(g * 255)
                            b = int(b * 255)
                        elif current_texture:
                            # Interpolate u, v
                            p.t = (p.alpha*p0.t*(1/p0.z) + p.beta*p1.t*(1/p1.z) + p.gamma*p2.t*(1/p2.z))*p.z
                            
                            # print(f"{i=};{j=};{bounding_box.shape=}")

                            if i+1 >= w:
                                L = 0
                            else:
                                p_right = bounding_box[i + 1, j]
                                p_right.t = (p_right.alpha*p0.t*(1/p0.z) + p_right.beta*p1.t*(1/p1.z) + p_right.gamma*p2.t*(1/p2.z))*p.z
                                if j+1 >= h:
                                    L = 0
                                else:
                                    p_up    = bounding_box[i, j + 1]
                                    p_up.t = (p_up.alpha*p0.t*(1/p0.z) + p_up.beta*p1.t*(1/p1.z) + p_up.gamma*p2.t*(1/p2.z))*p.z
                                    L = mipmap.calculate_L(p, p_right, p_up)
                            
                            r, g, b, a = mipmap.get_texture(*p.t, L)
                        elif lighting:
                            world_x = 1/(p.alpha/p0.x + p.beta/p1.x + p.gamma/p2.x) # Interpolate x coordinate with harmonic weighted mean
                            world_y = 1/(p.alpha/p0.y + p.beta/p1.y + p.gamma/p2.y) # Interpolate y coordinate with harmonic weighted mean
                            look_direction = GL.camera_position - np.array([world_x, world_y, p.z])
                            look_direction = look_direction * 1./np.linalg.norm(look_direction)                         
                            
                            if lighting.lower() == "face":
                                p.n = p0.n
                            elif lighting.lower() == "interpolate":
                                # bari = construct_baricentric_coordinates(p, [p0.n, p1.n, p2.n]) 
                                p.n = p.alpha*p0.n/p0.z + p.beta*p1.n/p1.z + p.gamma*p2.n/p2.z
                                p.n = p.n * 1.0/np.linalg.norm(p.n)
                            else:
                                print("WARNING: lighting was specified, however {} is not a valid type, use 'Face' or 'Interpolate'".format(lighting))

                            light_color_effect = [0., 0., 0.] # Black RGB
                            for light in GL.lights:
                                # Extract light parameters
                                ambient_intensity = light.ambientIntensity
                                intensity = light.intensity
                                light_color = light.color
                                direction = light.direction

                                # Calculate Look direction vector combined with normal
                                Lv = (look_direction+direction)
                                Lv = Lv * 1./np.linalg.norm(Lv)
                                Lv = np.sign(Lv) * np.abs(Lv) ** (shininess*128.0)

                                ambient_factor = diffuse_color*ambient_intensity
                                diffuse_factor = diffuse_color*max(0, np.dot(p.n, direction)+0.1)*intensity
                                specular_factor = specular_color*max(0, np.dot(p.n, Lv)+0.1)*intensity
                                total_factor = (ambient_factor + diffuse_factor + specular_factor)
                                light_color_effect += light_color * total_factor
                            # print(light_color_effect)
                            r, g, b = np.uint8(255.0 * (light_color_effect + emissive_color))

                        else: # Draw color per vertex means we must interpolate with baricentric coordinates
                            r, g, b = get_emissive_rgb(colors)
                
                    # Transparency
                    if ENABLE_TRANSPARENCY:
                        prev = gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.RGB8)
                        alpha = colors["transparency"]
                        prev = np.array(prev,dtype=np.float64)*alpha
                        curr = [r,g,b]
                        curr = np.array(curr,dtype=np.float64)*(1-alpha)
                        r,g,b = curr+prev

                    # Draw
                    gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.RGB8, (r,g,b))            

        # Rebind buffer
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores
        # Exemplo:
        # gpu.GPU.draw_pixel(3, 1, 255, 0, 0) # altera um pixel da imagem (u, v, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        respoints = reshape_points2D(point)
        # print(respoints)
        for p in respoints:
            gpu.GPU.draw_pixel(p.get_pixel(), gpu.GPU.RGB8, get_emissive_rgb(colors))

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo poviewpointnto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel(pos_x, pos_y, 255, 0, 0) # altera um pixel da imagem (u, v, r, g, b)
        respoints = reshape_points2D(lineSegments)
        p0 = respoints.pop(0)
        p1 = respoints.pop(0)
        line = draw_line(p0, p1)
        for p in line:
            gpu.GPU.draw_pixel(p.get_pixel(), gpu.GPU.RGB8, get_emissive_rgb(colors))

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        # print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        # print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores

        respoints = deque(reshape_points2D(vertices))
        if GL.SSAA != 1: gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"]) # SSAA
        while len(respoints) != 0:
            a, b, c = respoints.popleft(), respoints.popleft(), respoints.popleft()
            if not GL.in_screen(a): continue
            if not GL.in_screen(b): continue
            if not GL.in_screen(c): continue
            # print(a,b,c)
            tri = draw_triangle(a,b,c)
            for p in tri:
                gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.RGB8, get_emissive_rgb(colors))
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores

        norm_3d = prepare_points_3d(point, GL.transform_stack.peek(), GL.projection)        
        tris = np.reshape(np.array(norm_3d), (-1, 3))

        for tri in tris:
            p0, p1, p2 = tri
            normal = np.cross((p1-p0).normalize().to_array(), (p2-p0).normalize().to_array())
            # normal = normal * 1./np.linalg.norm(normal)
            p0.n = normal
            p1.n = normal
            p2.n = normal

        GL.draw_triangles(tris, colors, lighting="face")
        
    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Viewpoint : ", end='')
        # print("position = {0} ".format(position), end='')
        # print("orientation = {0} ".format(orientation), end='')
        # print("fieldOfView = {0} ".format(fieldOfView))
        GL.camera_position = np.array([position[0], position[1], position[2]])
        camera = look_at(CustomPoint3D(*GL.camera_position), CustomPoint3D(*orientation[:3]), orientation[-1])
        # print("lookat:\n",camera)
        project = make_projection_matrix(near=GL.near, far=GL.far, fovd=fieldOfView, w=GL.width*GL.SSAA, h=GL.height*GL.SSAA)
        # print(project)
        GL.projection = np.matmul(project, camera)
        # print(GL.projection)

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Transform : ", end='')
        # if translation:
        #     print("translation = {0} ".format(translation), end='') # imprime no terminal
        # if scale:
        #     print("scale = {0} ".format(scale), end='') # imprime no terminal
        # if rotation:
        #     print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        # print("")

        GL.transform_stack.push(make_transform(translation, scale, (CustomPoint3D(*rotation[:3]), rotation[-1])))
        # print(GL.transform_stack.peek()) 

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")
        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleStripSet : pontos = {0} ".format(point), end='')
        # for i, strip in enumerate(stripCount):
        #     print("strip[{0}] = {1} ".format(i, strip), end='')
        # print("")
        # print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores
        
        norm_3d = prepare_points_3d(point, GL.transform_stack.peek(), GL.projection)      
        
        # Identify all triangles
        tris = []
        for count in stripCount:
            for i in range(count-2):
                if i % 2 == 0:
                    p0,p1,p2 = norm_3d[i], norm_3d[i+1], norm_3d[i+2]
                else:
                    p0,p1,p2 = norm_3d[i], norm_3d[i+2], norm_3d[i+1]
                tris.append([p0, p1, p2])
        
        # Draw
        GL.draw_triangles(tris, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        # print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        norm_3d = prepare_points_3d(point, GL.transform_stack.peek(), GL.projection)      
        
        # Identify all triangles
        tris = []
        for i in range(len(index)-2):
            if i % 2 == 0:
                p0,p1,p2 = norm_3d[index[i]], norm_3d[index[i+1]], norm_3d[index[i+2]]
            else:
                p0,p1,p2 = norm_3d[index[i]], norm_3d[index[i+2]], norm_3d[index[i+1]]
            tris.append([p0, p1, p2])
        
        # Draw
        GL.draw_triangles(tris, colors)
        
    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""

        # Apply scale matrix
        scale = [
            [size[0]/2, 0        , 0        ],
            [0        , size[1]/2, 0        ],
            [0        , 0        , size[2]/2]
        ]

        # Vertices are hardcoded, since box is always centered at origin
        vertices = np.array([
            [-1, -1, -1], # 0
            [ 1, -1, -1], # 1
            [-1, -1,  1], # 2
            [ 1, -1,  1], # 3
            [-1,  1, -1], # 4
            [ 1,  1, -1], # 5
            [-1,  1,  1], # 6
            [ 1,  1,  1], # 7
        ])

        # Indexes for each of the twelve triangles
        indexes = np.array([
            [ 0, 3, 1], [ 0, 2, 3], # Bottom Face
            [ 4, 5, 7], [ 4, 7, 6], # Top Face
            [ 1, 3, 7], [ 1, 7, 5], # +x Face
            [ 0, 4, 6], [ 0, 6, 2], # -x Face
            [ 2, 6, 7], [ 2, 7, 3], # -z Face
            [ 0, 1, 5], [ 0, 5, 4], # +z Face
        ])

        # Pre computed normals
        normals = np.array([    
            [  0, -1,  0,  1], [  0, -1,  0,  1], # Bottom Face
            [  0,  1,  0,  1], [  0,  1,  0,  1], # Top Face
            [  1,  0,  0,  1], [  1,  0,  0,  1], # +x Face
            [ -1,  0,  0,  1], [ -1,  0,  0,  1], # -x Face
            [  0,  0,  1,  1], [  0,  0,  1,  1], # +z Face
            [  0,  0, -1,  1], [  0,  0, -1,  1], # -z Face
        ])*-1
        # Transform normals to match final cube
        normals = np.matmul(GL.projection, np.matmul(GL.transform_stack.peek(), normals.T)).T
        x = normals[:,0] * np.reciprocal(normals[:,-1])
        y = normals[:,1] * np.reciprocal(normals[:,-1])
        # z =  * np.reciprocal(normals[:,-1])
        normals = np.vstack((x, y, normals[:,2])).T
        norms = np.linalg.norm(normals, keepdims=True, axis=1)
        normals = normals / norms[:,]

        # Identify all triangles
        vertices = np.matmul(scale, vertices.T).T
        points = np.array([vertices[i] for i in indexes])
        points = np.array(points).flatten()
        points = np.array(prepare_points_3d(points, GL.transform_stack.peek(), GL.projection))
        tris = np.reshape(points, (-1, 3))

        for tri, normal in zip(tris, normals):
            for p in tri:
                p.n = normal

        # Draw
        GL.draw_triangles(tris, colors, lighting='face')

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.
        if texCoord and texCoordIndex and current_texture:
            texCoord = np.reshape(texCoord, (-1, 2))

        if color is None: colorPerVertex = False # X3D is pretty stupid

        norm_3d = prepare_points_3d(coord, GL.transform_stack.peek(), GL.projection)

        if colorPerVertex:
            color = np.reshape(color, (-1, 3))
        curr_tri = deque(maxlen=3)
        
        tris = []
        for e in zip_longest(coordIndex, colorIndex, texCoordIndex):
            if e[0] == -1:
                curr_tri = deque(maxlen=3)
                continue
            curr_tri.append(e)
            if len(curr_tri) == 3: # Three points, draw triangle
                
                # Gather triangle vertices
                a, b, c = curr_tri
                p0, p1, p2 = norm_3d[a[0]], norm_3d[b[0]], norm_3d[c[0]]
                
                # Update point color
                if colorPerVertex: p0.c, p1.c, p2.c = color[a[1]], color[b[1]], color[c[1]]
                
                # Update point texture
                if current_texture: p0.t, p1.t, p2.t = texCoord[a[2]], texCoord[b[2]], texCoord[c[2]]
                    
                # Update normals
                normal = np.cross((p1-p0).normalize().to_array(), (p2-p0).normalize().to_array())
                p0.n = normal
                p1.n = normal
                p2.n = normal

                tris.append([p0, p1, p2])
        GL.draw_triangles(tris, colors, colorPerVertex, current_texture, lighting="face")

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        # print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

        vertices, indexes = icosphere(3)
        vertices = np.array(vertices)

        vertices = np.array(prepare_points_3d(vertices.flatten()*radius, GL.transform_stack.peek(), GL.projection))

        center = np.matmul(GL.projection, np.matmul(GL.transform_stack.peek(), np.array([[0., 0., 0., 1]]).T)).T
        x = center[:,0] * np.reciprocal(center[:,-1])
        y = center[:,1] * np.reciprocal(center[:,-1])
        center = np.vstack((x, y, center[:,2])).T
        norms = np.linalg.norm(center, keepdims=True, axis=1)
        center = center / norms[:,]
        center = CustomPoint3D(*center[0])

        tris = []
        for tri in indexes:
            # Get vertex coordinates
            p0, p1, p2 = vertices[tri]
            p0.n = (p0 - center).normalize().to_array()
            p1.n = (p1 - center).normalize().to_array()
            p2.n = (p2 - center).normalize().to_array()
            # print("Tri:")
            # print(np.dot(p0.n, GL.lights[0].direction))
            # print(np.dot(p1.n, GL.lights[0].direction))
            # print(np.dot(p2.n, GL.lights[0].direction))
            # print("="*25)
            tris.append([p0, p1, p2])
        
        GL.draw_triangles(tris, colors, lighting="interpolate")

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal
        if headlight:
            # print("Headlight enabled!")
            GL.lights.append(
                DirectionalLight(
                    ambientIntensity=0.,
                    color=[1., 1., 1.],
                    intensity=1.,
                    direction=[0., 0., -1]
                )
            )
        # else:
        #     print("Headlight disabled!")

    @staticmethod
    def clear_lights():
        GL.lights = []

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        # print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        # print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal
        light = DirectionalLight(
            ambientIntensity,
            color,
            intensity,
            direction
        )

        GL.lights.append(light)

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        # print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # Cada ponto corresponde a um indice de key
        t = set_fraction / (key[1] - key[0]) % 1
        # Reshape points
        points = np.reshape(np.array(keyValue), (-1, 3))
        # Infer 4 indexes
        p1 = 0
        for i, e in enumerate(key):
            if set_fraction < e:
                p1 = i-1
                break
        p2 = (p1 + 1)%len(key)
        p0 = (p1 - 1)%len(key)
        p3 = (p1 + 2)%len(key)

        # Collect coordinates
        P = points[[p0, p1, p2, p3]]

        Catmull_Rom = np.array(
            [
                [-0.5,  3./2., -3./2.,  0.5],
                [   1, -5./2.,     2., -0.5],
                [-0.5,     0.,    0.5,    0],
                [   0,      1,      0,    0],
            ]
        )

        t2 = t*t
        t3 = t2*t

        T = np.array([t3, t2, t, 1.0])

        # Transform
        return np.matmul(T, np.matmul(Catmull_Rom, P))

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.
        # Cada ponto corresponde a um indice de key
        t = set_fraction / (key[1] - key[0]) % 1
        # Reshape points
        points = np.reshape(np.array(keyValue), (-1, 4))
        # Infer 4 indexes
        p1 = 0
        for i, e in enumerate(key):
            if set_fraction < e:
                p1 = i-1
                break
        p2 = (p1 + 1)%len(key)
        p0 = (p1 - 1)%len(key)
        p3 = (p1 + 2)%len(key)

        # Collect coordinates
        P = points[[p0, p1, p2, p3]]

        Catmull_Rom = np.array(
            [
                [-0.5,  3./2., -3./2.,  0.5],
                [   1, -5./2.,     2., -0.5],
                [-0.5,     0.,    0.5,    0],
                [   0,      1,      0,    0],
            ]
        )

        t2 = t*t
        t3 = t2*t

        T = np.array([t3, t2, t, 1.0])

        # Transform
        return np.matmul(T, np.matmul(Catmull_Rom, P))

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
