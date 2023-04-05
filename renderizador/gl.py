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

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    transform_stack = TransformStack()
    projection = transform_stack.peek()

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
        in_screen = lambda pt2d: (pt2d.x>0 and pt2d.x<(GL.width*GL.SSAA) and pt2d.y>0 and pt2d.y<(GL.height*GL.SSAA)) # Frustum Culling
        if GL.SSAA != 1: gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"]) # SSAA
        while len(respoints) != 0:
            a, b, c = respoints.popleft(), respoints.popleft(), respoints.popleft()
            if not in_screen(a): continue
            if not in_screen(b): continue
            if not in_screen(c): continue
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
        outside = lambda n: n<0.0 or n>1.0 # Simple macro
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["DEPTH"])
        for p0, p1, p2 in tris:
            # Bounding box optimization
            bounding_box = get_bounding_box_pixels(p0,p1,p2)
            drawList = []
            for p in bounding_box:
                
                # Baricentric coordinates (dot product)
                bari = construct_baricentric_coordinates(p, [p0,p1,p2])
                # 0 to 1 means inside
                if outside(bari[0]): continue
                if outside(bari[1]): continue
                if outside(bari[2]): continue
                alpha, beta, gamma = bari

                # Z-Buffer
                # print("bbox p: {}".format(p))
                if gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F) < p.z: continue # ja desenhamos coisas na frente
                gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F, [p.z]) # novo ponto no buffer

                # Append to list of drawable pixels.
                drawList.append(
                    CustomPoint3D(
                        p.x,
                        p.y,
                        p.z,
                        p.w,
                        alpha,
                        beta,
                        gamma
                    )
                )

        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])
        if GL.SSAA != 1: gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"]) # SSAA
        for p in drawList:
            # print(p)
            prev = gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.RGB8)
            alpha = colors["transparency"]
            prev = np.array(prev,dtype=np.float64)*alpha
            curr = get_emissive_rgb(colors)
            curr = np.array(curr,dtype=np.float64)*(1-alpha)
            r,g,b = curr+prev
            gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.RGB8, (r,g,b)) # novo ponto no buffer
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])

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
        camera = look_at(CustomPoint3D(position[0], position[1], position[2]), CustomPoint3D(*orientation[:3]), orientation[-1])
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
        
        norm_2d = prepare_points(point, GL.transform_stack.peek(), GL.projection)        
        respoints = reshape_points2D(norm_2d)
        for count in stripCount:
            for i in range(count-2):
                if i % 2 == 0:
                    a,b,c = respoints[i], respoints[i+1], respoints[i+2]
                else:
                    a,b,c = respoints[i], respoints[i+2], respoints[i+1]
                tri = draw_triangle(a,b,c)
                for p in tri:
                    gpu.GPU.draw_pixel(p.get_pixel(), gpu.GPU.RGB8, get_emissive_rgb(colors))

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

        norm_2d = prepare_points(point, GL.transform_stack.peek(), GL.projection)        
        respoints = reshape_points2D(norm_2d)
        if GL.SSAA != 1:
            gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"])
        for i in range(len(index)-2):
            if i % 2 == 0:
                a,b,c = respoints[index[i]], respoints[index[i+1]], respoints[index[i+2]]
            else:
                a,b,c = respoints[index[i]], respoints[index[i+2]], respoints[index[i+1]]
            tri = draw_triangle(a,b,c)
            for p in tri:
                gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.RGB8, get_emissive_rgb(colors))
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])
        
    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Box : size = {0}".format(size)) # imprime no terminal pontos
        # print("Box : colors = {0}".format(colors)) # imprime no terminal as cores
        SCALE = [
            [size[0]/2, 0        , 0        ],
            [0        , size[1]/2, 0        ],
            [0        , 0        , size[2]/2]
        ]
        VERTEX = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [-1,  1,  1]
        ])

        VERTEX = np.matmul(SCALE, VERTEX.T).T

        INDEXES = [
            1,0,2,2,0,3,
            1,2,6,6,5,1,
            2,3,7,7,6,2,
            3,0,4,4,7,3,
            0,1,5,5,4,0,
            4,5,6,6,7,4
        ]

        # print(VERTEX)
        points = [VERTEX[i] for i in INDEXES]    
        points = np.array(points).flatten()
        GL.triangleSet(points, colors)

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

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedFaceSet : ")
        # if coord:
        #     print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        # print("colorPerVertex = {0}".format(colorPerVertex))
        # if colorPerVertex and color and colorIndex:
        #     print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        # if texCoord and texCoordIndex:
        #     print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        # if current_texture:
        #     image = gpu.GPU.load_texture(current_texture[0])
        #     print("\t Matriz com image = {0}".format(image))
        #     print("\t Dimensões da image = {0}".format(image.shape))
        # print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores
        
        if texCoord and texCoordIndex and current_texture:
            mipmap = Mipmap(gpu.GPU.load_texture(current_texture[0]))
            return

        if color is None: colorPerVertex = False # X3D is pretty stupid

        norm_3d = prepare_points_3d(coord, GL.transform_stack.peek(), GL.projection)
        # print(norm_3d)
        # gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["DEPTH"])
        # print(gpu.GPU.get_frame_buffer())
        outside = lambda n: n<0.0 or n>1.0 # Simple macro
        if colorPerVertex:
            color = np.reshape(color, (-1, 3))
        curr_tri = deque(maxlen=3)
        for i, e in enumerate(zip_longest(coordIndex, colorIndex)):
            if e[0] == -1:
                curr_tri = deque(maxlen=3)
                continue
            curr_tri.append(e)
            if len(curr_tri) == 3:
                a, b, c = curr_tri
                p0, p1, p2 = norm_3d[a[0]], norm_3d[b[0]], norm_3d[c[0]]
                if colorPerVertex: c0, c1, c2 = color[a[1]], color[b[1]], color[c[1]]

                # Bounding box optimization
                bounding_box = get_bounding_box_pixels(p0,p1,p2)
                in_screen = lambda pt2d: (pt2d.x>0 and pt2d.x<(GL.width*GL.SSAA) and pt2d.y>0 and pt2d.y<(GL.height*GL.SSAA)) # Frustum Culling

                for p in bounding_box:

                    if not in_screen(p): continue # p out of screen
                    
                    # Baricentric coordinates (dot product)
                    bari = construct_baricentric_coordinates(p, [p0,p1,p2])
                    # 0 to 1 means inside
                    if outside(bari[0]): continue
                    if outside(bari[1]): continue
                    if outside(bari[2]): continue
                    alpha, beta, gamma = bari
                    p.z = 1/(alpha/p0.z + beta/p1.z + gamma/p2.z) # Interpolate z coordinate with harmonic weighted mean

                    # Z-Buffer
                    # print("bbox p: {}".format(p))
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["DEPTH"])
                    # if p.z != 0: print(p.z)
                    if gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F) < p.z: continue # ja desenhamos coisas na frente
                    gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.DEPTH_COMPONENT32F, [p.z]) # novo ponto no buffer

                    # Bind draw buffer
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])
                    if GL.SSAA != 1: gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["HIGHRES"]) # SSAA

                    # Drawing
                    if not colorPerVertex: # Solid colors, just use emissive
                        r, g, b = get_emissive_rgb(colors)
                    else: # Draw color per vertex means we must interpolate with baricentric coordinates
                        r, g, b = (alpha*c0*(1/p0.z) + beta*c1*(1/p1.z) + gamma*c2*(1/p2.z))*p.z # Compute weighted RGB by z
                        # if p.get_flat_pixel()[1]<20:
                        #     print("Ponto: ", p)
                        #     print("Cores originais: ", c0, c1, c2)
                        #     print("Cores interpoladas: ", r,g,b)
                        # Convert to 0.255 int
                        r = int(r * 255)
                        g = int(g * 255)
                        b = int(b * 255)
                        # print(r,g,b)
                    # After getting r,g,b compute transparency
                    # prev = gpu.GPU.read_pixel(p.get_flat_pixel(), gpu.GPU.RGB8)
                    # alpha = colors["transparency"]
                    # prev = np.array(prev,dtype=np.float64)*alpha
                    # curr = [r,g,b]
                    # curr = np.array(curr,dtype=np.float64)*(1-alpha)
                    # r,g,b = curr+prev
                    # Draw after transparency
                    gpu.GPU.draw_pixel(p.get_flat_pixel(), gpu.GPU.RGB8, (r,g,b))
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.renderer.framebuffers["FRONT"])
        return

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

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

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

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
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
