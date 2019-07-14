import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''Ok, vamos tentar representar uma grid aqui sem morrer, pra fazer o trabalho
da equacao 5.2.10, com alfa sendo o parametro de difusao e u a velocidade do
esquema, U(x,0) = 1 e U(0, t) = 0'''


def metodo_Explicito(grid, i, j):
    if(j < len(grid) - 1):
        grid[i-1][j] = delta_t * ((alfa * ((grid[i][j-1] - 2 * grid[i][j] + grid[i][j+1])/(delta_x**2))) - (u * (grid[i][j+1] - grid[i][j-1]) / (2*delta_x))) + grid[i][j]
        #grid[i-1][j] = round(grid[i-1][j], 10)
    else:#tratando a pontinha da matriz
        grid[i-1][j] = delta_t *((alfa * ((grid[i][j-1] - 2 * grid[i][j] + 0)/(delta_x**2))) - (u * (0 - grid[i][j-1]) / (2*delta_x))) + grid[i][j]
        #grid[i-1][j] = round(grid[i-1][j], 10)
    return grid[i-1][j]

def preenche_Matriz(grid): #fazendo para t = 100, no esquema da tese do cara
    global resultados_U, alfa, u, delta_x
    for i in range(len(grid)-1, 0, -1):
        for j in range(1, len(grid)):
            if(i + j) == 51 and (i < len(grid)) and (i > 1):
                #print(i, j)
                grid[i-1][j] = metodo_Explicito(grid, i, j)
                grid[i-1][j-1] = metodo_Explicito(grid, i, j-1)
                grid[i-1][j+1] = metodo_Explicito(grid, i, j+1)
    '''k = 0
    #preenchendo o resto da grid, ate a metade
    for i in range(len(grid) - 2, 0, -1):
        aux = int(len(grid)/2 + k)
        if(aux > len(grid) - 1):
            aux = len(grid)-1
        print(aux)
        while(grid[i][aux] != 0):
            aux -= 1
        while (aux >= len(grid)/2):
            #print(i, aux)
            grid[i][aux] = metodo_Explicito(grid, i, aux)
            aux -= 1
        k += 1'''
    return grid

#definindo valores iniciais

fig = plt.figure()
ax = plt.axes(projection='3d')

#deltas de x e y iguais pra comecar com uma grid bonitinhas
delta_t = 1
delta_x = 3.2

x_atual = 1
t_atual = 0

tempototal = []
passototal = []
grid = [[0]*51 for i in range (51)] #grid onde a gente vai calcular os valores

for i in range(len(grid)):
    tempototal.append(i * delta_t)
    passototal.append(i * delta_x)

for i in range(len(grid)):
    grid[-1][i] = 1

#zero criterio pra comecar o u e alfa vai dar ruim com certeza
alfa = 5

u =  3


courant = (u * delta_t)  / (delta_x) #courant deve em teoria ser proximo de 1 pro bagulho ficar legal
peclet = (courant/2)/(alfa * delta_t/delta_x**2)
grid = preenche_Matriz(grid)
for i in range(len(grid)):
    print(grid[i])
print('courant:', courant)
print('peclet:', peclet)
G = np.matrix(grid)
X, T = np.meshgrid(np.arange(G.shape[0]), np.arange(G.shape[1]))

surf = ax.plot_surface(X, T, G, cmap= 'coolwarm', linewidth=0, antialiased=True)
#ax.set_zlim(0, 1)
ax.set_xlabel('delta_x')
ax.set_ylabel('delta_t')
ax.set_zlabel('delta_h')
plt.matshow(grid)
plt.show()
