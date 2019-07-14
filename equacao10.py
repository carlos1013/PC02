from math import pow, sqrt, pi, e
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
Ok, vamos tentar representar uma grid aqui sem morrer, pra fazer o trabalho da
equacao 5.2.10, com alfa sendo o parametro de difusao e u a velocidade do
esquema, U(x,0) = 1 e U(0, t) = 0. Primeiramente a gente vai adotar a forma
U[n+1][j] = gama*U[n][j+1] + (1 - Cr - 2gama)*U[n][j] + (Cr + gama)U[n][j-1],
de acordo com Abott & Basco. Isso vai permitir preencher a grid via fatoracao LU,
e se tudo der certo ele vai sair lindjo.
'''
def f(x):
    sigma = 0.02
    mu = 0.3

    exp = -(1/2)*pow((x-mu)/sigma, 2)
    return pow(e, exp)/(sqrt(2*pi)*sigma)

def preenche_A(matriz, courant, gama):
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            if(j == i):
                matriz[i][j] =round(1 - courant - 2 * gama, 5)
                if(j > 0):
                    matriz[i][j-1] = round(courant + gama, 5)
                if(j < len(matriz)-1):
                    matriz[i][j+1] = round(gama, 5)
    return matriz

def preenche_Matriz(grid, matriz_A, matriz_U):
    for i in range(len(grid)-2, 0, -1):
        matriz_U = np.matmul(matriz_A, matriz_U)#multiplicando as matrizes
        for j in range(1, len(grid[0])):
            grid[i][j] = matriz_U[j-1][0]
    matriz_U = np.matmul(matriz_A, matriz_U)#multiplicando as matrizes
    for j in range(1, len(grid[0])):
        grid[0][j] = matriz_U[j-1][0]
    return grid

fig = plt.figure()
ax = plt.axes(projection='3d')

#definindo valores iniciais
alfa = 40
u = 8

delta_t = 1
delta_x = 9

x_atual = 1
t_atual = 0

tempototal = []
passototal = []

#definindo Cr, Peclet e Gama
courant = (u * delta_t)  / (delta_x) #courant deve em teoria ser proximo de 1 pro bagulho ficar legal
peclet = (courant/2)/(alfa * delta_t/delta_x**2)
gama = alfa* delta_t / delta_x**2
text_peclet = 'Peclet:'+ str(peclet)
text_courant = 'Courant:' + str(courant)
text_gama = 'Gama:' + str(gama)
#Fazendo 3 matrizes, as duas necessarias pra fatoracao LU e a q vai ter os results mesmo

grid = [[0]*100 for i in range (100)] #grid onde a gente vai preencher os valores
matriz_U = [[0]]*(len(grid[0])-1)
matriz_A = [[0]*(len(grid[0])-1) for i in range ((len(grid[0])-1))]
matriz_A = preenche_A(matriz_A, courant, gama)
#print(len(matriz_U),len(matriz_U[0]), len(matriz_A), len(matriz_A[0]))
for i in range(0, len(grid[0])):
    x = (i)/(len(grid))
    grid[-1][i] = f(x)

for i in range(1, len(grid[0])-1):
    matriz_U[i-1][0] = grid[-1][i]


for i in range(0, len(grid)-1):
    tempototal.append(i * delta_t)
    passototal.append(i * delta_x)



grid = preenche_Matriz(grid, matriz_A, matriz_U)
for i in range(len(grid)):
    print(grid[i])
G = [[0]*int((len(grid[0])/2)) for i in range(0, int((len(grid)/2)))]
for i in range(0, int((len(grid)/2))):
    for j in range(0, int((len(grid[0])/2))):
        G[i][j] = grid[int((len(grid)/4)) + i][int((len(grid[0])/4)) + j]
G = np.matrix(grid)
X, T = np.meshgrid(np.arange(G.shape[0]),np.arange(G.shape[1]))
#G =G.reshape(15,50)
surf = ax.plot_surface(X, T, G, cmap= 'hot', linewidth=0, antialiased=True)
#ax.scatter3D(passototal, tempototal, G, c=G, cmap='Greens')
ax.text2D(0.05, 0.90, text_peclet, transform=ax.transAxes)
ax.text2D(0.05, 0.85, text_courant, transform=ax.transAxes)
ax.text2D(0.05, 0.80, text_gama, transform=ax.transAxes)
ax.set_zlim(0, 1)
ax.set_xlabel('delta_x')
ax.set_ylabel('delta_t')
ax.set_zlabel('delta_h')
#plt.matshow(G, cmap = 'hot')
plt.show()
