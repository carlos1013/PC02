import math
import matplotlib.pyplot as plt
import numpy as np

'''Ok, vamos tentar representar uma grid aqui sem morrer, pra fazer o trabalho
da equacao 5.2.10, com alfa sendo o parametro de difusao e u a velocidade do
esquema, U(x,0) = 1 e U(0, t) = 0'''


def metodo_Explicito(grid, i, j):
    if(j < len(grid) - 1):
        grid[i-1][j] = (alfa * ((grid[i][j-1] - 2 * grid[i][j] + grid[i][j+1])/(delta_x**2))) - (u * (grid[i][j+1] - grid[i][j-1]) / (2*delta_x)) + grid[i][j]
    else:#tratando o fim da matriz
        grid[i-1][j] = (alfa * ((grid[i][j-1] - 2 * grid[i][j] + 0)/(delta_x**2))) - (u * (0 - grid[i][j-1]) / (2*delta_x)) + grid[i][j]
    return grid[i-1][j]

def preenche_Matriz(grid): #fazendo para t = 100, no esquema da tese do cara
    global resultados_U, alfa, u, delta_x
    for i in range(len(grid)-1, 0, -1):
        for j in range(5, len(grid)):
            if(i + j) == 15 and (i < 11) and (i > 5):
                print(i, j)
                grid[i-1][j] = metodo_Explicito(grid, i, j)
                grid[i-1][j-1] = metodo_Explicito(grid, i, j-1)
                grid[i-1][j+1] = metodo_Explicito(grid, i, j+1)
    k = 0
    for i in range(len(grid) - 2):
        aux = len(grid)/2 + k
        while(grid[i][aux] != 0):
            aux -= 1
        
    return grid

#definindo valores iniciais

#deltas de x e y iguais pra comecar com uma grid bonitinhas
delta_t = 1
delta_x = 10

x_atual = 1
t_atual = 0

tempototal = 100

grid = [[0]*11 for i in range (11)] #grid onde a gente vai calcular os valores
for i in range(len(grid)):
    grid[-1][i] = 1

#zero criterio pra comecar o u e alfa vai dar ruim com certeza
alfa = 0.5
u =  0.7


courant = (u * delta_t)  / (delta_x) #courant deve em teoria ser proximo de 1 pro bagulho ficar legal
grid = preenche_Matriz(grid)
for i in range(len(grid)):
    print(grid[i])
