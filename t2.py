from math import pow, sqrt, pi, e
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def aprox2(i, n):
    p1 = ( B*(1-o) - (o/6)*(pow(o, 2) - 3*o + 2) )*U[i+1][n]
    p2 = ( B*(2 - 3*o) - (o/2)*(pow(o,2) - 2*o - 1) )*U[i][n]
    p3 = ( B*(1-3*o) - (o/2)*(pow(o, 2) - o - 2) )*U[i-1][n]
    p4 = ( B*o + (o/6)*(pow(o, 2) - 1) )*U[i-2][n]
    return U[i][n] + p1 - p2 + p3 + p4


def aprox1(i, n):
    p1 = a*( (U[i+1][n] - 2*U[i][n] + U[i-1][n])/pow(delta_x, 2) )
    p2 = u*( (U[i+1][n] - U[i-1][n])/2*delta_x )
    return delta_t*(p1 - p2) - U[i][n]


def f(x):
    sigma = 0.02
    mu = 0.3

    exp = -(1/2)*pow((x-mu)/sigma, 2)
    return pow(e, exp)/(sqrt(2*pi)*sigma)


def g(x):
    return x*(1-x)


def teste(i, j):
    return (U[i-1][j] - 2*U[i][j] + U[i+1][j])*delta_t/pow(delta_x, 2) + U[i][j]

fig = plt.figure()
ax = plt.axes(projection='3d')

n = 10
delta_x = 3.2
delta_t = 1
x = [i*delta_x for i in range(n)]  # valores de x
t = [i*delta_t for i in range(n)]  # valores de t

u = 3  # velocidade
a = 5  # parametro de difusao

o = u*delta_t/delta_x
B = a*delta_t/pow(delta_x, 2)
Pe = u*delta_x/(2*a)

courant = (u * delta_t)  / (delta_x) #courant deve em teoria ser proximo de 1 pro bagulho ficar legal
peclet = (courant/2)/(a * delta_t/delta_x**2)

print('courant:', courant)
print('peclet:', peclet)

U = []
for _ in range(n):
    U.append([0]*n)

for i in range(n):
    U[-1][i] = 1


for Tj in range(n-1):
    for Xi in range(1, n-1):
        U[Xi][Tj+1] = aprox1(Xi, Tj)

for linha in U:
    for item in linha:
        print(round(item, 5), end=" ")
    print()
G = np.matrix(U)
X, T = np.meshgrid(np.arange(G.shape[0]), np.arange(G.shape[1]))

surf = ax.plot_surface(X, T, G, cmap= 'coolwarm', linewidth=0, antialiased=True)
#ax.set_zlim(0, 1)
ax.set_xlabel('delta_x')
ax.set_ylabel('delta_t')
ax.set_zlabel('delta_h')
plt.show()
