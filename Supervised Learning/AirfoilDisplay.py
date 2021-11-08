from cmath import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
import pylab

# importing  all the functions
from Potential import *

def Airfoil(delta, lamda, alpha, n, n_points):

    A = Airfoil_boundary(delta, lamda, n, n_points);    

    x = [ele.real for ele in A] # extract real part
    y = [ele.imag for ele in A] # extract imaginary part


    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Airfoil')
    plt.plot(x, y)
    plt.show()

    Cp = Pressure_field_boundary(alpha, delta, lamda, n,  n_points)

    plt.plot(x[1:int(n_points/2)+1], -Cp[1:int(n_points/2)+1], color = "red", label = "extradós")
    plt.plot(x[int(n_points/2):n_points], -Cp[int(n_points/2):n_points], color = "green", label = "intradós")
    plt.xlim(min(x), max(x))
    plt.ylim(- 1, max(-Cp))
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x - axis')
    plt.ylabel('cp -  axis')
    plt.title('Cp')
    plt.legend()
    plt.show()

    x2 = np.linspace(min(x) - 2, max(x) + 2, 200)
    y2 = np.linspace(min(y) - 2, max(y) + 2, 200)
    contour = np.zeros((np.size(x2), np.size(y2)))
    CP_field = np.zeros((np.size(x2), np.size(y2)))

    for i  in range(0,np.size(x2)):
        for j  in range(0,np.size(y2)):

            val = InverseTransform(complex(x2[i],y2[j]), delta, lamda, n)
            contour[j,i] = Potencial(val, alpha, delta, lamda, n)
            temp = Potential_derivative(val, alpha, delta, lamda, n)
            CP_field [j,i] = 1 - abs(temp**2)


    x2max = max(x2)
    x2min = min(x2)
    y2max = max(y2)
    y2min = min(y2)

    x2, y2 = meshgrid(x2,y2)

    fig = plt.figure()

    fig.add_subplot(211)
    plt.title('Stream lines')
    plt.xlim(x2min, x2max)
    plt.ylim(y2min, y2max)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.contour(x2, y2, contour, 50, cmap='RdGy')
    plt.plot(x, y)
    plt.colorbar()

    fig.add_subplot(212)
    plt.title('Pressure Field')
    plt.xlabel('X')
    plt.ylabel('Y')

    normi = mpl.colors.Normalize(vmin=-2, vmax = 1)
    cont_PSD = plt.contourf(x2, y2, CP_field , 500,linestyle=None,norm=normi, extend='both')

    plt.plot(x, y)
    plt.colorbar()
    plt.show()
