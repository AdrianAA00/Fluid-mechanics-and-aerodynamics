from cmath import pi
from xml.etree.ElementInclude import XINCLUDE
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
import pylab
from math import atan, cos, pi, sin
from numpy.linalg import *
from numpy import transpose
import sympy 

# importing  all the functions
from Potential import *



def Airfoil_net(R_max, n_R, delta, lamda, alpha, n, n_points):

    x = np.zeros((n_R,n_points), dtype=np.double)
    y = np.zeros((n_R,n_points), dtype=np.double)   
    
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Airfoil')
  
    for i in range(n_R):
        if i == 0:
            R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
        else:
            #R = (R_max-math.sqrt(delta**2 + (1 + lamda)**2))*i/(n_R-1) + math.sqrt(delta**2 + (1 + lamda)**2)
            R = (R_max-math.sqrt(delta**2 + (1 + lamda)**2))*(1-cos((i/(n_R-1))*pi/2)) + math.sqrt(delta**2 + (1 + lamda)**2)

        A = Airfoil_R(R,delta, lamda, n, n_points-1)    

        x[i,:] = [ele.real for ele in A/R_max] # extract real part
        y[i,:] = [ele.imag for ele in A/R_max] # extract imaginary part

        plt.plot(x[i,:], y[i,:],c="black")
        plt.scatter(x[i,:], y[i,:],c="blue", s= 5)

    plt.show()
    return x,y

def Derivatives(x,y,n_points,n_R):

    ##Dx
    Dx = np.zeros((n_R*n_points,n_R*n_points), dtype=np.double)
    for i in range(n_R):
        for j in range(n_points):
            if i  < n_R-1 and j  < n_points-1 : #Forward derivatives 
                x1 = x[i+1,j]-x[i,j]
                x2 = x[i,j+1]-x[i,j]
                y1 = y[i+1,j]-y[i,j]
                y2 = y[i,j+1]-y[i,j]
                Dx[j + i*n_points,j + i*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
                Dx[j + i*n_points,j + 1 + i*n_points] = -y1/((x1-x2*y1/y2)*y2)
                Dx[j + i*n_points,j + (i+1)*n_points] = 1/(x1-x2*y1/y2)

            elif i == n_R - 1 and j > 0:  #Backward derivatives in last layer
                x1 = -x[i,j]+x[i-1,j]
                x2 = -x[i,j]+x[i,j-1]
                y1 = -y[i,j]+y[i-1,j]
                y2 = -y[i,j]+y[i,j-1]
                Dx[j + i*n_points,j + i*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
                Dx[j + i*n_points,j - 1 + i*n_points] = -y1/((x1-x2*y1/y2)*y2)
                Dx[j + i*n_points,j + (i-1)*n_points] = 1/(x1-x2*y1/y2)

            elif i < n_R-1 and j == n_points-1 : #Backward derivatives
                x1 = x[i+1,j]-x[i,j]
                x2 = x[i,j-1]-x[i,j]
                y1 = y[i+1,j]-y[i,j]
                y2 = y[i,j-1]-y[i,j]  
                Dx[j + i*n_points,j + i*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
                Dx[j + i*n_points, j-1 + i*n_points] = -y1/((x1-x2*y1/y2)*y2)
                Dx[j + i*n_points,j + (i+1)*n_points] = 1/(x1-x2*y1/y2)

            elif i == n_R - 1 and j == 0 : #Backward derivatives  
                x1 = -x[i,j]+x[i-1,j]
                x2 = -x[i,j]+x[i,n_points-1]
                y1 = -y[i,j]+y[i-1,j]
                y2 = -y[i,j]+y[i,n_points-1]  
                Dx[j + i*n_points,j + i*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
                Dx[j + i*n_points,n_points-1 + i*n_points] = -y1/((x1-x2*y1/y2)*y2)
                Dx[j + i*n_points,j + (i-1)*n_points] = 1/(x1-x2*y1/y2)

    # print("Dx",Dx)

    ##Dy
    Dy = np.zeros((n_R*n_points,n_R*n_points), dtype=np.double)
    for i in range(n_R):
        for j in range(n_points):
            if i  < n_R-1 and j  < n_points-1 : #Forward derivatives 
                x1 = x[i+1,j]-x[i,j]
                x2 = x[i,j+1]-x[i,j]
                y1 = y[i+1,j]-y[i,j]
                y2 = y[i,j+1]-y[i,j]
                Dy[j + i*n_points,j + i*n_points] = (-1 + x1/x2)/(y1-y2*x1/x2)
                Dy[j + i*n_points,j + 1 + i*n_points] = -x1/((y1-y2*x1/x2)*x2)
                Dy[j + i*n_points,j + (i+1)*n_points] = 1/(y1-y2*x1/x2)

            elif i == n_R - 1 and j > 0:  #Backward derivatives in last layer
                x1 = -x[i,j]+x[i-1,j]
                x2 = -x[i,j]+x[i,j-1]
                y1 = -y[i,j]+y[i-1,j]
                y2 = -y[i,j]+y[i,j-1]
                Dy[j + i*n_points,j + i*n_points] = (-1 + x1/x2)/(y1-y2*x1/x2)
                Dy[j + i*n_points,j - 1 + i*n_points] = -x1/((y1-y2*x1/x2)*x2)
                Dy[j + i*n_points,j + (i-1)*n_points] = 1/(y1-y2*x1/x2)

            elif i < n_R-1 and j == n_points-1 : #Forward derivatives
                x1 = x[i+1,j]-x[i,j]
                x2 = x[i,j-1]-x[i,j]
                y1 = y[i+1,j]-y[i,j]
                y2 = y[i,j-1]-y[i,j]  
                Dy[j + i*n_points,j + i*n_points] = (-1 + x1/x2)/(y1-y2*x1/x2)
                Dy[j + i*n_points, j-1 + i*n_points] = -x1/((y1-y2*x1/x2)*x2)
                Dy[j + i*n_points,j + (i+1)*n_points] = 1/(y1-y2*x1/x2)

            elif i == n_R - 1 and j == 0 : #Backward derivatives  
                x1 = -x[i,j]+x[i-1,j]
                x2 = -x[i,j]+x[i,n_points-1]
                y1 = -y[i,j]+y[i-1,j]
                y2 = -y[i,j]+y[i,n_points-1]  
                Dy[j + i*n_points,j + i*n_points] = (-1 + x1/x2)/(y1-y2*x1/x2)
                Dy[j + i*n_points,n_points-1 + i*n_points] = -x1/((y1-y2*x1/x2)*x2)
                Dy[j + i*n_points,j + (i-1)*n_points] = 1/(y1-y2*x1/x2)

    # print("Dy",Dy)

    #Dxx, Dyy, Dxy
    F0 = F1 = F2 = F3 = F4 = F5 = 1
    Dxx = np.zeros((n_R*n_points,n_R*n_points), dtype=np.double)
    Dyy = np.zeros((n_R*n_points,n_R*n_points), dtype=np.double)
    Dxy = np.zeros((n_R*n_points,n_R*n_points), dtype=np.double)

    for i in range(n_R):
        for j in range(n_points):

            if i  == 0 and 0 == j: #Forward derivatives 
                x1 = x[i,j+1]-x[i,j]
                x2 = x[i,j+2]-x[i,j]
                x3 = x[i+1,j]-x[i,j]
                x4 = x[i+1,j+1]-x[i,j]
                x5 = x[i+1,j+2]-x[i,j]
                y1 = y[i,j+1]-y[i,j]
                y2 = y[i,j+2]-y[i,j]
                y3 = y[i+1,j]-y[i,j]
                y4 = y[i+1,j+1]-y[i,j]
                y5 = y[i+1,j+2]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + 2 + i*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + (i+1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j + 1 + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 2 + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + 2 + i*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + (i+1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j + 1 + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 2 + (i+1)*n_points] = - (2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + 2 + i*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + (i+1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j + 1 + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 2 + (i+1)*n_points] = - (1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)

            if i  == 0 and 0 < j < n_points - 1: #Forward derivatives 
                x1 = x[i,j-1]-x[i,j]
                x2 = x[i,j+1]-x[i,j]
                x3 = x[i+1,j-1]-x[i,j]
                x4 = x[i+1,j]-x[i,j]
                x5 = x[i+1,j+1]-x[i,j]
                y1 = y[i,j-1]-y[i,j]
                y2 = y[i,j+1]-y[i,j]
                y3 = y[i+1,j-1]-y[i,j]
                y4 = y[i+1,j]-y[i,j]
                y5 = y[i+1,j+1]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
            
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j - 1 + (i+1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 1 + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j - 1 + (i+1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 1 + (i+1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j - 1 + (i+1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 1 + (i+1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


            if i == 0 and j == n_points - 1:  #Modified in boundary
                x1 = x[i,j-1]-x[i,j]
                x2 = x[i,j-2]-x[i,j]
                x3 = x[i+1,j]-x[i,j]
                x4 = x[i+1,j-1]-x[i,j]
                x5 = x[i+1,j-2]-x[i,j]
                y1 = y[i,j-1]-y[i,j]
                y2 = y[i,j-2]-y[i,j]
                y3 = y[i+1,j]-y[i,j]
                y4 = y[i+1,j-1]-y[i,j]
                y5 = y[i+1,j-2]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j - 2 + i*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + (i+1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j - 1 + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j - 2 + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j - 2 + i*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + (i+1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j - 1 + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j - 2 + (i+1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j - 2 + i*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + (i+1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j - 1 + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j - 2 + (i+1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)

            if 0 < i < n_R-1 and 0 == j: #Middle points
                x1 = x[i-1,j]-x[i,j]
                x2 = x[i-1,j+1]-x[i,j]
                x3 = x[i,j+1]-x[i,j]
                x4 = x[i+1,j]-x[i,j]
                x5 = x[i+1,j+1]-x[i,j]
                y1 = y[i-1,j]-y[i,j]
                y2 = y[i-1,j+1]-y[i,j]
                y3 = y[i,j+1]-y[i,j]
                y4 = y[i+1,j]-y[i,j]
                y5 = y[i+1,j+1]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + 1 + (i-1)*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 1 + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)
                
                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + 1 + (i-1)*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 1 + (i+1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + 1 + (i-1)*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 1 + (i+1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


            if 0 < i < n_R-1 and 0 < j < n_points-1 : #Middle points
                x1 = x[i-1,j]-x[i,j]
                x2 = x[i,j-1]-x[i,j]
                x3 = x[i,j+1]-x[i,j]
                x4 = x[i+1,j-1]-x[i,j]
                x5 = x[i+1,j+1]-x[i,j]
                y1 = y[i-1,j]-y[i,j]
                y2 = y[i,j-1]-y[i,j]
                y3 = y[i,j+1]-y[i,j]
                y4 = y[i+1,j-1]-y[i,j]
                y5 = y[i+1,j+1]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j - 1 + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 1 + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)
                
                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j - 1 + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 1 + (i+1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j - 1 + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 1 + (i+1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


            if 0 < i < n_R-1 and j == n_points-1 : #Middle points
                x1 = x[i-1,j-1]-x[i,j]
                x2 = x[i-1,j]-x[i,j]
                x3 = x[i,j-1]-x[i,j]
                x4 = x[i+1,j-1]-x[i,j]
                x5 = x[i+1,j]-x[i,j]
                y1 = y[i-1,j-1]-y[i,j]
                y2 = y[i-1,j]-y[i,j]
                y3 = y[i,j-1]-y[i,j]
                y4 = y[i+1,j-1]-y[i,j]
                y5 = y[i+1,j]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j - 1 + (i-1)*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j - 1 + (i+1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + (i+1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)
                
                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j - 1 + (i-1)*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j - 1 + (i+1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + (i+1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j - 1 + (i-1)*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j - 1 + (i+1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + (i+1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)

            if i == n_R-1 and 0 == j: #Exterior points
                x1 = x[i-1,j]-x[i,j]
                x2 = x[i-1,j+1]-x[i,j]
                x3 = x[i-1,j+2]-x[i,j]
                x4 = x[i,j+1]-x[i,j]
                x5 = x[i,j+2]-x[i,j]
                y1 = y[i-1,j]-y[i,j]
                y2 = y[i-1,j+1]-y[i,j]
                y3 = y[i-1,j+2]-y[i,j]
                y4 = y[i,j+1]-y[i,j]
                y5 = y[i,j+2]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + 1 + (i-1)*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + 2 + (i-1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 2 + i*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + 1 + (i-1)*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + 2 + (i-1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 2 + i*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + 1 + (i-1)*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + 2 + (i-1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 2 + i*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


            if i == n_R-1 and 0 <  j < n_points-1 : #Exterior points
                x1 = x[i,j-1]-x[i,j]
                x2 = x[i,j+1]-x[i,j]
                x3 = x[i-1,j-1]-x[i,j]
                x4 = x[i-1,j]-x[i,j]
                x5 = x[i-1,j+1]-x[i,j]
                y1 = y[i,j-1]-y[i,j]
                y2 = y[i,j+1]-y[i,j]
                y3 = y[i-1,j-1]-y[i,j]
                y4 = y[i-1,j]-y[i,j]
                y5 = y[i-1,j+1]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j + 1 + i*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j - 1 + (i-1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j + 1 + (i-1)*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j + 1 + i*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j - 1 + (i-1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j + 1 + (i-1)*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j + 1 + i*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j - 1 + (i-1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j + 1 + (i-1)*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


            if i == n_R-1 and j == n_points-1 : #Exterior points
                x1 = x[i-1,j-2]-x[i,j]
                x2 = x[i-1,j-1]-x[i,j]
                x3 = x[i-1,j]-x[i,j]
                x4 = x[i,j-2]-x[i,j]
                x5 = x[i,j-1]-x[i,j]
                y1 = y[i-1,j-2]-y[i,j]
                y2 = y[i-1,j-1]-y[i,j]
                y3 = y[i-1,j]-y[i,j]
                y4 = y[i,j-2]-y[i,j]
                y5 = y[i,j-1]-y[i,j]
                xx1 = x1**2
                xx2 = x2**2
                xx3 = x3**2
                xx4 = x4**2
                xx5 = x5**2
                yy1 = y1**2
                yy2 = y2**2
                yy3 = y3**2
                yy4 = y4**2
                yy5 = y5**2
                xy1 = y1*x1
                xy2 = y2*x2
                xy3 = y3*x3
                xy4 = y4*x4
                xy5 = y5*x5

                A = (x1*xx2*xy3*y4*yy5 - x1*xx2*xy3*y5*yy4 - x1*xx2*xy4*y3*yy5 + x1*xx2*xy4*y5*yy3 + x1*xx2*xy5*y3*yy4 - x1*xx2*xy5*y4*yy3 - x1*xx3*xy2*y4*yy5 + x1*xx3*xy2*y5*yy4 + x1*xx3*xy4*y2*yy5 - x1*xx3*xy4*y5*yy2 - x1*xx3*xy5*y2*yy4 + x1*xx3*xy5*y4*yy2 + x1*xx4*xy2*y3*yy5 - x1*xx4*xy2*y5*yy3 - x1*xx4*xy3*y2*yy5 + x1*xx4*xy3*y5*yy2 + x1*xx4*xy5*y2*yy3 - x1*xx4*xy5*y3*yy2 - x1*xx5*xy2*y3*yy4 + x1*xx5*xy2*y4*yy3 + x1*xx5*xy3*y2*yy4 - x1*xx5*xy3*y4*yy2 - x1*xx5*xy4*y2*yy3 + x1*xx5*xy4*y3*yy2 - x2*xx1*xy3*y4*yy5 + x2*xx1*xy3*y5*yy4 + x2*xx1*xy4*y3*yy5 - x2*xx1*xy4*y5*yy3 - x2*xx1*xy5*y3*yy4 + x2*xx1*xy5*y4*yy3 + x2*xx3*xy1*y4*yy5 - x2*xx3*xy1*y5*yy4 - x2*xx3*xy4*y1*yy5 + x2*xx3*xy4*y5*yy1 + x2*xx3*xy5*y1*yy4 - x2*xx3*xy5*y4*yy1 - x2*xx4*xy1*y3*yy5 + x2*xx4*xy1*y5*yy3 + x2*xx4*xy3*y1*yy5 - x2*xx4*xy3*y5*yy1 - x2*xx4*xy5*y1*yy3 + x2*xx4*xy5*y3*yy1 + x2*xx5*xy1*y3*yy4 - x2*xx5*xy1*y4*yy3 - x2*xx5*xy3*y1*yy4 + x2*xx5*xy3*y4*yy1 + x2*xx5*xy4*y1*yy3 - x2*xx5*xy4*y3*yy1 + x3*xx1*xy2*y4*yy5 - x3*xx1*xy2*y5*yy4 - x3*xx1*xy4*y2*yy5 + x3*xx1*xy4*y5*yy2 + x3*xx1*xy5*y2*yy4 - x3*xx1*xy5*y4*yy2 - x3*xx2*xy1*y4*yy5 + x3*xx2*xy1*y5*yy4 + x3*xx2*xy4*y1*yy5 - x3*xx2*xy4*y5*yy1 - x3*xx2*xy5*y1*yy4 + x3*xx2*xy5*y4*yy1 + x3*xx4*xy1*y2*yy5 - x3*xx4*xy1*y5*yy2 - x3*xx4*xy2*y1*yy5 + x3*xx4*xy2*y5*yy1 + x3*xx4*xy5*y1*yy2 - x3*xx4*xy5*y2*yy1 - x3*xx5*xy1*y2*yy4 + x3*xx5*xy1*y4*yy2 + x3*xx5*xy2*y1*yy4 - x3*xx5*xy2*y4*yy1 - x3*xx5*xy4*y1*yy2 + x3*xx5*xy4*y2*yy1 - x4*xx1*xy2*y3*yy5 + x4*xx1*xy2*y5*yy3 + x4*xx1*xy3*y2*yy5 - x4*xx1*xy3*y5*yy2 - x4*xx1*xy5*y2*yy3 + x4*xx1*xy5*y3*yy2 + x4*xx2*xy1*y3*yy5 - x4*xx2*xy1*y5*yy3 - x4*xx2*xy3*y1*yy5 + x4*xx2*xy3*y5*yy1 + x4*xx2*xy5*y1*yy3 - x4*xx2*xy5*y3*yy1 - x4*xx3*xy1*y2*yy5 + x4*xx3*xy1*y5*yy2 + x4*xx3*xy2*y1*yy5 - x4*xx3*xy2*y5*yy1 - x4*xx3*xy5*y1*yy2 + x4*xx3*xy5*y2*yy1 + x4*xx5*xy1*y2*yy3 - x4*xx5*xy1*y3*yy2 - x4*xx5*xy2*y1*yy3 + x4*xx5*xy2*y3*yy1 + x4*xx5*xy3*y1*yy2 - x4*xx5*xy3*y2*yy1 + x5*xx1*xy2*y3*yy4 - x5*xx1*xy2*y4*yy3 - x5*xx1*xy3*y2*yy4 + x5*xx1*xy3*y4*yy2 + x5*xx1*xy4*y2*yy3 - x5*xx1*xy4*y3*yy2 - x5*xx2*xy1*y3*yy4 + x5*xx2*xy1*y4*yy3 + x5*xx2*xy3*y1*yy4 - x5*xx2*xy3*y4*yy1 - x5*xx2*xy4*y1*yy3 + x5*xx2*xy4*y3*yy1 + x5*xx3*xy1*y2*yy4 - x5*xx3*xy1*y4*yy2 - x5*xx3*xy2*y1*yy4 + x5*xx3*xy2*y4*yy1 + x5*xx3*xy4*y1*yy2 - x5*xx3*xy4*y2*yy1 - x5*xx4*xy1*y2*yy3 + x5*xx4*xy1*y3*yy2 + x5*xx4*xy2*y1*yy3 - x5*xx4*xy2*y3*yy1 - x5*xx4*xy3*y1*yy2 + x5*xx4*xy3*y2*yy1)
                
                #Dxx
                #F0
                Dxx[j + i*n_points,j + i*n_points] = (2/A)*(F0*x1*xy2*y3*yy4 - F0*x1*xy2*y4*yy3 - F0*x1*xy3*y2*yy4 + F0*x1*xy3*y4*yy2 + F0*x1*xy4*y2*yy3 - F0*x1*xy4*y3*yy2 - F0*x2*xy1*y3*yy4 + F0*x2*xy1*y4*yy3 + F0*x2*xy3*y1*yy4 - F0*x2*xy3*y4*yy1 - F0*x2*xy4*y1*yy3 + F0*x2*xy4*y3*yy1 + F0*x3*xy1*y2*yy4 - F0*x3*xy1*y4*yy2 - F0*x3*xy2*y1*yy4 + F0*x3*xy2*y4*yy1 + F0*x3*xy4*y1*yy2 - F0*x3*xy4*y2*yy1 - F0*x4*xy1*y2*yy3 + F0*x4*xy1*y3*yy2 + F0*x4*xy2*y1*yy3 - F0*x4*xy2*y3*yy1 - F0*x4*xy3*y1*yy2 + F0*x4*xy3*y2*yy1 - F0*x1*xy2*y3*yy5 + F0*x1*xy2*y5*yy3 + F0*x1*xy3*y2*yy5 - F0*x1*xy3*y5*yy2 - F0*x1*xy5*y2*yy3 + F0*x1*xy5*y3*yy2 + F0*x2*xy1*y3*yy5 - F0*x2*xy1*y5*yy3 - F0*x2*xy3*y1*yy5 + F0*x2*xy3*y5*yy1 + F0*x2*xy5*y1*yy3 - F0*x2*xy5*y3*yy1 - F0*x3*xy1*y2*yy5 + F0*x3*xy1*y5*yy2 + F0*x3*xy2*y1*yy5 - F0*x3*xy2*y5*yy1 - F0*x3*xy5*y1*yy2 + F0*x3*xy5*y2*yy1 + F0*x5*xy1*y2*yy3 - F0*x5*xy1*y3*yy2 - F0*x5*xy2*y1*yy3 + F0*x5*xy2*y3*yy1 + F0*x5*xy3*y1*yy2 - F0*x5*xy3*y2*yy1 + F0*x1*xy2*y4*yy5 - F0*x1*xy2*y5*yy4 - F0*x1*xy4*y2*yy5 + F0*x1*xy4*y5*yy2 + F0*x1*xy5*y2*yy4 - F0*x1*xy5*y4*yy2 - F0*x2*xy1*y4*yy5 + F0*x2*xy1*y5*yy4 + F0*x2*xy4*y1*yy5 - F0*x2*xy4*y5*yy1 - F0*x2*xy5*y1*yy4 + F0*x2*xy5*y4*yy1 + F0*x4*xy1*y2*yy5 - F0*x4*xy1*y5*yy2 - F0*x4*xy2*y1*yy5 + F0*x4*xy2*y5*yy1 + F0*x4*xy5*y1*yy2 - F0*x4*xy5*y2*yy1 - F0*x5*xy1*y2*yy4 + F0*x5*xy1*y4*yy2 + F0*x5*xy2*y1*yy4 - F0*x5*xy2*y4*yy1 - F0*x5*xy4*y1*yy2 + F0*x5*xy4*y2*yy1 - F0*x1*xy3*y4*yy5 + F0*x1*xy3*y5*yy4 + F0*x1*xy4*y3*yy5 - F0*x1*xy4*y5*yy3 - F0*x1*xy5*y3*yy4 + F0*x1*xy5*y4*yy3 + F0*x3*xy1*y4*yy5 - F0*x3*xy1*y5*yy4 - F0*x3*xy4*y1*yy5 + F0*x3*xy4*y5*yy1 + F0*x3*xy5*y1*yy4 - F0*x3*xy5*y4*yy1 - F0*x4*xy1*y3*yy5 + F0*x4*xy1*y5*yy3 + F0*x4*xy3*y1*yy5 - F0*x4*xy3*y5*yy1 - F0*x4*xy5*y1*yy3 + F0*x4*xy5*y3*yy1 + F0*x5*xy1*y3*yy4 - F0*x5*xy1*y4*yy3 - F0*x5*xy3*y1*yy4 + F0*x5*xy3*y4*yy1 + F0*x5*xy4*y1*yy3 - F0*x5*xy4*y3*yy1 + F0*x2*xy3*y4*yy5 - F0*x2*xy3*y5*yy4 - F0*x2*xy4*y3*yy5 + F0*x2*xy4*y5*yy3 + F0*x2*xy5*y3*yy4 - F0*x2*xy5*y4*yy3 - F0*x3*xy2*y4*yy5 + F0*x3*xy2*y5*yy4 + F0*x3*xy4*y2*yy5 - F0*x3*xy4*y5*yy2 - F0*x3*xy5*y2*yy4 + F0*x3*xy5*y4*yy2 + F0*x4*xy2*y3*yy5 - F0*x4*xy2*y5*yy3 - F0*x4*xy3*y2*yy5 + F0*x4*xy3*y5*yy2 + F0*x4*xy5*y2*yy3 - F0*x4*xy5*y3*yy2 - F0*x5*xy2*y3*yy4 + F0*x5*xy2*y4*yy3 + F0*x5*xy3*y2*yy4 - F0*x5*xy3*y4*yy2 - F0*x5*xy4*y2*yy3 + F0*x5*xy4*y3*yy2) 
                #F1
                Dxx[j + i*n_points,j - 2 + (i-1)*n_points] = (2/A)*(- F1*x2*xy3*y4*yy5 + F1*x2*xy3*y5*yy4 + F1*x2*xy4*y3*yy5 - F1*x2*xy4*y5*yy3 - F1*x2*xy5*y3*yy4 + F1*x2*xy5*y4*yy3 + F1*x3*xy2*y4*yy5 - F1*x3*xy2*y5*yy4 - F1*x3*xy4*y2*yy5 + F1*x3*xy4*y5*yy2 + F1*x3*xy5*y2*yy4 - F1*x3*xy5*y4*yy2 - F1*x4*xy2*y3*yy5 + F1*x4*xy2*y5*yy3 + F1*x4*xy3*y2*yy5 - F1*x4*xy3*y5*yy2 - F1*x4*xy5*y2*yy3 + F1*x4*xy5*y3*yy2 + F1*x5*xy2*y3*yy4 - F1*x5*xy2*y4*yy3 - F1*x5*xy3*y2*yy4 + F1*x5*xy3*y4*yy2 + F1*x5*xy4*y2*yy3 - F1*x5*xy4*y3*yy2)
                #F2
                Dxx[j + i*n_points,j - 1 + (i-1)*n_points] = (2/A)*(F2*x1*xy3*y4*yy5 - F2*x1*xy3*y5*yy4 - F2*x1*xy4*y3*yy5 + F2*x1*xy4*y5*yy3 + F2*x1*xy5*y3*yy4 - F2*x1*xy5*y4*yy3 - F2*x3*xy1*y4*yy5 + F2*x3*xy1*y5*yy4 + F2*x3*xy4*y1*yy5 - F2*x3*xy4*y5*yy1 - F2*x3*xy5*y1*yy4 + F2*x3*xy5*y4*yy1 + F2*x4*xy1*y3*yy5 - F2*x4*xy1*y5*yy3 - F2*x4*xy3*y1*yy5 + F2*x4*xy3*y5*yy1 + F2*x4*xy5*y1*yy3 - F2*x4*xy5*y3*yy1 - F2*x5*xy1*y3*yy4 + F2*x5*xy1*y4*yy3 + F2*x5*xy3*y1*yy4 - F2*x5*xy3*y4*yy1 - F2*x5*xy4*y1*yy3 + F2*x5*xy4*y3*yy1)
                #F3
                Dxx[j + i*n_points,j + (i-1)*n_points] = (2/A)*(- F3*x1*xy2*y4*yy5 + F3*x1*xy2*y5*yy4 + F3*x1*xy4*y2*yy5 - F3*x1*xy4*y5*yy2 - F3*x1*xy5*y2*yy4 + F3*x1*xy5*y4*yy2 + F3*x2*xy1*y4*yy5 - F3*x2*xy1*y5*yy4 - F3*x2*xy4*y1*yy5 + F3*x2*xy4*y5*yy1 + F3*x2*xy5*y1*yy4 - F3*x2*xy5*y4*yy1 - F3*x4*xy1*y2*yy5 + F3*x4*xy1*y5*yy2 + F3*x4*xy2*y1*yy5 - F3*x4*xy2*y5*yy1 - F3*x4*xy5*y1*yy2 + F3*x4*xy5*y2*yy1 + F3*x5*xy1*y2*yy4 - F3*x5*xy1*y4*yy2 - F3*x5*xy2*y1*yy4 + F3*x5*xy2*y4*yy1 + F3*x5*xy4*y1*yy2 - F3*x5*xy4*y2*yy1)
                #F4
                Dxx[j + i*n_points,j - 2 + i*n_points] = (2/A)*(+ F4*x1*xy2*y3*yy5 - F4*x1*xy2*y5*yy3 - F4*x1*xy3*y2*yy5 + F4*x1*xy3*y5*yy2 + F4*x1*xy5*y2*yy3 - F4*x1*xy5*y3*yy2 - F4*x2*xy1*y3*yy5 + F4*x2*xy1*y5*yy3 + F4*x2*xy3*y1*yy5 - F4*x2*xy3*y5*yy1 - F4*x2*xy5*y1*yy3 + F4*x2*xy5*y3*yy1 + F4*x3*xy1*y2*yy5 - F4*x3*xy1*y5*yy2 - F4*x3*xy2*y1*yy5 + F4*x3*xy2*y5*yy1 + F4*x3*xy5*y1*yy2 - F4*x3*xy5*y2*yy1 - F4*x5*xy1*y2*yy3 + F4*x5*xy1*y3*yy2 + F4*x5*xy2*y1*yy3 - F4*x5*xy2*y3*yy1 - F4*x5*xy3*y1*yy2 + F4*x5*xy3*y2*yy1)
                #F5
                Dxx[j + i*n_points,j - 1 + i*n_points] = (2/A)*(- F5*x1*xy2*y3*yy4 + F5*x1*xy2*y4*yy3 + F5*x1*xy3*y2*yy4 - F5*x1*xy3*y4*yy2 - F5*x1*xy4*y2*yy3 + F5*x1*xy4*y3*yy2 + F5*x2*xy1*y3*yy4 - F5*x2*xy1*y4*yy3 - F5*x2*xy3*y1*yy4 + F5*x2*xy3*y4*yy1 + F5*x2*xy4*y1*yy3 - F5*x2*xy4*y3*yy1 - F5*x3*xy1*y2*yy4 + F5*x3*xy1*y4*yy2 + F5*x3*xy2*y1*yy4 - F5*x3*xy2*y4*yy1 - F5*x3*xy4*y1*yy2 + F5*x3*xy4*y2*yy1 + F5*x4*xy1*y2*yy3 - F5*x4*xy1*y3*yy2 - F5*x4*xy2*y1*yy3 + F5*x4*xy2*y3*yy1 + F5*x4*xy3*y1*yy2 - F5*x4*xy3*y2*yy1)

                #Dyy
                #F0
                Dyy[j + i*n_points,j + i*n_points] = -(2/A)*(F0*x1*xx2*xy3*y4 - F0*x1*xx2*xy4*y3 - F0*x1*xx3*xy2*y4 + F0*x1*xx3*xy4*y2 + F0*x1*xx4*xy2*y3 - F0*x1*xx4*xy3*y2 - F0*x2*xx1*xy3*y4 + F0*x2*xx1*xy4*y3 + F0*x2*xx3*xy1*y4 - F0*x2*xx3*xy4*y1 - F0*x2*xx4*xy1*y3 + F0*x2*xx4*xy3*y1 + F0*x3*xx1*xy2*y4 - F0*x3*xx1*xy4*y2 - F0*x3*xx2*xy1*y4 + F0*x3*xx2*xy4*y1 + F0*x3*xx4*xy1*y2 - F0*x3*xx4*xy2*y1 - F0*x4*xx1*xy2*y3 + F0*x4*xx1*xy3*y2 + F0*x4*xx2*xy1*y3 - F0*x4*xx2*xy3*y1 - F0*x4*xx3*xy1*y2 + F0*x4*xx3*xy2*y1 - F0*x1*xx2*xy3*y5 + F0*x1*xx2*xy5*y3 + F0*x1*xx3*xy2*y5 - F0*x1*xx3*xy5*y2 - F0*x1*xx5*xy2*y3 + F0*x1*xx5*xy3*y2 + F0*x2*xx1*xy3*y5 - F0*x2*xx1*xy5*y3 - F0*x2*xx3*xy1*y5 + F0*x2*xx3*xy5*y1 + F0*x2*xx5*xy1*y3 - F0*x2*xx5*xy3*y1 - F0*x3*xx1*xy2*y5 + F0*x3*xx1*xy5*y2 + F0*x3*xx2*xy1*y5 - F0*x3*xx2*xy5*y1 - F0*x3*xx5*xy1*y2 + F0*x3*xx5*xy2*y1 + F0*x5*xx1*xy2*y3 - F0*x5*xx1*xy3*y2 - F0*x5*xx2*xy1*y3 + F0*x5*xx2*xy3*y1 + F0*x5*xx3*xy1*y2 - F0*x5*xx3*xy2*y1 + F0*x1*xx2*xy4*y5 - F0*x1*xx2*xy5*y4 - F0*x1*xx4*xy2*y5 + F0*x1*xx4*xy5*y2 + F0*x1*xx5*xy2*y4 - F0*x1*xx5*xy4*y2 - F0*x2*xx1*xy4*y5 + F0*x2*xx1*xy5*y4 + F0*x2*xx4*xy1*y5 - F0*x2*xx4*xy5*y1 - F0*x2*xx5*xy1*y4 + F0*x2*xx5*xy4*y1 + F0*x4*xx1*xy2*y5 - F0*x4*xx1*xy5*y2 - F0*x4*xx2*xy1*y5 + F0*x4*xx2*xy5*y1 + F0*x4*xx5*xy1*y2 - F0*x4*xx5*xy2*y1 - F0*x5*xx1*xy2*y4 + F0*x5*xx1*xy4*y2 + F0*x5*xx2*xy1*y4 - F0*x5*xx2*xy4*y1 - F0*x5*xx4*xy1*y2 + F0*x5*xx4*xy2*y1 - F0*x1*xx3*xy4*y5 + F0*x1*xx3*xy5*y4 + F0*x1*xx4*xy3*y5 - F0*x1*xx4*xy5*y3 - F0*x1*xx5*xy3*y4 + F0*x1*xx5*xy4*y3 + F0*x3*xx1*xy4*y5 - F0*x3*xx1*xy5*y4 - F0*x3*xx4*xy1*y5 + F0*x3*xx4*xy5*y1 + F0*x3*xx5*xy1*y4 - F0*x3*xx5*xy4*y1 - F0*x4*xx1*xy3*y5 + F0*x4*xx1*xy5*y3 + F0*x4*xx3*xy1*y5 - F0*x4*xx3*xy5*y1 - F0*x4*xx5*xy1*y3 + F0*x4*xx5*xy3*y1 + F0*x5*xx1*xy3*y4 - F0*x5*xx1*xy4*y3 - F0*x5*xx3*xy1*y4 + F0*x5*xx3*xy4*y1 + F0*x5*xx4*xy1*y3 - F0*x5*xx4*xy3*y1 + F0*x2*xx3*xy4*y5 - F0*x2*xx3*xy5*y4 - F0*x2*xx4*xy3*y5 + F0*x2*xx4*xy5*y3 + F0*x2*xx5*xy3*y4 - F0*x2*xx5*xy4*y3 - F0*x3*xx2*xy4*y5 + F0*x3*xx2*xy5*y4 + F0*x3*xx4*xy2*y5 - F0*x3*xx4*xy5*y2 - F0*x3*xx5*xy2*y4 + F0*x3*xx5*xy4*y2 + F0*x4*xx2*xy3*y5 - F0*x4*xx2*xy5*y3 - F0*x4*xx3*xy2*y5 + F0*x4*xx3*xy5*y2 + F0*x4*xx5*xy2*y3 - F0*x4*xx5*xy3*y2 - F0*x5*xx2*xy3*y4 + F0*x5*xx2*xy4*y3 + F0*x5*xx3*xy2*y4 - F0*x5*xx3*xy4*y2 - F0*x5*xx4*xy2*y3 + F0*x5*xx4*xy3*y2) 
                #F1
                Dyy[j + i*n_points,j - 2 + (i-1)*n_points] = -(2/A)*(- F1*x2*xx3*xy4*y5 + F1*x2*xx3*xy5*y4 + F1*x2*xx4*xy3*y5 - F1*x2*xx4*xy5*y3 - F1*x2*xx5*xy3*y4 + F1*x2*xx5*xy4*y3 + F1*x3*xx2*xy4*y5 - F1*x3*xx2*xy5*y4 - F1*x3*xx4*xy2*y5 + F1*x3*xx4*xy5*y2 + F1*x3*xx5*xy2*y4 - F1*x3*xx5*xy4*y2 - F1*x4*xx2*xy3*y5 + F1*x4*xx2*xy5*y3 + F1*x4*xx3*xy2*y5 - F1*x4*xx3*xy5*y2 - F1*x4*xx5*xy2*y3 + F1*x4*xx5*xy3*y2 + F1*x5*xx2*xy3*y4 - F1*x5*xx2*xy4*y3 - F1*x5*xx3*xy2*y4 + F1*x5*xx3*xy4*y2 + F1*x5*xx4*xy2*y3 - F1*x5*xx4*xy3*y2)
                #F2
                Dyy[j + i*n_points,j - 1 + (i-1)*n_points] = -(2/A)*(F2*x1*xx3*xy4*y5 - F2*x1*xx3*xy5*y4 - F2*x1*xx4*xy3*y5 + F2*x1*xx4*xy5*y3 + F2*x1*xx5*xy3*y4 - F2*x1*xx5*xy4*y3 - F2*x3*xx1*xy4*y5 + F2*x3*xx1*xy5*y4 + F2*x3*xx4*xy1*y5 - F2*x3*xx4*xy5*y1 - F2*x3*xx5*xy1*y4 + F2*x3*xx5*xy4*y1 + F2*x4*xx1*xy3*y5 - F2*x4*xx1*xy5*y3 - F2*x4*xx3*xy1*y5 + F2*x4*xx3*xy5*y1 + F2*x4*xx5*xy1*y3 - F2*x4*xx5*xy3*y1 - F2*x5*xx1*xy3*y4 + F2*x5*xx1*xy4*y3 + F2*x5*xx3*xy1*y4 - F2*x5*xx3*xy4*y1 - F2*x5*xx4*xy1*y3 + F2*x5*xx4*xy3*y1)
                #F3
                Dyy[j + i*n_points,j + (i-1)*n_points] = -(2/A)*(- F3*x1*xx2*xy4*y5 + F3*x1*xx2*xy5*y4 + F3*x1*xx4*xy2*y5 - F3*x1*xx4*xy5*y2 - F3*x1*xx5*xy2*y4 + F3*x1*xx5*xy4*y2 + F3*x2*xx1*xy4*y5 - F3*x2*xx1*xy5*y4 - F3*x2*xx4*xy1*y5 + F3*x2*xx4*xy5*y1 + F3*x2*xx5*xy1*y4 - F3*x2*xx5*xy4*y1 - F3*x4*xx1*xy2*y5 + F3*x4*xx1*xy5*y2 + F3*x4*xx2*xy1*y5 - F3*x4*xx2*xy5*y1 - F3*x4*xx5*xy1*y2 + F3*x4*xx5*xy2*y1 + F3*x5*xx1*xy2*y4 - F3*x5*xx1*xy4*y2 - F3*x5*xx2*xy1*y4 + F3*x5*xx2*xy4*y1 + F3*x5*xx4*xy1*y2 - F3*x5*xx4*xy2*y1)
                #F4
                Dyy[j + i*n_points,j - 2 + i*n_points] = -(2/A)*(+ F4*x1*xx2*xy3*y5 - F4*x1*xx2*xy5*y3 - F4*x1*xx3*xy2*y5 + F4*x1*xx3*xy5*y2 + F4*x1*xx5*xy2*y3 - F4*x1*xx5*xy3*y2 - F4*x2*xx1*xy3*y5 + F4*x2*xx1*xy5*y3 + F4*x2*xx3*xy1*y5 - F4*x2*xx3*xy5*y1 - F4*x2*xx5*xy1*y3 + F4*x2*xx5*xy3*y1 + F4*x3*xx1*xy2*y5 - F4*x3*xx1*xy5*y2 - F4*x3*xx2*xy1*y5 + F4*x3*xx2*xy5*y1 + F4*x3*xx5*xy1*y2 - F4*x3*xx5*xy2*y1 - F4*x5*xx1*xy2*y3 + F4*x5*xx1*xy3*y2 + F4*x5*xx2*xy1*y3 - F4*x5*xx2*xy3*y1 - F4*x5*xx3*xy1*y2 + F4*x5*xx3*xy2*y1)
                #F5
                Dyy[j + i*n_points,j - 1 + i*n_points] = -(2/A)*(- F5*x1*xx2*xy3*y4 + F5*x1*xx2*xy4*y3 + F5*x1*xx3*xy2*y4 - F5*x1*xx3*xy4*y2 - F5*x1*xx4*xy2*y3 + F5*x1*xx4*xy3*y2 + F5*x2*xx1*xy3*y4 - F5*x2*xx1*xy4*y3 - F5*x2*xx3*xy1*y4 + F5*x2*xx3*xy4*y1 + F5*x2*xx4*xy1*y3 - F5*x2*xx4*xy3*y1 - F5*x3*xx1*xy2*y4 + F5*x3*xx1*xy4*y2 + F5*x3*xx2*xy1*y4 - F5*x3*xx2*xy4*y1 - F5*x3*xx4*xy1*y2 + F5*x3*xx4*xy2*y1 + F5*x4*xx1*xy2*y3 - F5*x4*xx1*xy3*y2 - F5*x4*xx2*xy1*y3 + F5*x4*xx2*xy3*y1 + F5*x4*xx3*xy1*y2 - F5*x4*xx3*xy2*y1)

                #Dxy
                #F0
                Dxy[j + i*n_points,j + i*n_points] = -(1/A)*(F0*x1*xx2*y3*yy4 - F0*x1*xx2*y4*yy3 - F0*x1*xx3*y2*yy4 + F0*x1*xx3*y4*yy2 + F0*x1*xx4*y2*yy3 - F0*x1*xx4*y3*yy2 - F0*x2*xx1*y3*yy4 + F0*x2*xx1*y4*yy3 + F0*x2*xx3*y1*yy4 - F0*x2*xx3*y4*yy1 - F0*x2*xx4*y1*yy3 + F0*x2*xx4*y3*yy1 + F0*x3*xx1*y2*yy4 - F0*x3*xx1*y4*yy2 - F0*x3*xx2*y1*yy4 + F0*x3*xx2*y4*yy1 + F0*x3*xx4*y1*yy2 - F0*x3*xx4*y2*yy1 - F0*x4*xx1*y2*yy3 + F0*x4*xx1*y3*yy2 + F0*x4*xx2*y1*yy3 - F0*x4*xx2*y3*yy1 - F0*x4*xx3*y1*yy2 + F0*x4*xx3*y2*yy1 - F0*x1*xx2*y3*yy5 + F0*x1*xx2*y5*yy3 + F0*x1*xx3*y2*yy5 - F0*x1*xx3*y5*yy2 - F0*x1*xx5*y2*yy3 + F0*x1*xx5*y3*yy2 + F0*x2*xx1*y3*yy5 - F0*x2*xx1*y5*yy3 - F0*x2*xx3*y1*yy5 + F0*x2*xx3*y5*yy1 + F0*x2*xx5*y1*yy3 - F0*x2*xx5*y3*yy1 - F0*x3*xx1*y2*yy5 + F0*x3*xx1*y5*yy2 + F0*x3*xx2*y1*yy5 - F0*x3*xx2*y5*yy1 - F0*x3*xx5*y1*yy2 + F0*x3*xx5*y2*yy1 + F0*x5*xx1*y2*yy3 - F0*x5*xx1*y3*yy2 - F0*x5*xx2*y1*yy3 + F0*x5*xx2*y3*yy1 + F0*x5*xx3*y1*yy2 - F0*x5*xx3*y2*yy1 + F0*x1*xx2*y4*yy5 - F0*x1*xx2*y5*yy4 - F0*x1*xx4*y2*yy5 + F0*x1*xx4*y5*yy2 + F0*x1*xx5*y2*yy4 - F0*x1*xx5*y4*yy2 - F0*x2*xx1*y4*yy5 + F0*x2*xx1*y5*yy4 + F0*x2*xx4*y1*yy5 - F0*x2*xx4*y5*yy1 - F0*x2*xx5*y1*yy4 + F0*x2*xx5*y4*yy1 + F0*x4*xx1*y2*yy5 - F0*x4*xx1*y5*yy2 - F0*x4*xx2*y1*yy5 + F0*x4*xx2*y5*yy1 + F0*x4*xx5*y1*yy2 - F0*x4*xx5*y2*yy1 - F0*x5*xx1*y2*yy4 + F0*x5*xx1*y4*yy2 + F0*x5*xx2*y1*yy4 - F0*x5*xx2*y4*yy1 - F0*x5*xx4*y1*yy2 + F0*x5*xx4*y2*yy1 - F0*x1*xx3*y4*yy5 + F0*x1*xx3*y5*yy4 + F0*x1*xx4*y3*yy5 - F0*x1*xx4*y5*yy3 - F0*x1*xx5*y3*yy4 + F0*x1*xx5*y4*yy3 + F0*x3*xx1*y4*yy5 - F0*x3*xx1*y5*yy4 - F0*x3*xx4*y1*yy5 + F0*x3*xx4*y5*yy1 + F0*x3*xx5*y1*yy4 - F0*x3*xx5*y4*yy1 - F0*x4*xx1*y3*yy5 + F0*x4*xx1*y5*yy3 + F0*x4*xx3*y1*yy5 - F0*x4*xx3*y5*yy1 - F0*x4*xx5*y1*yy3 + F0*x4*xx5*y3*yy1 + F0*x5*xx1*y3*yy4 - F0*x5*xx1*y4*yy3 - F0*x5*xx3*y1*yy4 + F0*x5*xx3*y4*yy1 + F0*x5*xx4*y1*yy3 - F0*x5*xx4*y3*yy1 + F0*x2*xx3*y4*yy5 - F0*x2*xx3*y5*yy4 - F0*x2*xx4*y3*yy5 + F0*x2*xx4*y5*yy3 + F0*x2*xx5*y3*yy4 - F0*x2*xx5*y4*yy3 - F0*x3*xx2*y4*yy5 + F0*x3*xx2*y5*yy4 + F0*x3*xx4*y2*yy5 - F0*x3*xx4*y5*yy2 - F0*x3*xx5*y2*yy4 + F0*x3*xx5*y4*yy2 + F0*x4*xx2*y3*yy5 - F0*x4*xx2*y5*yy3 - F0*x4*xx3*y2*yy5 + F0*x4*xx3*y5*yy2 + F0*x4*xx5*y2*yy3 - F0*x4*xx5*y3*yy2 - F0*x5*xx2*y3*yy4 + F0*x5*xx2*y4*yy3 + F0*x5*xx3*y2*yy4 - F0*x5*xx3*y4*yy2 - F0*x5*xx4*y2*yy3 + F0*x5*xx4*y3*yy2) 
                #F1
                Dxy[j + i*n_points,j - 2 + (i-1)*n_points] = -(1/A)*(- F1*x2*xx3*y4*yy5 + F1*x2*xx3*y5*yy4 + F1*x2*xx4*y3*yy5 - F1*x2*xx4*y5*yy3 - F1*x2*xx5*y3*yy4 + F1*x2*xx5*y4*yy3 + F1*x3*xx2*y4*yy5 - F1*x3*xx2*y5*yy4 - F1*x3*xx4*y2*yy5 + F1*x3*xx4*y5*yy2 + F1*x3*xx5*y2*yy4 - F1*x3*xx5*y4*yy2 - F1*x4*xx2*y3*yy5 + F1*x4*xx2*y5*yy3 + F1*x4*xx3*y2*yy5 - F1*x4*xx3*y5*yy2 - F1*x4*xx5*y2*yy3 + F1*x4*xx5*y3*yy2 + F1*x5*xx2*y3*yy4 - F1*x5*xx2*y4*yy3 - F1*x5*xx3*y2*yy4 + F1*x5*xx3*y4*yy2 + F1*x5*xx4*y2*yy3 - F1*x5*xx4*y3*yy2)
                #F2
                Dxy[j + i*n_points,j - 1 + (i-1)*n_points] = -(1/A)*(+ F2*x1*xx3*y4*yy5 - F2*x1*xx3*y5*yy4 - F2*x1*xx4*y3*yy5 + F2*x1*xx4*y5*yy3 + F2*x1*xx5*y3*yy4 - F2*x1*xx5*y4*yy3 - F2*x3*xx1*y4*yy5 + F2*x3*xx1*y5*yy4 + F2*x3*xx4*y1*yy5 - F2*x3*xx4*y5*yy1 - F2*x3*xx5*y1*yy4 + F2*x3*xx5*y4*yy1 + F2*x4*xx1*y3*yy5 - F2*x4*xx1*y5*yy3 - F2*x4*xx3*y1*yy5 + F2*x4*xx3*y5*yy1 + F2*x4*xx5*y1*yy3 - F2*x4*xx5*y3*yy1 - F2*x5*xx1*y3*yy4 + F2*x5*xx1*y4*yy3 + F2*x5*xx3*y1*yy4 - F2*x5*xx3*y4*yy1 - F2*x5*xx4*y1*yy3 + F2*x5*xx4*y3*yy1)
                #F3
                Dxy[j + i*n_points,j + (i-1)*n_points] = -(1/A)*(- F3*x1*xx2*y4*yy5 + F3*x1*xx2*y5*yy4 + F3*x1*xx4*y2*yy5 - F3*x1*xx4*y5*yy2 - F3*x1*xx5*y2*yy4 + F3*x1*xx5*y4*yy2 + F3*x2*xx1*y4*yy5 - F3*x2*xx1*y5*yy4 - F3*x2*xx4*y1*yy5 + F3*x2*xx4*y5*yy1 + F3*x2*xx5*y1*yy4 - F3*x2*xx5*y4*yy1 - F3*x4*xx1*y2*yy5 + F3*x4*xx1*y5*yy2 + F3*x4*xx2*y1*yy5 - F3*x4*xx2*y5*yy1 - F3*x4*xx5*y1*yy2 + F3*x4*xx5*y2*yy1 + F3*x5*xx1*y2*yy4 - F3*x5*xx1*y4*yy2 - F3*x5*xx2*y1*yy4 + F3*x5*xx2*y4*yy1 + F3*x5*xx4*y1*yy2 - F3*x5*xx4*y2*yy1)
                #F4
                Dxy[j + i*n_points,j - 2 + i*n_points] = -(1/A)*(+ F4*x1*xx2*y3*yy5 - F4*x1*xx2*y5*yy3 - F4*x1*xx3*y2*yy5 + F4*x1*xx3*y5*yy2 + F4*x1*xx5*y2*yy3 - F4*x1*xx5*y3*yy2 - F4*x2*xx1*y3*yy5 + F4*x2*xx1*y5*yy3 + F4*x2*xx3*y1*yy5 - F4*x2*xx3*y5*yy1 - F4*x2*xx5*y1*yy3 + F4*x2*xx5*y3*yy1 + F4*x3*xx1*y2*yy5 - F4*x3*xx1*y5*yy2 - F4*x3*xx2*y1*yy5 + F4*x3*xx2*y5*yy1 + F4*x3*xx5*y1*yy2 - F4*x3*xx5*y2*yy1 - F4*x5*xx1*y2*yy3 + F4*x5*xx1*y3*yy2 + F4*x5*xx2*y1*yy3 - F4*x5*xx2*y3*yy1 - F4*x5*xx3*y1*yy2 + F4*x5*xx3*y2*yy1)
                #F5
                Dxy[j + i*n_points,j - 1 + i*n_points] = -(1/A)*(- F5*x1*xx2*y3*yy4 + F5*x1*xx2*y4*yy3 + F5*x1*xx3*y2*yy4 - F5*x1*xx3*y4*yy2 - F5*x1*xx4*y2*yy3 + F5*x1*xx4*y3*yy2 + F5*x2*xx1*y3*yy4 - F5*x2*xx1*y4*yy3 - F5*x2*xx3*y1*yy4 + F5*x2*xx3*y4*yy1 + F5*x2*xx4*y1*yy3 - F5*x2*xx4*y3*yy1 - F5*x3*xx1*y2*yy4 + F5*x3*xx1*y4*yy2 + F5*x3*xx2*y1*yy4 - F5*x3*xx2*y4*yy1 - F5*x3*xx4*y1*yy2 + F5*x3*xx4*y2*yy1 + F5*x4*xx1*y2*yy3 - F5*x4*xx1*y3*yy2 - F5*x4*xx2*y1*yy3 + F5*x4*xx2*y3*yy1 + F5*x4*xx3*y1*yy2 - F5*x4*xx3*y2*yy1)


    # print("Dxx", Dxx)
    # print("Dyy", Dyy)
    # print("Dxy", Dxy)

    return Dx, Dy, Dxx, Dyy, Dxy   

def BoundaryConditions(x,y,n_points,n_R,alpha,Dxx,Dyy):

    X = np.zeros((n_points*n_R), dtype=np.double)
    b = np.zeros((n_points*n_R), dtype=np.double)

    #Dirichlet boundary
    j = 0
    i = 0
    X[j + i *n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    

    # print(x[i[0],j[0]])
    # j[1] = int(n_points/2) + 1
    # i[1] = n_R - 1
    # X[j[1] + i*n_points] = x[i[1],j[1]]*cos(alpha) + y[i[1],j[1]]*sin(alpha) 
    # j[2] = int(n_points/2) - 1 
    # i[2] = n_R - 1
    # X[j[2] + i*n_points] = x[i[2],j[2]]*cos(alpha) + y[i[2],j[2]]*sin(alpha) 
    # j[3] = int(n_points/2)
    # i[3] = n_R - 2
    # X[j[3] + i[3]*n_points] = x[i[3],j[3]]*cos(alpha) + y[i[3],j[3]]*sin(alpha) 

    # j = int(n_points/2) + 1
    # i = n_R - 2
    # X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 

    BCC_wall = np.zeros((n_points,n_R*n_points), dtype=np.double)

    for l in range(n_points): #Wall condition
        #l = n_points-1 - l #Inverse path so as to no modify derivatives

        if l < int(n_points/2): #Wall tangency 
            x1 = x[1,l]-x[0,l]
            x2 = x[0,l+1]-x[0,l]
            y1 = y[1,l]-y[0,l]
            y2 = y[0,l+1]-y[0,l]
            BCC_wall[l,l] = -1 
            BCC_wall[l,l + 1*n_points] = (x2 + y2*y2/x2)/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))
            BCC_wall[l,l + 1 + 0*n_points] = - (x1 + y1*y2/x2)/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))

            #X[l] = (X[l + 1*n_points]*(x2 + y2*y2/x2) - X[l + 1 + 0*n_points]*(x1 + y1*y2/x2))/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))

        if l > int(n_points/2) + 1:
            x1 = x[1,l]-x[0,l]
            x2 = x[0,l-1]-x[0,l]
            y1 = y[1,l]-y[0,l]
            y2 = y[0,l-1]-y[0,l]
            BCC_wall[l,l] = -1 
            BCC_wall[l,l + 1*n_points] = (x2 + y2*y2/x2)/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))
            BCC_wall[l,l - 1 + 0*n_points] = - (x1 + y1*y2/x2)/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))

            #X[l] = (X[l + 1*n_points]*(x2 + y2*y2/x2) - X[l + 1 + 0*n_points]*(x1 + y1*y2/x2))/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))

    # Kutta condition//Carefull Allow circulation
    BCC_wall[int(n_points/2) + 0, int(n_points/2) + 0*n_points] = 1
    BCC_wall[int(n_points/2) + 0, int(n_points/2) + 1*n_points] = -1
    BCC_wall[int(n_points/2) + 1, int(n_points/2) + 1 + 0*n_points] = 1
    BCC_wall[int(n_points/2) + 1, int(n_points/2) + 1 + 1*n_points] = -1

    #Continuos flow in discontinuity:
    BCC_cont = np.zeros((2*(n_R-1),n_R*n_points), dtype=np.double)
 
    for i in range(1,n_R): 
        x1u = x[i-1,n_points-1]-x[i,n_points-1]
        x2u = x[i,n_points-2]-x[i,n_points-1]
        y1u = y[i-1,n_points-1]-y[i,n_points-1]
        y2u = y[i,n_points-2]-y[i,n_points-1]

        x1d = x[i-1,0]-x[i,0]
        x2d = x[i,1]-x[i,0]
        y1d = y[i-1,0]-y[i,0]
        y2d = y[i,1]-y[i,0]

        #Dxu = Dxd
        BCC_cont[i-1,0 + i*n_points] = (-1 + y1d/y2d)/(x1d-x2d*y1d/y2d)
        BCC_cont[i-1,0 + (i-1)*n_points] = 1/(x1d-x2d*y1d/y2d)
        BCC_cont[i-1,1 + i*n_points]  = -y1d/((x1d-x2d*y1d/y2d)*y2d)

        BCC_cont[i-1,n_points - 1 + i*n_points] = -(-1 + y1u/y2u)/(x1u-x2u*y1u/y2u)
        BCC_cont[i-1,n_points - 1 + (i-1)*n_points] = -1/(x1u-x2u*y1u/y2u)
        BCC_cont[i-1,n_points - 2 + i*n_points]  = +y1u/((x1u-x2u*y1u/y2u)*y2u)

        #Dyu = Dyd
        BCC_cont[(i-1) + n_R-1,0 + i*n_points] = (-1 + x1d/x2d)/(y1d-y2d*x1d/x2d)
        BCC_cont[(i-1) + n_R-1,0 + (i-1)*n_points] = 1/(y1d-y2d*x1d/x2d)
        BCC_cont[(i-1) + n_R-1,1 + i*n_points]  = -x1d/((y1d-y2d*x1d/x2d)*x2d)

        BCC_cont[(i-1) + n_R-1,n_points - 1 + i*n_points] = -(-1 + x1u/x2u)/(y1u-y2u*x1u/x2u)
        BCC_cont[(i-1) + n_R-1,n_points - 1 + (i-1)*n_points] = -1/(y1u-y2u*x1u/x2u)
        BCC_cont[(i-1) + n_R-1,n_points - 2 + i*n_points]  = x1u/((y1u-y2u*x1u/x2u)*x2u)

    Dx_y = Dxx + Dyy

    #Eliminate derivatives in wall and introduce tangency
    for j in range(n_points):
        Dx_y[j + 0*n_points,:] = BCC_wall[j,:]

    #Continuity in discontinous step
    for i in range(1,n_R):
        Dx_y[0 + i*n_points,:] = BCC_cont[i-1,:]
        Dx_y[n_points - 1 + i*n_points,:] = BCC_cont[i-1 + n_R-1,:]

    #Exterior boundary
    BCC_ext = np.zeros((n_points-1,n_R*n_points), dtype=np.double)        

    for j in range(n_points-1):
        if j < n_points - 1:
            x1 = x[n_R-2,j]-x[n_R-1,j]
            x2 = x[n_R-1,j + 1]-x[n_R-1,j]
            y1 = y[n_R-2,j]-y[n_R-1,j]
            y2 = y[n_R-1,j + 1]-y[n_R-1,j]

            #Dx
            BCC_ext[j,j + (n_R - 1)*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
            BCC_ext[j,j + (n_R - 2)*n_points] = 1/(x1-x2*y1/y2)
            BCC_ext[j,j + 1 + (n_R - 1)*n_points]  = -y1/((x1-x2*y1/y2)*y2)

        if j == (n_points):
            x1 = x[n_R-2,j]-x[n_R-1,j]
            x2 = x[n_R-2,j - 1]-x[n_R-1,j]
            y1 = y[n_R-2,j]-y[n_R-1,j]
            y2 = y[n_R-2,j - 1]-y[n_R-1,j]

            #Dx
            BCC_ext[j,j + (n_R - 1)*n_points] = (-1 + y1/y2)/(x1-x2*y1/y2)
            BCC_ext[j,j + (n_R - 2)*n_points] = 1/(x1-x2*y1/y2)
            BCC_ext[j,j - 1 + (n_R - 1)*n_points]  = -y1/((x1-x2*y1/y2)*y2)


    
    #Exterior boundary
    for j in range(n_points-1):
        #Dx
        Dx_y[j + (n_R - 1)*n_points,:] = BCC_ext[j,:]

    b = np.dot(Dx_y,X)

    for j in range(n_points-1):
        b[j + (n_R - 1)*n_points] = cos(alpha)

    return X, Dx_y, b


def InitialValue(x,y,n_R,n_points,alpha):
    X = np.zeros((n_R*n_points), dtype=np.double)

    for i in range(n_R): #Exterior boundary
        for j in range(n_points):
            X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha)
    
    return X

def DiferentialEquation(x,y,n_points,n_R,alpha,delta,lamda,n):

    Dx, Dy, Dxx, Dyy, Dxy = Derivatives(x,y,n_points,n_R)

    # print("Rank Dx", matrix_rank(Dx))
    # print("Rank Dy", matrix_rank(Dy))
    # print("Rank Dxx", matrix_rank(Dxx))
    # print("Rank Dyy", matrix_rank(Dyy))
    # print("Rank Dxy", matrix_rank(Dxy))

    #X = InitialValue(x,y,n_R,n_points,alpha)

    X, Dx_y, b= BoundaryConditions(x,y,n_points,n_R,alpha,Dxx,Dyy)

    #Imposing Dirichlet Boundary Condition
    Dx_y_red_ = np.zeros((n_R*n_points,n_R*n_points - 1),dtype=np.double)
    Dx_y_red = np.zeros((n_R*n_points-1,n_R*n_points - 1),dtype=np.double)
    X_red = np.zeros((n_R*n_points - 1),dtype=np.double)
    b_red = np.zeros((n_R*n_points - 1),dtype=np.double)

    Dx_y_red_ = Dx_y[:,1:]
    Dx_y_red = Dx_y_red_ [1:,:]
    b_red =  b[1:]

    #print("Rank Dx_y", matrix_rank(Dx_y_red))

    # _, inds = sympy.Matrix(Dx_y).T.rref()   # to check the rows you need to transpose!
    # print(inds)

    # X_red = solve(Dx_y_red, b_red)
    # X[1:] = X_red
    X = solve(Dx_y, b)

    #Analytical solution
    contour = np.zeros((n_R, n_points), dtype=np.double)

    for i  in range(0,n_R):
        for j  in range(0,n_points):
            val = InverseTransform(complex(x[i,j],y[i,j]), delta, lamda, n = 2)
            contour[i,j] = Potencial(val, alpha, delta, lamda, n = 2)

    # fig = plt.figure(figsize=(40,40))
    # fig.add_subplot(111)
    # plt.title('Analytic solution')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.contour(x, y, contour, 200, cmap='gist_gray')
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure(figsize=(40,40))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(-20, 20)
    # ax.set_ylim3d(-20, 20)
    # ax.set_zlim3d(-20, 20)
    # ax.plot_trisurf(x.flatten(),y.flatten(),contour.flatten(),cmap='Greys', edgecolor='none')
    # ax.scatter(x.flatten(),y.flatten(),contour.flatten(),s=1, c='Black') 
    # ax.set_title("Analytic potential")
    # plt.show()

    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-1, 1)
    ax.plot_trisurf(x.flatten(),y.flatten(),X,cmap='Greys', edgecolor='none')
    ax.scatter(x.flatten(),y.flatten(),X,s=1, c='Black') 
    ax.set_title("Compressible potential")
    plt.show()
    
    X_B = np.zeros((n_R,n_points), dtype=np.double)
    for m in range(n_R):
        for n in range(n_points):
            X_B[m,n] = X[n + m*n_points]
    fig.add_subplot(211)
    plt.title('Potential contour lines')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(x, y, X_B, 200, cmap='gist_gray')
    plt.colorbar()
    plt.show()

    return