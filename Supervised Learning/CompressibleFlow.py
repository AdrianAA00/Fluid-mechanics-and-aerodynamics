from cmath import pi
from xml.etree.ElementInclude import XINCLUDE
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
import pylab
from math import atan, cos, pi, sin
from numpy.linalg import *
from numpy import transpose

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

def BoundaryConditions(X,x,y,n_points,n_R,alpha,Dx,Dy):

    #Exterior boundary
    j = int(n_points/2)
    i = n_R - 1
    X[j+ i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) + 1
    i = n_R - 1
    X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) - 1
    i = n_R - 1
    j = int(n_points/2) - 2
    i = n_R - 1
    X[j+ i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) + 2
    i = n_R - 1
    X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) - 3
    i = n_R - 1
    X[j+ i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) + 3
    i = n_R - 1
    X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha) 
    j = int(n_points/2) - 4
    i = n_R - 1


    for l in range(n_points): #Wall condition
        #l = n_points-1 - l #Inverse path so as to no modify derivatives

        if 0 < l < n_points - 1: #Wall tangency 
            x1 = x[1,l]-x[0,l]
            x2 = x[0,l+1]-x[0,l]
            y1 = y[1,l]-y[0,l]
            y2 = y[0,l+1]-y[0,l]

            X[l] = (X[l + 1*n_points]*(x2 + y2*y2/x2) - X[l + 1 + 0*n_points]*(x1 + y1*y2/x2))/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))

        if  l == 0: #Wall tangency
            l = n_points-1
            x1 = x[1,l]-x[0,l]
            x2 = x[0,l-1]-x[0,l]
            y1 = y[1,l]-y[0,l]
            y2 = y[0,l-1]-y[0,l]

            X[l] = (X[l + 1*n_points]*(x2 + y2*y2/x2) - X[l - 1 + 0*n_points]*(x1 + y1*y2/x2))/((x2 + y2*y2/x2) - (x1 + y1*y2/x2))
        
    devX = np.dot(Dx,X)
    devY = np.dot(Dy,X)
    
    # Kutta condition//Carefull Allow circulation
    X[int(n_points/2) + 1] = X[int(n_points/2)] 
    X[int(n_points/2) + 1*n_points] = X[int(n_points/2)] 
    X[int(n_points/2) + 1 + 1*n_points] = X[int(n_points/2)] 
    X[int(n_points/2) + 2] = X[int(n_points/2)] 
    X[int(n_points/2) - 1] = X[1] 

    for k in range(n_R-1): #Careful with 0 point
        #k = n_R-2 - k
        x1 = x[k,n_points - 2]-x[k+1,n_points - 1]
        x2 = x[k+1,n_points - 2]-x[k+1,n_points - 1]
        y1 = y[k,n_points - 2]-y[k+1,n_points - 1]
        y2 = y[k+1,n_points - 2]-y[k+1,n_points - 1]
        X[n_points - 1 + (k + 1)*n_points] = ((devY[0 + (k + 1)*n_points]/devX[0 + (k + 1)*n_points])*(X[n_points - 2 + k*n_points]/(x1-y1*x2/y2)-X[n_points - 2 + (k + 1)*n_points]*y1/(x1*y2-y1*x2)) - ((X[n_points - 2 + k*n_points]/(y1-x1*y2/x2)-X[n_points - 2 + (k + 1)*n_points]*x1/(y1*x2-x1*y2))))/((devY[0 + (k + 1)*n_points]/devX[0 + (k + 1)*n_points])*(1-y1/y2)/(x1-y1*x2/y2)+(-1+x1/x2)/(y1-x1*y2/x2))
    
    #print(X[n_points - 1 + (k + 1)*n_points])

    return X

def InitialValue(x,y,n_R,n_points,alpha):
    X = np.zeros((n_R*n_points), dtype=np.double)

    for i in range(n_R): #Exterior boundary
        for j in range(n_points):
            X[j + i*n_points] = x[i,j]*cos(alpha) + y[i,j]*sin(alpha)
    
    return X

def DiferentialEquation(x,y,n_points,n_R,alpha,delta,lamda,n):

    Dx, Dy, Dxx, Dyy, Dxy = Derivatives(x,y,n_points,n_R)

    print("Rank Dx", matrix_rank(Dx))
    print("Rank Dy", matrix_rank(Dy))
    print("Rank Dxx", matrix_rank(Dxx + Dyy))
    print("Rank Dyy", matrix_rank(Dyy))
    print("Rank Dxy", matrix_rank(Dxy))
    
    I = np.identity((n_R*n_points), dtype=np.double)
    X = InitialValue(x,y,n_R,n_points,alpha)

    # # ##Explicit
    # t_step = 0.001
    # max_iter = 10
    
    # for m in range(max_iter):
    #     X = X - np.dot(transpose(Dxx + Dyy),np.dot(Dxx + Dyy,X))*t_step
    #     X = BoundaryConditions(X,x,y,n_points,n_R,alpha,Dx,Dy)

    #     err = np.dot(Dxx + Dyy,X)
    #     print("Error iter", m, ":", np.linalg.norm((err[1:]))/(n_R*n_points))
 
    # print(err)
            
    # #Implicit
    t_step = 1000
    max_iter = 1000
    #ab = np.zeros((max_iter), dtype=np.double)
    #o = np.zeros((max_iter), dtype=np.integer)
    
    #D_inv = np.linalg.pinv(Dxx + Dyy)

    for m in range(max_iter):
        X = solve(I + np.dot(transpose(Dxx + Dyy),Dxx + Dyy)*t_step,X)
        X = BoundaryConditions(X,x,y,n_points,n_R,alpha,Dx,Dy)

        err = np.dot(Dxx + Dyy,X)
        #ab[m] = np.linalg.norm((err[1:]))/(n_R*n_points)
        #o[m] = m
        print("Error iter", m, ":", np.linalg.norm((err[1:]))/(n_R*n_points))
        
    # #plt.plot(o,ab)
    # #plt.show
 
    # print(err)

    # Implicit AM4
    # t_step1 = 0.1
    # t_step2 = 0.1
    # max_iter = 1000
    # X_AM = np.zeros((5,n_R*n_points), dtype=np.double) #Store previous values

    # for m in range(max_iter):
    #     if m < 4: #Storing previous values
    #         X = solve(I + np.dot(transpose(Dxx + Dyy),Dxx + Dyy)*t_step1,X)
    #         X_AM[m,:] = BoundaryConditions(X,x,y,n_points,n_R,alpha,Dx,Dy)

    #     else:
    #         X_AM[m,:] = solve(I + np.dot(transpose(Dxx + Dyy),Dxx + Dyy)*(251/720)*t_step2,(t_step2/720)*(646*))
        
    #     err = np.dot(Dxx + Dyy,X)
    #     #ab[m] = np.linalg.norm((err[1:]))/(n_R*n_points)
    #     #o[m] = m
    #     print("Error iter", m, ":", np.linalg.norm((err[1:]))/(n_R*n_points))
        

    # Modified algorithm 
    # t_step = 0.0000001
    # max_iter = 5
    # for m in range(max_iter):
    #     X = BoundaryConditions(X,x,y,n_points,n_R,alpha,Dx,Dy)

    #     for i in range(1,n_R):
    #         for j in range(n_points):
    #             v = np.dot(Dxx + Dyy,X)
    #             X[j + i*n_points] = X[j + i*n_points] - v[j + i*n_points]*t_step/(Dxx[j + i*n_points,j + i*n_points] + Dyy[j + i*n_points,j + i*n_points])
    #     for i in range(n_R):
    #         i = n_R - 1 - i
    #         for j in range(n_points):
    #             j = n_points - 1 - i
    #             v = np.dot(Dxx + Dyy,X)
    #             X[j + i*n_points] = X[j + i*n_points] - v[j + i*n_points]*t_step/(Dxx[j + i*n_points,j + i*n_points] + Dyy[j + i*n_points,j + i*n_points])

    #     err = np.dot(Dxx + Dyy,X)
    #     print("Error iter", m, ":", np.linalg.norm((err[1:]))/(n_R*n_points))
            
    # print(err)  

    ##Cheking boundary conditions
    # print(np.divide(np.dot(Dy,X),np.dot(Dx,X)))
    # tan = np.zeros((n_points), dtype=np.double)
    # for i in range(n_points):
    #     tan[i] = (y[0,i] - y[0,i-1])/(x[0,i] - x[0,i-1])
    # print(tan)

    #Analytical solution
    # contour = np.zeros((n_R, n_points), dtype=np.double)

    # for i  in range(0,n_R):
    #     for j  in range(0,n_points):

    #         val = InverseTransform(complex(x[i,j],y[i,j]), delta, lamda, n = 2)
    #         contour[i,j] = Potencial(val, alpha, delta, lamda, n = 2)
            
    # print(contour)
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
    ax.set_zlim3d(-2, 2)
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