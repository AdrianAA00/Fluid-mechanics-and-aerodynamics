import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
import pylab
from sympy import N
from math import pi

def Potential_derivative(x, alpha, delta, lamda, n):

    i = complex(0,1)                  # Complex constant
    x0 = complex(-lamda, delta)       # Circunference origin
    R = math.sqrt(delta**2 + (1 + lamda)**2)        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R) # Angle for Yukovski Edge Condition

    Df_Dchi = cmath.exp(-i*alpha) - R**2 * cmath.exp(i*alpha) / ((x-x0)**2) + 2*R*i*math.sin(alpha - theta0)/(x-x0)
    Dchi_Dz = (((x**2 - 1)/(4*n**2))*( (x+1)**n - (x-1)**n )**2) / ((x+1)**n * (x-1)**n)

    return Df_Dchi*Dchi_Dz

def Transform(x, delta, lamda, n):
    
    F = n * ( (x+1)**n + (x-1)**n ) / ( (x+1)**n - (x-1)**n )

    return F

def Airfoil_boundary(delta, lamda, n, n_points):

    R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / n_points
    Z_ = np.zeros((n_points + 1,1), dtype=np.complex64)
    
    #Cimcumference
    for i in range(0,n_points+1):
        x = R * math.cos(angle * i + theta0) - lamda
        y = R * math.sin(angle * i + theta0) + delta
        z = complex(x,y)

        z2 = Transform(z, delta, lamda, n)
        Z_[i] = np.array([z2])

    return Z_

def Airfoil_boundary_Training(delta, lamda, n, n_points):

    R = math.sqrt(delta**2 + (1 + lamda)**2)         # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / (n_points + 1)
    Z_ = np.zeros((n_points,1), dtype=np.complex64)
    
    #Cimcumference, finding radius for each angle
    for i in range(0,n_points):
        x = R * math.cos(angle * (i + 1) + theta0) - lamda
        y = R * math.sin(angle * (i + 1) + theta0) + delta
        z = complex(x,y)

        z2 = Transform(z, delta, lamda, n)
        Z_[i] = np.array([z2])

    # K_points_c = 1000
    # R = math.sqrt(delta**2 + (1 + lamda)**2)         # Circunference Radio (Passing through (1,0) )
    # theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    # angle_K = 2 * (math.pi) / (K_points_c + 1)
    # Z_K = np.zeros((K_points_c,1), dtype=np.complex64)
    # Arg_Z_K = np.zeros((K_points_c,1), dtype=np.complex64)
    
    # #Cimcumference, finding radius for each angle
    # for i in range(0,K_points_c):
    #     x = R * math.cos(angle_K * (i + 1) + theta0) - lamda
    #     y = R * math.sin(angle_K * (i + 1) + theta0) + delta
    #     z = complex(x,y)

    #     z2 = Transform(z, delta, lamda, n)
    #     Z_K[i] = np.array([z2])

    # Arg_Z_K = np.angle(Z_K)                             # Arguments of airfoil
    # #print("Ang_Z = ", Arg_Z_K[499:999])

    # angle = 2 * (math.pi) / (n_points + 1)
    # Z_ = np.zeros((n_points,1), dtype=np.complex64)
    # R_ = np.zeros((n_points,1), dtype=np.complex64)

    # #Cimcumference, finding radius for each angle
    # for i in range(0,n_points):
    #     for j in range(0,K_points_c-1):
    #         if Arg_Z_K[j] >= 0:
    #             if Arg_Z_K[j] <= angle * (i + 1):
    #                 if Arg_Z_K[j + 1] > angle * (i + 1):
    #                     R_[i] = np.abs(Z_K[j]) 
    #                     print(Z_K[j])
    #                     print(R_[i])
    #         else: 
    #             if Arg_Z_K[j] <= - 2*math.pi + angle * (i + 1):
    #                 if Arg_Z_K[j + 1] > - 2*math.pi + angle * (i + 1):
    #                     R_[i] = np.abs(Z_K[j])  
    #                     print(Z_K[j])
    #                     print(R_[i])
            
    #print("R_ = ", R_)     

    return Z_

def Pressure_field_boundary(alpha, delta, lamda, n, n_points):

    R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / n_points
    Cp_ = np.zeros((n_points,1), dtype=np.double)

    for i in range(1,n_points):
        x = R * math.cos(angle * i + theta0) - lamda
        y = R * math.sin(angle * i + theta0) + delta
        z = complex(x,y)
        
        v = Potential_derivative(z, alpha, delta, lamda, n)
    
        cp = 1 - abs(v * v)
        Cp_[i] = cp

    return Cp_

def Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points):

    R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / (n_points + 1)
    Cp_ = np.zeros((n_points,1), dtype=np.double)

    for i in range(n_points):
        x = R * math.cos(angle * (i + 1) + theta0) - lamda
        y = R * math.sin(angle * (i + 1) + theta0) + delta
        z = complex(x,y)
        
        v = Potential_derivative(z, alpha, delta, lamda, n)
    
        cp = 1 - abs(v * v)
        Cp_[i] = cp

    return Cp_

def Potencial(x, alpha, delta, lamda, n):
    
    i = complex(0,1)                  # Complex constant
    x0 = complex(-lamda, delta)       # Circunference origin
    R = math.sqrt(delta**2 + (1 + lamda)**2)        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R) # Angle for Yukovski Edge Condition

    F = cmath.exp(-i*alpha)*(x-x0) + R**2 * cmath.exp(i*alpha) / (x-x0) + 2*R*i*math.sin(alpha - theta0)*cmath.log(x-x0)
    F = np.real(F)

    return F

def InverseTransform(x, delta, lamda, n):
    F = ( (x-n)**(1/n) + (x+n)**(1/n) ) / ( (x+n)**(1/n) - (x-n)**(1/n) )
    #F = (2*x + (x+n)**(1/n)*(x-n)**(1-1/n) + (x-n)**(1/n)*(x+n)**(1-1/n))/(2*n - (x-n)**(1/n)*(x+n)**(1-1/n) + (x+n)**(1/n)*(x-n)**(1-1/n))
    return F

def InverseTransformD(x, delta, lamda, n):
    F = ( (x-n)**(1/n) + (x+n)**(1/n) ) / ( (x+n)**(1/n) - (x-n)**(1/n) )
    #F = (2*x + (x+n)**(1/n)*(x-n)**(1-1/n) + (x-n)**(1/n)*(x+n)**(1-1/n))/(2*n - (x-n)**(1/n)*(x+n)**(1-1/n) + (x+n)**(1/n)*(x-n)**(1-1/n))

    if np.imag(x) > 0:
        F = ( - (x-n)**(1/n) + (x+n)**(1/n) ) / ( + (x+n)**(1/n) + (x-n)**(1/n) )

    return F

def Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points):
    Z_ = np.zeros((n_points,2), dtype=np.complex64)
    
    K_points_c = 200
    R = math.sqrt(delta**2 + (1 + lamda)**2)         # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle_K = 2 * (math.pi) / (K_points_c + 1)
    Z_K = np.zeros((K_points_c), dtype=np.complex64)
  
    #Cimcumference, finding radius for each angle
    for i in range(0,K_points_c):
        x = R * math.cos(angle_K * (i + 1) + theta0) - lamda
        y = R * math.sin(angle_K * (i + 1) + theta0) + delta
        z = complex(x,y)

        z2 = Transform(z, delta, lamda, n)
        Z_K[i] = np.array([z2])

    x_min = min(np.real(Z_K))
    incre_x = (2-x_min)/(n_points+1)

    E = np.zeros((n_points), dtype=np.double)
    C = np.zeros((n_points), dtype=np.double)

    #Cimcumference, finding x
    i = int(0)
    for i in range(0,int(n_points)):
        for j in range(0,K_points_c-1):
            if  (2 - (i + 1)*incre_x) <= np.real(Z_K[j]):
                if (2 - (i + 1)*incre_x) > np.real(Z_K[j+1]):
                    Z_[i,0] = Z_K[j]
    
            if (x_min + (i + 1)*incre_x) >= np.real(Z_K[j]):
                if (x_min + (i + 1)*incre_x) < np.real(Z_K[j+1]):
                        Z_[n_points-1-i,1] = Z_K[j]
            
    #print("Z_ = ", Z_)  

    for i in range(0,int(n_points)):
        E[i] = np.imag(Z_[i,0]) - np.imag(Z_[i,1])
        C[i] = (np.imag(Z_[i,0]) + np.imag(Z_[i,1]))/2

    # print("E = ", E)  
    # print("C = ", C)  

    i = int(0)
    Cp = np.zeros((n_points), dtype=np.double)

    # extract real part
    x0 = [ele.real for ele in Z_[:,0]]
    # extract imaginary part
    y0 = [ele.imag for ele in Z_[:,0]]

    x1 = [ele.real for ele in Z_[:,1]]
    # extract imaginary part
    y1 = [ele.imag for ele in Z_[:,1]]

    # n_poi = 100
    # A = Airfoil_boundary(delta, lamda, n, n_poi);    

    # x = [ele.real for ele in A] # extract real part
    # y = [ele.imag for ele in A] # extract imaginary part

    # # plot the complex numbers
    # plt.scatter(x0, y0)
    # plt.scatter(x1, y1)
    # plt.plot(x, y)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylabel('y')
    # plt.xlabel('x')
    # plt.show()

    #x0 = np.zeros((n_points), dtype=np.double)
    #y0 = np.zeros((n_points), dtype=np.double)
    #x1 = np.zeros((n_points), dtype=np.double)
    #y1 = np.zeros((n_points), dtype=np.double)

    for i in range(0,n_points):
        x0[i] = np.real(InverseTransform(Z_[i,0], delta, lamda, n))
        y0[i] = np.imag(InverseTransform(Z_[i,0], delta, lamda, n))
        x1[i] = np.real(InverseTransformD(Z_[i,1], delta, lamda, n))
        y1[i] = np.imag(InverseTransformD(Z_[i,1], delta, lamda, n))

    # # plot the complex numbers
    # plt.scatter(x0, y0)
    # plt.scatter(x1, y1)
    # plt.plot(x0, y0)
    # plt.plot(x1, y1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylabel('Imaginary')
    # plt.xlabel('Real')
    # plt.show()

    for i in range(0,int(n_points)):
        v_e = Potential_derivative(complex(x0[i],y0[i]), alpha, delta, lamda, n)
        v_i = Potential_derivative(complex(x1[i],y1[i]), alpha, delta, lamda, n)
       
        if i < n_points-1:
            cp_e = (1 - abs(v_e * v_e))
            cp_i = (1 - abs(v_i * v_i))
        else:
            cp_e = (1 - abs(v_e * v_e))#*(4/n_points)/((4/n_points)**2+(E[i]+C[i]-E[i-1]-C[i-1])**2)**(1/2)
            cp_i = (1 - abs(v_i * v_i))#*(4/n_points)/((4/n_points)**2+(-E[i]+C[i]+E[i-1]-C[i-1])**2)**(1/2)
        

        Cp[i] = cp_i - cp_e

    # plt.plot(np.real(Z_[:,0]), Cp, color = "red", label = "Cp_global")
    # plt.scatter(np.real(Z_[:,0]), Cp, color = "black", label = "data")
    # plt.xlabel('x - axis')
    # plt.ylabel('cp -  axis')
    # plt.title('Cp')
    # plt.legend()
    # plt.show()

    #print("Cp = ", Cp)  
    
    return Cp, E, C


def Airfoil_R(R, delta, lamda, n, n_points):

    #R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / (n_points + 1)
    Z_ = np.zeros((n_points + 1,1), dtype=np.complex64)
    
    #Cimcumference
    for i in range(0,n_points+1):
        x = R * math.cos(angle * i + theta0 + angle/3 + pi) - lamda
        y = R * math.sin(angle * i + theta0 + angle/3 + pi) + delta
        z = complex(x,y)

        z2 = Transform(z, delta, lamda, n)
        Z_[i] = np.array([z2])

    return Z_

