import cmath
import math
import numpy as np

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

    R = math.sqrt(delta**2 + (1 + lamda)**2);        # Circunference Radio (Passing through (1,0) )
    theta0 = math.asin(- delta/R)                    # Angle for Yukovski Edge Condition
    angle = 2 * (math.pi) / (n_points + 1)
    Z_ = np.zeros((n_points,1), dtype=np.complex64)
    
    #Cimcumference
    for i in range(0,n_points):
        x = R * math.cos(angle * (i + 1) + theta0) - lamda
        y = R * math.sin(angle * (i + 1) + theta0) + delta
        z = complex(x,y)

        z2 = Transform(z, delta, lamda, n)
        Z_[i] = np.array([z2])

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
    F = np.imag(F)

    return F


def InverseTransform(x, delta, lamda, n):

    F = ( (x-n)**(1/n) + (x+n)**(1/n) ) / ( (x+n)**(1/n) - (x-n)**(1/n) )

    return F

