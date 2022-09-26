import cmath
from cmath import pi
import math
import numpy as np
import matplotlib.pyplot as plt

from Potential import *

def TrainingData(alpha_N, delta_N, lamda_N, n_N, n_points, alpha_max ,alpha_min, delta_max, lamda_max, n_min) :

    count = int(0)
    maxcount = alpha_N * delta_N * lamda_N * n_N
    maxcount2 = int((alpha_N * delta_N * lamda_N * n_N)/4)

    Y = np.zeros((maxcount,n_points), dtype=np.double)
    X = np.zeros((maxcount,2*n_points), dtype=np.double)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min)
        for j in range(0,delta_N):
            delta = (j + 1) * delta_max/alpha_N
            for k in range(0,lamda_N):
                lamda = (k + 1) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min                                        
                   
                    temp1 = Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points)
                    temp2 = Airfoil_boundary_Training(delta, lamda, n, n_points)

                    for m in range(0,n_points):
                        Y[count,m] = temp1[m]                                     #Training[different airfoils, pressure in each n_points]
                        X[count,m] = np.real(temp2[m])                            #Z[different airfoils, complex coordinates n_points]
                        X[count,m + n_points] = np.imag(temp2[m]) 

                    count = count + 1

    # alpha = 10 
    # delta = 0.1
    # lamda = 0.1
    # n = 2
    # temp1 = Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points)
    # temp2 = Airfoil_boundary_Training(delta, lamda, n, n_points)

    # for m in range(0,n_points):
    #    Y[count,m] = temp1[m]                                     #Training[different airfoils, pressure in each n_points]
    #    X[count,m] = np.real(temp2[m])                            #Z[different airfoils, complex coordinates n_points]
    #    X[count,m + n_points] = np.imag(temp2[m]) 

    plt.plot(X[0,:int(n_points)], Y[0,:], color = "red", label = "extradós")
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x - axis')
    plt.ylabel('cp -  axis')
    plt.title('Cp')
    plt.legend()
    plt.show()

    Y_val = np.zeros((maxcount2,n_points), dtype=np.double)
    X_val = np.zeros((maxcount2,2*n_points), dtype=np.double)                
  
    count = int(0)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min + 0.33)
        for j in range(0,int(delta_N/2)):
            delta = (2*j + 1.33) * delta_max/alpha_N
            for k in range(0,int(lamda_N/2)):
                lamda = (2*k + 1.33) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min - 0.033            
                                                               
                    temp1_ = Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points)
                    temp2_ = Airfoil_boundary_Training(delta, lamda, n, n_points)

                    for m in range(0,n_points):
                        Y_val[count,m] = temp1_[m]                                     #Training[different airfoils, pressure in each n_points]
                        X_val[count,m] = np.real(temp2_[m])                            #Z[different airfoils, complex coordinates n_points]
                        X_val[count,m + n_points] = np.imag(temp2_[m]) 

                    count = count + 1                

    Y_test = np.zeros((maxcount2,n_points), dtype=np.double)
    X_test = np.zeros((maxcount2,2*n_points), dtype=np.double)                

    count = int(0)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min + 0.66)
        for j in range(0,int(delta_N/2)):
            delta = (2*j + 1.66) * delta_max/alpha_N
            for k in range(0,int(lamda_N/2)):
                lamda = (2*k + 1.66) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min - 0.066            
                                                               
                    temp1__ = Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points)
                    temp2__ = Airfoil_boundary_Training(delta, lamda, n, n_points)

                    for m in range(0,n_points):
                        Y_test[count,m] = temp1__[m]                                     #Training[different airfoils, pressure in each n_points]
                        X_test[count,m] = np.real(temp2__[m])                            #Z[different airfoils, complex coordinates n_points]
                        X_test[count,m + n_points] = np.imag(temp2__[m]) 

                    count = count + 1                     


    print("___________Training data_____________")
    print("______Training Data Obtained_________")
    print("size Output Data Y = ", Y.shape)
    print("size Input Data X = ", X.shape)

    print("_________Validation data_____________")
    print("size Output Data Y = ", Y_val.shape)
    print("size Input Data X = ", X_val.shape)

    
    print("_____________Test data_______________")
    print("size Output Data Y = ", Y_test.shape)
    print("size Input Data X = ", X_test.shape)


    return X,Y,X_val,Y_val,X_test,Y_test




