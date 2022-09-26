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
    X = np.zeros((maxcount,2*n_points + 1), dtype=np.double)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min)
        for j in range(0,delta_N):
            delta = (j + 1) * delta_max/alpha_N
            for k in range(0,lamda_N):
                lamda = (k + 1) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min

                    Y[count,:], X[count,0:n_points:1], X[count,n_points:2*n_points:1] = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
                    X[count,2*n_points] = alpha
                    count = count + 1

    Y_val = np.zeros((maxcount2,n_points), dtype=np.double)
    X_val = np.zeros((maxcount2,2*n_points + 1), dtype=np.double)                
  
    count = int(0)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min + 0.33)
        for j in range(0,int(delta_N/2)):
            delta = (2*j + 1.33) * delta_max/alpha_N
            for k in range(0,int(lamda_N/2)):
                lamda = (2*k + 1.33) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min - 0.033            
                                                               
                    Y_val[count,:], X_val[count,0:n_points:1], X_val[count,n_points:2*n_points:1] = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
                    X_val[count,2*n_points] = alpha
                    count = count + 1                

    Y_test = np.zeros((maxcount2,n_points), dtype=np.double)
    X_test = np.zeros((maxcount2,2*n_points + 1), dtype=np.double)                

    count = int(0)

    for i in range(0,alpha_N):
        alpha = (pi/180)*((alpha_max - alpha_min)*i/(alpha_N-1) + alpha_min + 0.66)
        for j in range(0,int(delta_N/2)):
            delta = (2*j + 1.66) * delta_max/alpha_N
            for k in range(0,int(lamda_N/2)):
                lamda = (2*k + 1.66) * lamda_max/lamda_N
                for l in range(0,n_N):
                    n = (2 - n_min)*l/(n_N - 1) + n_min - 0.066            
                                                               
                    Y_test[count,:], X_test[count,0:n_points:1], X_test[count,n_points:2*n_points:1] = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
                    X_test[count,2*n_points] = alpha
                    count = count + 1                     

    #X_test, Y_test = pd.read_csv("/Users/jojomoolayil/Book/Ch3/Data/train.csv")

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