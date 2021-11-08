import cmath
from cmath import pi
import math
import numpy as np

from Potential import *

def TrainingData(alpha_N, delta_N, lamda_N, n_N, n_points):

    count = int(0)
    maxcount = alpha_N * delta_N * lamda_N * n_N
    Y = np.zeros((maxcount, n_points), dtype=np.double)
    X = np.zeros((maxcount,n_points), dtype=np.complex64)

    for i in range(0,alpha_N):
        alpha = i
        for j in range(0,delta_N):
            delta = (j + 1) * 0.01
            for k in range(0,lamda_N):
                lamda = (k + 1) * 0.01
                for l in range(0,n_N):
                    #n = 2 - l /(n_N - 1)
                    n = 2
                    temp1 = Pressure_field_boundary_Training(alpha, delta, lamda, n, n_points)
                    temp2 = Airfoil_boundary_Training(delta, lamda, n, n_points)

                    for m in range(0,n_points):
                        Y[count,m] = temp1[m]                    #Training[different airfoils, pressure en each n_points]
                        X[count,m] = temp2[m]                             #Z[different airfoils, complex coordinates n_points]

                    count = count + 1

    print("Training Data Obtained")
    print("size Output Data = ", np.size(X))
    print("size Input Data = ", np.size(Y))   




