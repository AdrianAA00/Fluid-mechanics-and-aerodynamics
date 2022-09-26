import math
from re import A
import numpy as np
from sklearn.metrics import jaccard_score
from sympy import I
from numpy import linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
import pylab

from Potential import *

def NN(X, Y, L,theta,theta_end):

    columns_X = np.size(X[0,:])                                                 # It is going to be transposed, inversed order
    rows_X = np.size(X[:,0])
    rows_Y = np.size(Y[:,0])  

    X_extended = np.ones((rows_X+1,columns_X), dtype=np.double)                 # Extended X including b terms
    X_extended[1:,:] = X   
                                                     
    a = np.zeros((rows_X+1,columns_X,L+1), dtype=np.double)                       # a terms does not include b terms, added into z terms
    a_end = np.zeros((rows_Y,columns_X), dtype=np.double)                         # exit a does also not have b terms
    z = np.zeros((rows_X,columns_X,L+1), dtype=np.double)                         # including b terms
    #z_end = np.zeros((rows_Y,columns_X), dtype=np.double)                        It is not necessary because linear final activation function

    for i in range(0,L+1):
        if i == 0:
            a[:,:,0] = X_extended
            z[:,:,0] = X
        else:
            z[:,:,i] = np.dot(theta[:,:,i-1],a[:,:,i-1])                               
            temp = g(z[:,:,i])
            a[:,:,i] = np.ones((rows_X+1,columns_X), dtype=np.double)           # Activation function RELU   
            a[1:,:,i] = temp                      

    a_end = np.dot(theta_end,a[:,:,L])                                          # Final linear activation function for regression

    return a_end, a, z

#Activation function hidden layers and output 
def g(x):
    columns_X = np.size(x[0,:])
    rows_X = np.size(x[:,0])
    zero = np.zeros((rows_X,columns_X), dtype=np.double)
    #g = np.maximum(zero,x)
    g = np.zeros((rows_X,columns_X), dtype=np.double)

    for i in range(0,rows_X):
        for j in range(0,columns_X):
            if x[i,j] <= 0:
                g[i,j] = 0#0.001*x[i,j]
            else:
                g[i,j] = x[i,j]
    

    return g

#Derivative activation function hidden layers and output
def dg(x):

    columns_X = np.size(x[0,:])
    rows_X = np.size(x[:,0])

    dg = np.zeros((rows_X,columns_X), dtype=np.double)

    for i in range(0,rows_X):
        for j in range(0,columns_X):
            if x[i,j] <= 0:
                dg[i,j] = 0.00
            else:
                dg[i,j] = 1
    
    return dg

#Find Jacobian value. Backpropagation method:
def Jacobian(X, Y, L,theta,theta_end):
    columns_X = np.size(X[0,:]) 
    rows_X = np.size(X[:,0])
    rows_Y = np.size(Y[:,0])

    (a_end, a, z) = NN(X, Y, L,theta,theta_end)
    deltaLmas1 = a_end - Y                                              #Separate delta because of diferent size of Y and X

    delta = np.zeros((rows_X,columns_X,L+1), dtype=np.double)           #Not using extended X
    delta[:,:,L] = np.multiply(np.dot(np.transpose(theta_end[:,1:]),deltaLmas1),dg(z[:,:,L]))

    for i in range(L-1,-1,-1):                                          #No need to reach index 0,   L, L-1 ,..., 1
        delta[:,:,i] = np.multiply(np.dot(np.transpose(theta[:,1:,i]),delta[:,:,i+1]),dg(z[:,:,i]))

    triang = np.zeros((rows_X,rows_X + 1,L), dtype=np.double)
    triang_end = np.zeros((rows_Y,rows_X + 1), dtype=np.double)

    for k in range(0,columns_X):                                        #Adding all data from different airfoils 
        for l in range(0,L):                                            #For L+1 we use triang_end
            for i in range(0,rows_X):
                for j in range(0,rows_X + 1):
                        triang[i,j,l] = triang[i,j,l] + a[j,k,l]*delta[i,k,l+1]

        #Now with last layer
        for i in range(0,rows_Y):
            for j in range(0,rows_X + 1):
                    triang_end[i,j] = triang_end[i,j] + a[j,k,L]*deltaLmas1[i,k]

    J = (1/columns_X)*triang                                           #Average for all data
    J_end = (1/columns_X)*triang_end 

    return  J, J_end

# Training of the NN. Finding weights with gradient descent. Backpropagation for finding out gradient
def NN_training(X, Y, L,epsilon,alpha,maxiter):
    
    columns_X = np.size(X[0,:])                                       # It is going to be transposed, inversed order
    rows_X = np.size(X[:,0])
    rows_Y = np.size(Y[:,0])

    X_extended = np.ones((rows_X+1,columns_X), dtype=np.double)                 # Extended X including b terms
    X_extended[1:rows_X+1:1,:] = X   

    theta = np.random.rand(rows_X,rows_X+1,L)-0.5             #Matrix with weights Random initialization
    theta_end = np.random.rand(rows_Y,rows_X+1)-0.5            #Output weights size_Y * size_X
    #theta = np.ones((rows_X,rows_X+1,L), dtype=np.double)             #Matrix with weights Random initialization
    #theta_end = np.ones((rows_Y,rows_X+1), dtype=np.double)            #Output weights size_Y * size_X

    err = np.zeros((maxiter), dtype=np.double)                                #Error in each iteration gradient descent 

    #(J,J_end) = Jacobian(X, Y, L,theta,theta_end)
    #(J_n,J_end_n) = numerical_Jacobian(X,Y,L,theta,theta_end)
    
    #Try gradients are working correctly:
    #print("J",J)
    #print("J_N",J_n)
    #print(J[0,0,1]/J_n[0,0,1]*J_n - J)
    #print("J_end",J_end)
    #print("J_end_n",J_end_n)
    #print(J[0,0,1]/J_n[0,0,1]*J_end_n - J_end)

    for i in range(0,maxiter):
        temp1 = theta
        temp2 = theta_end

        (J,J_end) = Jacobian(X, Y, L,theta,theta_end)                         #Analytical Jacobian BackPropagation
        #(J2,J_end2) = numerical_Jacobian(X,Y,L,theta,theta_end)                 #Numerical Jacobian

        theta = theta - alpha*J                                                #Update theta with data gradient descent
        theta_end = theta_end  - alpha*J_end
        #print(theta,theta_end)
        #print(J_end)
        #print(J2,J_end2)
        (a_end, a, z) = NN(X, Y, L,theta,theta_end)
        #print(a_end)
        count = i

        if i>0:
            err[i] = la.norm((a_end - Y))/la.norm(Y)                           #Stop criteria in gradient descent
            print("Iteration number:", i, "Error:", err[i])

            if err[i] < epsilon:
                print("Convergence of gradient descent. Theta optimization")
                break

            if la.norm((err[i]-err[i-1])/err[i-1]) < epsilon**2:
                print("Convergence, error above limit. Modify NN")
                break
        
        if i == maxiter-1:
            print("Max iteration limit reached")

    #Plot error function in each iteration
    plt.xlabel('iteration')
    plt.ylabel('error')
    # #plt.xlim(min(x) - 1, max(x) + 1)
    # #plt.ylim(min(y) - 1, max(y) + 1)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Error NN')
    plt.plot(range(0,count), err[0:count])
    plt.show()

    return theta,theta_end

def numerical_Jacobian(X,Y,L,theta,theta_end):
    columns_X = np.size(X[0,:])                                       # It is going to be transposed, inversed order
    rows_X = np.size(X[:,0])
    rows_Y = np.size(Y[:,0])

    incre = 0.000000001
    J = np.zeros((rows_X,rows_X + 1,L), dtype=np.double)
    J_end = np.zeros((rows_Y,rows_X + 1), dtype=np.double)

    for l in range(0,L):                                            #For L+1 we use triang_end
        for i in range(0,rows_X):
            for j in range(0,rows_X + 1):
                (a_end, a, z) = NN(X, Y, L,theta,theta_end)
                J_ = (1/2)*(la.norm(a_end-Y))**2
                theta[i,j,l] = theta[i,j,l] + incre
                (a_end_N, a_N, z_N) = NN(X, Y, L,theta,theta_end)
                J_N = (1/2)*la.norm(a_end_N-Y)**2

                J[i,j,l] = (1/columns_X)*(J_N - J_)/incre
        
        for i in range(0,rows_Y):
            for j in range(0,rows_X + 1):
                (a_end, a, z) = NN(X, Y, L,theta,theta_end)
                J_en = (1/2)*la.norm(a_end-Y)**2
                theta_end[i,j] = theta_end[i,j] + incre
                (a_end_N, a_N, z_N) = NN(X, Y, L,theta,theta_end)
                J_N_end = (1/2)*(la.norm(a_end_N-Y))**2

                J_end[i,j] = (1/columns_X)*(J_N_end - J_en)/incre

    return J, J_end