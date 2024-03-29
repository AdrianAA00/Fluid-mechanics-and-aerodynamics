# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //                                                                                                                                                                 //
# //                                                                          Python                                                                                 //
# //                                                                                                                                                                 //
# //                                                                Author: Adrián Antón Álvarez                                                                     //                    
# //                                                                                                                                                                 //
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# //************************************************************************ DESCRIPTION ****************************************************************************//
# There are serveral codes which include non-compressible, compressible and viscous solutions. Then the Neural Network should be selected: NN1D (own developed NN)
# or Keras library. Finally, the user should select the model that he wants to use from all the available ones.
# //*****************************************************************************************************************************************************************//


# Main_
from cmath import pi
from tkinter import Y
from tkinter.ttk import Separator

from numpy import transpose
from keras.models import Sequential,save_model, load_model
from sympy import C

#___Available Solvers and models____
#from AirfoilDisplay import *
#from Wing3D import *
#from CompressibleFlow import *
#from Compressible2 import *
from CompressibleM import *
#from Supersonic import *
#from CompressibleNonLinear import *

##_________NN1D________________
#from NeuralNetwork import *
#from TrainingData import *

##_________NN1D________________
#from NeuralNetwork1D import * 
#from TrainingData1D import *

##_________Keras_______________
#from NN_keras import *
#from TrainingDataKeras import *

##_______Keras__E___C__________
from NN_keras import *
from TrainingDataKerasR import *

##_______Viscous________________
from NeuralNetworkViscous import *
from Wing3D_viscous import *


#_________________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________________Potential Field Representation_______________________________________________________________________
#Parameters airfoil
delta = 0.25                       #0 - 1
lamda = 0.175                      #0 - 1
alpha = (pi/180)*0              #Degrees
n = 1.8                            #1 < n < 2
n_points = 100
#Airfoil(delta, lamda, alpha, n, n_points)
#Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________________________________________________________
#_____________________________________________________________Parameters for training ____________________________________________________________________________________
#Number airfoils used for training
n_points = 25
alpha_N = 2              #Always 2 powers
delta_N = 4
lamda_N = 4
n_N = 2

alpha_max = 10
alpha_min = 9 
delta_max = 0.04
lamda_max = 0.02
n_min = 2
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________



#_________________________________________________________________________________________________________________________________________________________________________
#_____________________________________________________________Own Potential Neural Network _______________________________________________________________________________

#(X,Y) = TrainingData(alpha_N, delta_N, lamda_N, n_N, n_points)                                   

# #·····················Define NN······························
# #All hiden layers same number of cells as entry X + 1, b
# L = 2 #Number of Layers

# (theta,theta_end) = NN_training(X, Y, L,epsilon = 0.001,alpha = 0.5, maxiter = 1000)
# (a_end, a, z) = NN(X, Y, L,theta,theta_end)

# #·····················Results································
# print("Y_pred",a_end)
# print("Y",Y)
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________________________________________________________________
#_____________________________________________________________Keras Potential Neural Network _____________________________________________________________________________


#(X_train,Y_train,X_val,Y_val,X_test,Y_test) = TrainingData(alpha_N, delta_N, lamda_N, n_N, n_points, alpha_max ,alpha_min, delta_max, lamda_max, n_min)                               
# print("X", X_train)
# print("Y", Y_train)
# print("X", X_val)
# print("Y", Y_val)
# print("X", X_test)
# print("Y", Y_test)

#NN(X_train, Y_train, X_val, Y_val, X_test, Y_test)

#filepath = "./saved_airfoil_model"
#model = load_model(filepath, compile = True)
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________



#_________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________3D WING ___________________________________________________________________________________
# Set wing
c1 = 7
c2 = 3
c3 = 2

b = 70
SW = 20*(pi/180)
KURV = 0
 
m_panels = 10
m_change = 3

#Set airfoil
#Parameters airfoil
delta = 0.06                       #0 - 1
lamda = 0.12                      #0 - 1
alpha = (pi/180)*0              #Degrees
n = 2                          #1 < n < 2
n_points = int(5)

#Airfoil(delta, lamda, alpha, n, n_points)
# Cp, E, C = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
# X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z = Geometry_wing(c1,c2,c3,SW,KURV,b,m_change,m_panels,n_points,E,C,alpha)

# Solved_rot = Vortex3D(alpha, delta, lamda, n, n_points, m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,E,C)

# VelocityRepresentation(Solved_rot,n_points,m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,alpha)
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________



#_________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________Compressible airfoil_______________________________________________________________________
#Set airfoil
#Parameters airfoil
delta = 0.06                       #0 - 1
lamda = 0.12                      #0 - 1
alpha = (pi/180)*00              #Degrees
n = 2                          #1 < n < 2
n_points = 200
n_R = 16
R_max = 20
M = 10
gamma = 1.4

#x,y = Airfoil_net(R_max, n_R, delta, lamda, alpha, n, n_points)
# DiferentialEquation(x,y,n_points,n_R,alpha,delta,lamda,n,M)
#_________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________________





#_________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________Viscous airfoil_______________________________________________________________________
#Set airfoil
#Parameters airfoil
delta = 0.06                       #0 - 1
lamda = 0.12                      #0 - 1
alpha = (pi/180)*0              #Degrees
n = 2                          #1 < n < 2
n_points = 20
n_R = 64
R_max = 20
M = 0.5
gamma = 1.4

# x,y = Airfoil_net(R_max, n_R, delta, lamda, alpha, n, n_points)


# data = np.zeros((n_points,2), dtype=np.double)

# # define data
# data[:,0] = x[0,:]
# data[:,1] = y[0,:] 

# # save to csv file
# np.savetxt('x_airfoil.csv', data[:,0], delimiter=',')
# np.savetxt('y_airfoil.csv', data[:,1], delimiter=',')

#data2 = np.loadtxt(".\DataViscous.csv",dtype=float, delimiter=",")
#print(data2)

#NN_cl(alpha)
#N_cd(alpha)

#______________________________________________________________________________3D WING ___________________________________________________________________________________
# Set wing
c1 = 20
c2 = 8
c3 = 3

b = 65
SW = 60*(pi/180)
SW2 = 25*(pi/180)
KURV = 0
 
m_panels = 20
m_change = 12

#Set airfoil
#Parameters airfoil
delta = 0.06                       #0 - 1
lamda = 0.12                      #0 - 1
alpha = (pi/180)*5             #Degrees
n = 2                          #1 < n < 2
n_points = int(10)

#Airfoil(delta, lamda, alpha, n, n_points)

Cp, E, C = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z = Geometry_wing(c1,c2,c3,SW,KURV,b,m_change,m_panels,n_points,E,C,alpha,SW2)

Solved_rot = Vortex3D(alpha, delta, lamda, n, n_points, m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,E,C,SW,b)

#VelocityRepresentation(Solved_rot,n_points,m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,alpha)
#_________________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________  
