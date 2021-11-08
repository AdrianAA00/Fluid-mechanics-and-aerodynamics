# Main
from cmath import pi

### Enviroment Activation
# conda create -n myenv python=3.7 pandas jupyter seaborn scikit-learn keras tensorflow
# conda activate myenv
### Using Anaconda interpreter with their own libraries

# importing  all the functions
from AirfoilDisplay import *
from TrainingData import *

#Parameters airfoil
delta = 0.05                        #0 - 1
lamda = 0.02                        #0 - 1
alpha = (pi/180)*50                 #Degrees
n = 2                               # 1 < n < 2
n_points = 100

Airfoil(delta, lamda, alpha, n, n_points)

#Number airfoils used for training
n_points = 100
alpha_N = 10
delta_N = 10
lamda_N = 10
n_N = 1

TrainingData(alpha_N, delta_N, lamda_N, n_N, n_points)

