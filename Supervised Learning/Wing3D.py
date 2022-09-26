import math
from re import I, U
from turtle import shapesize
import numpy as np
import matplotlib.pyplot as plt
from math import atan, cos, pi, sin
from numpy import transpose
from numpy import linalg as LA
from Potential import *
from mpl_toolkits.mplot3d import Axes3D


def Geometry_wing(c1,c2,c3,SW,Kurv,b,m_change,m_panels,n_points,E,C,alpha):

    #Wing spinne coordinates
    x = np.zeros((m_panels+1), dtype=np.double)
    y = np.zeros((m_panels+1), dtype=np.double)
    z = np.zeros((m_panels+1), dtype=np.double)
    
    for i in range(m_panels+1):
        #y[i] = (b/2)*(-cos((2*i/m_panels)*pi/2)+1)/2
        y[i] = (b/(2*m_panels))*i

        #x[i] = (b/2)*(-cos((2*i/m_panels)*pi/2)+1)*SW/2 + Kurv*((-cos((2*i/m_panels)*pi/2)+1)/2)**2
        x[i] = (b/(2*m_panels))*i*SW/2 + Kurv*((b/(2*m_panels))*i)**2

        #z[i] = ((i/m_panels)**50)*2 + ((i/m_panels)**1.4)*5
        z[i] = 0

    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    # ax.scatter(x,y,z) 
    # ax.scatter(x,-y,z)
    # ax.set_title("Spline wing")
    # plt.show()

    X_ = np.zeros((m_panels+1,n_points+1), dtype=np.double)
    Y_ = np.zeros((m_panels+1,n_points+1), dtype=np.double)
    Z_i = np.zeros((m_panels+1,n_points+1), dtype=np.double)
    Z_e = np.zeros((m_panels+1,n_points+1), dtype=np.double)

    for i in  range(m_panels+1):
        for j in range(n_points+1):
            Y_[i,j] = y[i]

            if i <= m_change:
                #X_[i,j] = x[i] + (c1 - ((c1-c2)*(-cos((2*i/m_panels)*pi/2)+1)/(-cos((2*m_change/m_panels)*pi/2)+1)))*j/(n_points-1)
                X_[i,j] = x[i] + (c1 - ((c1-c2)*(i)/(m_change)))*j/(n_points-1)
            else:
                #X_[i,j] = x[i] + (c2 - ((c2-c3)*(+cos((2*m_change/m_panels)*pi/2)-cos((2*i/m_panels)*pi/2))/(cos((2*m_change/m_panels)*pi/2)+1)))*j/(n_points-1)
                X_[i,j] = x[i] + (c2 - ((c2-c3)*((i-m_change))/(-m_change+m_panels)))*j/(n_points-1)
            
            Z_i[i,0] = z[i]
            Z_e[i,0] = z[i]

            if j<n_points:
                Z_i[i,j+1] = z[i] + (E[n_points-1-j] + C[n_points-1-j])*(c1-(c1-c3)*(i/m_panels))/4
                Z_e[i,j+1] = z[i] + (-E[n_points-1-j] + C[n_points-1-j])*(c1-(c1-c3)*(i/m_panels))/4

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-10, 10)
    # ax.plot_trisurf(X_.flatten(),Y_.flatten(),Z_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_.flatten(),Y_.flatten(),Z_i.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_.flatten(),-Y_.flatten(),Z_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_.flatten(),-Y_.flatten(),Z_i.flatten(),cmap='Greys', edgecolor='none')
    ax.scatter(X_.flatten(),Y_.flatten(),Z_e.flatten(),s=1, c='Black') 
    ax.scatter(X_.flatten(),-Y_.flatten(),Z_e.flatten(),s=1,c='Black')
    ax.scatter(X_.flatten(),Y_.flatten(),Z_i.flatten(),s=1,c='Black') 
    ax.scatter(X_.flatten(),-Y_.flatten(),Z_i.flatten(),s=1,c='Black')
    # ax.scatter(x,y,z) 
    # ax.scatter(x,-y,z)

    ax.set_title("Spline wing")
    plt.show()

    return  X_,Y_,Z_i,Z_e,x,y,z

def VortexVelocR(rot,r,m):
    n = np.array([1, 0, 0])
    U = (rot/(4*pi))*(1 - np.dot(np.transpose(n),(r-m)/LA.norm(r-m)))*(np.cross((r-m),n)/LA.norm(np.cross((r-m),n))**2)

    return U

def VortexVelocR1R0(rot,r1,r0,m):
    U = (rot/(4*pi))*np.dot(np.transpose(r1-r0), (r1-m)/LA.norm(r1-m)- (r0-m)/LA.norm(r0-m))*(np.cross((r0-m),(r1-r0))/LA.norm(np.cross((r0-m),(r1-r0)))**2)            

    return U


def Rotational(alpha,delta, lamda, n, n_points,c,C,E):

    cp, E, C = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
    #cp = model.predict(C,E)

    Cp_global = 0

    for i in range(np.size(cp, 0)):
        Cp_global = Cp_global + cp[i]

    Cp_global = Cp_global/np.size(cp, 0)
    rot = Cp_global*c/2

    return rot

def Vortex3D(alpha, delta, lamda, n, n_points, m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,C,E):

    XYZ_R =  np.zeros((m_panels+1,3), dtype=np.double)
    XYZ_L =  np.zeros((m_panels+1,3), dtype=np.double)
    rot =  np.zeros((m_panels), dtype=np.double)

    XYZ_R[:,0] = X_3D[:,int(n_points/4)]
    XYZ_R[:,1] = Y_3D[:,0]
    XYZ_R[:,2] = z
    XYZ_L[:,0] = X_3D[:,int(n_points/4)]
    XYZ_L[:,1] = -Y_3D[:,0]
    XYZ_L[:,2] = z

    alpha_ = np.zeros((m_panels), dtype=np.double)

    #Angles of attack in each airfoil section
    for i in range(m_panels):
        alpha_[i] = alpha

    maxiter = 50

    for iter in range(maxiter):
        #Finding rotational in each point
        for i in range(m_panels):
            c = (((X_3D[i+1,n_points] - X_3D[i+1,0]) + (X_3D[i,n_points] - X_3D[i,0]))/2)                                                                                             #Interpolate value of chord in i panel
            #c = c*((Y[1+i]-Y[i])**2 + (X[1+i,0]-X[i,0])**2 + (Z[i+1]-Z[i])**2)**(1/2)/((X[i+1,0]-X[i,0])**2 + (Z[i+1]-Z[i])**2)**(1/2)                                   #Chord correction

            #rot[i] = (((X[i+1,0]-X[i,0])**2 + (Z[i+1]-Z[i])**2)**(1/2)/((Y[1+i]-Y[i])**2 + (X[1+i,0]-X[i,0])**2 + (Z[i+1]-Z[i])**2)**2)*Rotational(alpha_[i],delta, lamda, n, n_points,c,C,E)      #Rotational corrected
            rot[i] = Rotational(alpha_[i],delta, lamda, n, n_points,c,C,E)

        #print(rot)
        v = np.zeros((3,m_panels), dtype=np.double)

        for j in range(m_panels):       #Only study right wing, simetry
            m = (XYZ_R[j,:] + XYZ_R[j+1,:])/2 #Point in the middle of the panel chord 1/4

            for i in range(2*m_panels):
                if i < m_panels:     #Right wing
                    r0 = XYZ_R[i,:]              #Point in the middle of the panel chord 1/4
                    r1 = XYZ_R[i+1,:]
        
                    v[:,j] = v[:,j] + VortexVelocR(-rot[i],r0,m)
                    v[:,j] = v[:,j] + VortexVelocR(rot[i],r1,m)
                    #print(VortexVelocR(-rot[i],r0,m))
                    #print(VortexVelocR(rot[i],r1,m))

                else:    #Left wing
                    r0 = XYZ_L[i+1-m_panels,:]            #Point in the middle of the panel chord 1/4
                    r1 = XYZ_L[i-m_panels,:]
                
                    v[:,j] = v[:,j] + VortexVelocR(-rot[i-m_panels],r0,m)
                    v[:,j] = v[:,j] + VortexVelocR(rot[i-m_panels],r1,m)
        
        rate = 0.1

        for k in range(m_panels):
            alpha_[k] = alpha_[k] + (-alpha_[k] + math.atan(((v[1,k]*(z[k+1]-z[k])+v[2,k]*(y[k+1]-y[k]))/((z[k+1]-z[k])**2+(y[k+1]-y[k])**2)**(1/2)+math.sin(alpha))/(math.cos(alpha)+v[0,k])))*rate
        
        print("Iteration", iter, "percentage error",  100*abs(alpha_[int(m_panels/2)] - math.atan(v[2,int(m_panels/2)]+ math.sin(alpha)/math.cos(alpha)))/alpha_[int(m_panels/2)])

    print("Velocity, 1/4 cord, right wing", v)

    plt.title('Angle of attack due to induced velocity')
    plt.xlabel('Wing Span')
    plt.ylabel('Angle of Attack, degrees')
    plt.plot(y[:m_panels], alpha_*180/pi)
    plt.plot(-y[:m_panels], alpha_*180/pi)
    plt.show()

    Cp_global = 2*rot
    
    plt.title('Dimensionless lift per span unit')
    plt.xlabel('Wing Span y')
    plt.ylabel('dcp/dy')
    plt.plot(y[:m_panels], Cp_global)
    plt.plot(-y[:m_panels], Cp_global)
    plt.show()

    #3D Representation of the lift and induced drag:
    Cp_3D = np.zeros((m_panels,n_points), dtype=np.double)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(-50, 50)

    aux = np.zeros((n_points), dtype=np.double)
    Cp_3D_vector = np.zeros((m_panels,n_points,3), dtype=np.double)
    Cd_3D_vector = np.zeros((m_panels,3), dtype=np.double)

    Cl_wing = 0
    Cd_wing = 0
    c_media = 0

    for i in range(m_panels):
        c = (((X_3D[i+1,n_points] - X_3D[i+1,0]) + (X_3D[i,n_points] - X_3D[i,0]))/2)  
        aux1, E, C = Airfoil_boundary_Training_E_C(alpha_[i],delta, lamda, n, n_points)

        for j in range(n_points):
            Cp_3D[i,j] = aux1[n_points-1-j]#*c
            Cl_wing = Cl_wing + Cp_3D[i,j]*c
            Cd_wing = Cd_wing + Cp_3D[i,j]*c*sin(alpha-alpha_[i])

        c_media = c_media + c/m_panels
        
        Cp_3D_vector[i,:,2] = Cp_3D[i,:]
        Cd_3D_vector[i,0] = Cp_3D[i,n_points-1]*sin(alpha-alpha_[i])

        aux[:] = (Y_3D[i,0]+Y_3D[i+1,0])/2

        ax.quiver(X_3D[i,:n_points],aux,Z_3D_i[i,:n_points],Cp_3D_vector[i,:,0],Cp_3D_vector[i,:,1],Cp_3D_vector[i,:,2],length=10) 
        ax.quiver(X_3D[i,:n_points],-aux,Z_3D_i[i,:n_points],Cp_3D_vector[i,:,0],Cp_3D_vector[i,:,1],Cp_3D_vector[i,:,2],length=10) 
        ax.quiver(X_3D[i,n_points],aux[0],Z_3D_i[i,n_points],Cd_3D_vector[i,0],Cd_3D_vector[i,1],Cd_3D_vector[i,2],color="r",length=1000) 
        ax.quiver(X_3D[i,n_points],-aux[0],Z_3D_i[i,n_points],Cd_3D_vector[i,0],Cd_3D_vector[i,1],Cd_3D_vector[i,2],color="r",length=1000) 

    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),s=1, c='Black') 
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),s=1,c='Black')
    ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),s=1,c='Black') 
    ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),s=1,c='Black')

    ax.set_title("Lift")
    plt.show()

    #Cl, Cd y E:
    Cp2d,E,C = Airfoil_boundary_Training_E_C(alpha,delta, lamda, n, n_points)
    Cp2d_ = 0
    for i in range(n_points):
        Cp2d_ = Cp2d_ + Cp2d[i]
    Cp2d_ = Cp2d_/n_points
    Cl_wing = Cl_wing/(m_panels*n_points*c_media)
    Cd_wing = Cd_wing/(m_panels*n_points*c_media)
    E = Cl_wing/Cd_wing


    print("Airfoil Cl", Cp2d_)
    print("Global Cl wing",Cl_wing)
    print("Global Cd wing",Cd_wing)
    print("Global E wing",E)

    return rot

def VortexVelocity(point,XYZ_R,XYZ_L,rot,m_panels):
    v = np.zeros((3), dtype=np.double)
 
    for i in range(2*m_panels):
        if i < m_panels:     #Right wing
            r0 = XYZ_R[i,:]              #Point in the middle of the panel chord 1/4
            r1 = XYZ_R[i+1,:]
        
            v[:] = v[:] + VortexVelocR(-rot[i],r0,point)
            v[:] = v[:] + VortexVelocR(rot[i],r1,point)
            v[:] = v[:] + VortexVelocR1R0(rot[i],r1,r0,point)

        else:    #Left wing
            r0 = XYZ_L[i+1-m_panels,:]            #Point in the middle of the panel chord 1/4
            r1 = XYZ_L[i-m_panels,:]
                
            v[:] = v[:] + VortexVelocR(-rot[i-m_panels],r0,point)
            v[:] = v[:] + VortexVelocR(rot[i-m_panels],r1,point)
            v[:] = v[:] + VortexVelocR1R0(rot[i-m_panels],r1,r0,point)
    
    return v

def StreamLines(initial_point, steps, size_step, XYZ_R,XYZ_L,rot,m_panels,alpha):

    xyz_s = np.zeros((steps + 1,3), dtype=np.double)
    xyz_s[0,:]  = initial_point
    aux = size_step
    for i in range(steps):
        if xyz_s[i,0] <= 7 and xyz_s[i,0] >= -2:
            size_step = 0.1*aux
            xyz_s[i+1,:] = xyz_s[i,:] + size_step*(np.array([cos(alpha),0,sin(alpha)])+VortexVelocity(xyz_s[i,:] ,XYZ_R,XYZ_L,rot,m_panels))
        else:
            size_step = aux
            xyz_s[i+1,:] = xyz_s[i,:] + size_step*(np.array([cos(alpha),0,sin(alpha)])+VortexVelocity(xyz_s[i,:] ,XYZ_R,XYZ_L,rot,m_panels))

    return xyz_s       

def VelocityRepresentation(rot,n_points,m_panels,X_3D,Y_3D,Z_3D_i,Z_3D_e,x,y,z,alpha):

    XYZ_R =  np.zeros((m_panels+1,3), dtype=np.double)
    XYZ_L =  np.zeros((m_panels+1,3), dtype=np.double)

    XYZ_R[:,0] = X_3D[:,int(n_points/4)]
    XYZ_R[:,1] = Y_3D[:,0]
    XYZ_R[:,2] = z
    XYZ_L[:,0] = X_3D[:,int(n_points/4)]
    XYZ_L[:,1] = -Y_3D[:,0]
    XYZ_L[:,2] = z

    #Downwash
    N1 = 40            #points in the Treffe´s plane
    N2 = 40
    x_= 100            #Distance of the wing wake

    xyz = np.zeros((N1,N2,3), dtype=np.double)   
    xyz_ = np.zeros((N1*N2,3), dtype=np.double)                                
    v = np.zeros((N1,N2,3), dtype=np.double)
    v_ = np.zeros((N1*N2,3), dtype=np.double)

    incre = 35/m_panels

    for i in range(N1):
        for j in range(N2):
            xyz[i,j,:] = np.array([x_,incre*N1/2 - incre*i +  incre/2,incre*N2/2 - incre*j])    #Assesed only in the middle of pannels y coordinate
            v[i,j,:] = VortexVelocity(xyz[i,j,:],XYZ_R,XYZ_L,rot,m_panels)

    print(v[:,:,:])

    for i in range(3):
        xyz_[:,i] = xyz[:,:,i].flatten()
        v_[:,i] = v[:,:,i].flatten()

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-50, 150)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(-50, 50)
    ax.quiver(xyz_[:,0],xyz_[:,1],xyz_[:,2],v_[:,0],v_[:,1],v_[:,2],length=40) 

    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),s=1, c='Black') 
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),s=1,c='Black')
    ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),s=1,c='Black') 
    ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),s=1,c='Black')

    ax.set_title("Vortex velocity downwash")
    plt.show()

    #Stream lines
    steps = int(100)
    size_step = 1
    n_initial_points = 40

    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-5, 10)
    ax.set_ylim3d(-0, 35)
    ax.set_zlim3d(-5, 5)

    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),s=10, c='Black') 
    ax.scatter(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),s=10,c='Black')

    for j in range(m_panels):
        initial_point = np.array([(x[j]+x[j+1])/2-10,(y[j]+y[j+1])/2,(z[j]+z[j+1])/2])
        xyz_s = StreamLines(initial_point, steps, size_step, XYZ_R,XYZ_L,rot,m_panels,alpha)
        for i in range(steps):
            ax.plot([xyz_s[i,0], xyz_s[i+1,0]], [xyz_s[i,1], xyz_s[i+1,1]], zs=[xyz_s[i,2], xyz_s[i+1,2]],lw=1,c="Blue")
           # ax.plot([xyz_s[i,0], xyz_s[i+1,0]], [-xyz_s[i,1], -xyz_s[i+1,1]], zs=[xyz_s[i,2], xyz_s[i+1,2]],lw=0.4,c="Blue")

    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    # ax.plot_trisurf(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')

    # ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_e.flatten(),s=0.5,c='Black') 
    # ax.scatter(X_3D.flatten(),-Y_3D.flatten(),Z_3D_i.flatten(),s=0.5,c='Black')

    ax.set_title("Vortex velocity downwash")
    plt.show()

    #Vortices streamlines
    steps = int(800)
    size_step = 0.1
    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 20)
    ax.set_ylim3d(-0, 35)
    ax.set_zlim3d(-5, 5)
    initial_point = np.array([(x[m_panels])-5,(y[m_panels]+0.01),(z[m_panels])])
    xyz_s = StreamLines(initial_point, steps, size_step, XYZ_R,XYZ_L,rot,m_panels,alpha)

    for i in range(steps):
        ax.plot([xyz_s[i,0], xyz_s[i+1,0]], [xyz_s[i,1], xyz_s[i+1,1]], zs=[xyz_s[i,2], xyz_s[i+1,2]],lw=1,c="Blue")

    initial_point = np.array([(x[m_panels])-5,(y[m_panels]+0.01),(z[m_panels] + 0.5)])
    xyz_s = StreamLines(initial_point, steps, size_step, XYZ_R,XYZ_L,rot,m_panels,alpha)

    for i in range(steps):
        ax.plot([xyz_s[i,0], xyz_s[i+1,0]], [xyz_s[i,1], xyz_s[i+1,1]], zs=[xyz_s[i,2], xyz_s[i+1,2]],lw=1,c="Blue")   

    
    initial_point = np.array([(x[m_panels])-5,(y[m_panels]+0.01),(z[m_panels] - 0.5)])
    xyz_s = StreamLines(initial_point, steps, size_step, XYZ_R,XYZ_L,rot,m_panels,alpha)

    for i in range(steps):
        ax.plot([xyz_s[i,0], xyz_s[i+1,0]], [xyz_s[i,1], xyz_s[i+1,1]], zs=[xyz_s[i,2], xyz_s[i+1,2]],lw=1,c="Blue")     

    ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_e.flatten(),cmap='Greys', edgecolor='none')
    ax.plot_trisurf(X_3D.flatten(),Y_3D.flatten(),Z_3D_i.flatten(),cmap='Greys', edgecolor='none')
    ax.set_title("Vortex velocity downwash")

    plt.show()


    return



