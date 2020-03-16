"""
This Python file contains definitions for the three functions:
    (A)calibrateCamera3D(data)
    (B)visualiseCameraCalibration3D(data, P)
    (C)evaluateCameraCalibration3D(data, P).
    (D)Driver function main() to call the above three functions.

(A) calibrateCamera3D(data) performs camera calibration by computing Perspective matrix P 
using eigenvalues of matrix A.

(B)visualiseCameraCalibration3D(data, P) renders a single 2D plot showing 
(i) the measured 2D image point (as given in the question), and, 
(ii) the reprojection of the 3D calibration points as computed by P.

(C)evaluateCameraCalibration3D(data, P) evaluates  mean, variance, minimum, and maximum
distances in pixels between the measured and reprojected image feature locations.


Created on Sat Nov 23 22:28:54 2019

@author: Deepthi Jain Brahmadev.
Student ID: 19252262.
"""

import numpy as np
import matplotlib.pyplot as plt


def calibrateCamera3D(data):
    #Finding the length of Nx5 data matrix given.
    Data_L = len(data)   #data matrix given is of the size 491x5.
    
    #Construct a matrix of order 2*DataLX12 which is used to hold homogenous linear solution of the camera matrix.
    #Initially 0's are filled into A matrix.
    A = np.zeros((Data_L* 2, 12))
    
    #x,y points (2D) points of the data matrix given is extracted and multiplied by -1.
    Data2D_x_y = data[::,3:5]*-1
    
    #x,y coordinates are splitted from the extracted data2D matrix.
    Data_x = Data2D_x_y[::,0:1] #extracts 2D x coordinates.
    Data_y = Data2D_x_y[::,1:2] #extracts 2D y coordinates.
    
    #x,y coordinates are repeated to form a matrix of order 491x4 to compute dot product of 2D and 3D data points.
    Data2D_x= np.tile(Data_x,4)
    Data2D_y= np.tile(Data_y,4)
    
    #3D Data points are extracted from the given Data matrix i.e. X,Y,Z and converted to homogeneous points.
    #by appending 1 and converting the matrix to order 491x4.
    Data_X_Y_Z= data[::,0:3]
    ones_X_Y_Z= np.ones((491,1))
    Data_X_Y_Z_1= np.append(Data_X_Y_Z,ones_X_Y_Z,axis=1)
    #[X,Y,Z,1] homogeneous points are obtained.
    
    #2D homogeneous points and 3D homogeneous points are multiplied. (dot product).
    #[-xX,-xY,-xZ,-x] and [-yX,-yY,-yZ,-y] are obtained.
    DotP_x= Data2D_x*Data_X_Y_Z_1
    DotP_y= Data2D_y*Data_X_Y_Z_1
    
    #Fill in the values obtained into matrix A.
    #Firstly, 3D homogeneous points are filled into A matrix for 491x4 order.
    #The [X,Y,Z,1] is fills 491*4 places of matrix A, 491:982 rows are filled with 0's.
    #The 2 seperate matrices are concatenated to form Data_L*2 rows of A matrix. 
    AZeros= np.zeros((491,4))
    MergeX= np.concatenate((Data_X_Y_Z_1,AZeros),axis=0)
    

    
    #MergeX and MergeY obtained from filling homogeneous coordinates into A matrix are combined.
    A=np.append(MergeX,MergeY,axis=1)
    
    #The last 4 columns of A matrix are filled with dat products obtained from multiplication of 2D and 3D Data Points.
    #[-xX,-xY,-xZ,-x] fills the first 491 rows corresponding to last 4 columns of A matrix.
    #[-yX,-yY,-yZ,-y] fills the last 491 rows corresponding to last 4 columns of A matrix.
    MergeDotP= np.concatenate((DotP_x,DotP_y),axis=0)
    
    #MergeDotP matrix obatined and A matrix filled with 3D homogeneous points are combined.
    #Matrix obtained is of the order Data_L*2,12 i.e. 982x12.
    A=np.append(A,MergeDotP,axis=1)
    
    #Compute the eigenvalues and eigen vectors of A'A
    #Index of the minimum eigenvalue is computed using minimum eigenvalue. 
    E,V = np.linalg.eig(A.transpose().dot(A))
    
    #extract the estimated parameter as the eigenvector corresponding
    #to the smallest eigenvalue. In this case this corresponds to the col
    #vector of V indexed using the index of the minimum eigenvalue in E.
    Ind = V[:,np.argmin(E)]
    
    #construct the 3x4 camera matrix and inserting in groups of 4
    #the eigenvalues
    P = np.zeros((3,4))
    P[0:] = Ind[0:4]
    P[1:] = Ind[4:8]
    P[2:] = Ind[8:]
    print(P)

    return P

    
def visualiseCameraCalibration3D(data,P):
    
    #3D Data points are extracted from the given Data matrix i.e. X,Y,Z and converted to homogeneous points.
    #by appending 1 and converting the matrix to order 491x4.
    Data_X_Y_Z= data[::,0:3]
    ones_X_Y_Z= np.ones((491,1))
    Data_X_Y_Z_1= np.append(Data_X_Y_Z,ones_X_Y_Z,axis=1)
    #[X,Y,Z,1] homogeneous points are obtained.
    
    #reprojection matrix is computed by performing dot product multiplication 
    #of Camera projection matrix and transpose of homogeneous 3D points.
    #Resultant matrix is again transposed.
    Data3DR= P.dot(Data_X_Y_Z_1.transpose())
    Res3D= Data3DR.transpose()

    #The resultant reprojected matrix x and y coordinates is then divided by its w cordinates
    Res3D[:,0] = Res3D[:,0]/Res3D[:,2]
    Res3D[:,1] = Res3D[:,1]/Res3D[:,2]

    #plot the reprojected 2D vs the 2D image points.
    #blue color is used plot the reprjected matrix.
    #red color is the 2D image points.
    fig = plt.figure()
    ax=fig.gca()
    ax.plot(Res3D[:,0],Res3D[:,1], 'b.')
    ax.plot(data[:,3],data[:,4],'r.')
    plt.show()
    

def evaluateCameraCalibration3D(data,P):
    #3D Data points are extracted from the given Data matrix i.e. X,Y,Z and converted to homogeneous points.
    #by appending 1 and converting the matrix to order 491x4.
    Data_X_Y_Z= data[::,0:3]
    ones_X_Y_Z= np.ones((491,1))
    Data_X_Y_Z_1= np.append(Data_X_Y_Z,ones_X_Y_Z,axis=1)
    #[X,Y,Z,1] homogeneous points are obtained.
    
    #reprojection matrix is computed by performing dot product multiplication 
    #of Camera projection matrix and transpose of homogeneous 3D points.
    #Resultant matrix is again transposed.
    Data3DR= P.dot(Data_X_Y_Z_1.transpose())
    Res3D= Data3DR.transpose()
    
    Data2D_x_y= data[::,3:5]
    
    #Convert 2DData into homogeneous [x,y,1]
    Data2DH= Data2D_x_y*-1
    onesH= np.ones((491,1))
    Data2DH= np.append(Data2DH,onesH,axis=1)
    
    # distacne between reprojected and 2DData points.
    AvgDist = np.subtract(Data2DH,Res3D)
    print("Average distance between the reprojected and 2D image matrices is: " + str(np.mean(AvgDist)))
	
	# Variance between reprojected and 2DData points. 
    print("Standard deviation/variance of measured 2D image points: " + str(np.std(Data2DH)))
    print("Standard deviation/variance of re-projected 2D points: " + str(np.std(Res3D)))

	# Maximum distance between reprojected and 2DData points.
    print("Maximum distance between the two reprojected and 2D image matrices is: " + str(np.abs(np.max(AvgDist))))

	# Minimum distance between reprojected and 2DData points.
    print("Minimum distance between the two reprojected and 2D image matrices is: " + str(np.abs(np.min(AvgDist))))

 
def main():
    #Given 2D image projections is loaded.
    data = np.loadtxt('data.txt')
    
    #function that computes camera calibration.
    P = calibrateCamera3D(data)
    
    #function plots the orginal data on a graph comapirs it with reprojected points.
    visualiseCameraCalibration3D(data, P)
    
    #Evalutes the mean, variance maximum and minum distance between the two mattrix.
    evaluateCameraCalibration3D(data, P)


main()


#Note:
#The following analysis is computed by referring to the lecture video available in the below link.
#Source: https://www.youtube.com/watch?v=HoBKG82A9xs 
#The lecture video provides how A matrix is computed in camera calibration.