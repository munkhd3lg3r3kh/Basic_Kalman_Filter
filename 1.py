import matplotlib.pyplot as plt
import csv
import numpy as np

ref = []
sensor = []

xhat = []
xest = []

cov = []
covhat = []
index = 0
xest[0] = 0
cov[0] = 0
t = 0
A = None
B = None
C = None
R = 10

def initCons():
    A = np.array([[1, t, t*t/2], [0, 1, t], [0, 0, 1]])
    invA = np.linalg.inv(A)

    B = np.array([[t*t/2], [t], [1]])
    invB = np.linalg.inv(B)

    C = np.array([1,0,0])
    invB = np.linalg.inv(B)

def kalman_filter(i, ut, zt):
    xhat[i] = A*xest[i] + B*ut
    covhat[i] = A*cov[i] + R

    K = (covhat[i]*C)/(C*covhat[i]+R)

    xest[i+1] = xhat[i] + K*(zt-C*xhat)
    cov[i+1] = (np.eye(3)-K*C)*covhat[i]

def reader():
    with open("example.csv", newline='') as File :
        reader = csv.reader(File)
        for row in reader : 
            if row[0] != "":
                index += 1            
                ref.append(float(row[1]))
                sensor.append(float(row[2]))
if __name__ == "__main__" :
    reader()
    ut = 0.1
    for i in range(index):
        initCons()
        xhat[i] = A*xest[i] + B*ut
        covhat[i] = A*cov[i] + R

        K = (covhat[i]*C)/(C*covhat[i]+R)

        xest[i+1] = xhat[i] + K*(sensor[i]-C*xhat)
        cov[i+1] = (np.eye(3)-K*C)*covhat[i]
        
        t += 1
    