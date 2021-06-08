from matplotlib import pyplot as mp
from numpy.linalg import matrix_power
import numpy as np
t = 0
A = None
B = None
C = None

def initCons():
    global A; global B; global C;
    A = np.array([[1, t, t*t/2], [0, 1, t], [0, 0, 1]])
    invA = np.linalg.inv(A)

    B = np.array([[t*t/2], [t], [1]])

    C = np.array([1,0,0])
    invB = np.linalg.inv(B)

if __name__ == "__main__":
    t+=2
    initCons()
    print(C)
    matrix_power(C,2)
    print(C)
    
