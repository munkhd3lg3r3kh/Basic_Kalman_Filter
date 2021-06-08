from numpy import *
from numpy.linalg import inv
import numpy as np
from matplotlib import pyplot as mp

#time step of mobile movement 
dt = 0.1 
# Initialization of state matrices 
X = np.array([[0.0], [0.0], [0.1], [0.1]]) 
P = np.eye(4) 
print(P)
mp.plot(P)
mp.show()
A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) 
Q = np.eye(X.shape[0]) 
B = np.eye(X.shape[0]) 
U = np.zeros((X.shape[0],1))
# Measurement matrices 
Y = np.array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] + abs(np.random.randn(1)[0])]]) 
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) 
R = np.eye(Y.shape[0]) 
# Number of iterations in Kalman Filter 
N_iter = 50 

def kf_predict(X, P, A, Q, B, U): 
    X = dot(A, X) + dot(B, U) 
    P = dot(A, dot(P, A.T)) + Q 
    return(X,P)

def kf_update(X, P, Y, H, R): 
    IM = dot(H, X) 
    IS = R + dot(H, dot(P, H.T)) 
    K = dot(P, dot(H.T, inv(IS))) 
    X = X + dot(K, (Y-IM)) 
    P = P - dot(K, dot(IS, K.T)) 
    LH = gauss_pdf(Y, IM, IS) 
    return (X,P,K,IM,IS,LH) 
    
def gauss_pdf(X, M, S): 
    if M.shape[1] == 1: 
        DX = X - tile(M, X.shape[1]) 
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0) 
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S)) 
        P = exp(-E) 
    elif X.shape[1] == 1: 
        DX = tile(X, M.shape[1])- M 
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0) 
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S)) 
        P = exp(-E) 
    else: 
        DX = X-M 
        E = 0.5 * dot(DX.T, dot(inv(S), DX)) 
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S)) 
        P = exp(-E) 
    return (P[0],E[0])

if __name__ == "__main__" :
    # Applying the Kalman Filter 
    for i in range(0, N_iter): 
        (X, P) = kf_predict(X, P, A, Q, B, U) 
        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R) 
        Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] + abs(0.1 * np.random.randn(1)[0])]])
    
    #mp.plot(Y)
    #mp.plot(X)
    print(P)
    mp.plot(P)
    mp.show()