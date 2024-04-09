#Librairies
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import kron, norm
from R_func import R_func
from cvx_func_bisection import cvx_func_bisection, cvx_func_bisection_2

# Implementation of the main code
K = 4 # number of UE
M = 6  # number of transmitter
P_n_dB = np.arange(15, 31, 1)  # Power values in dBm
Pn = 10 ** ((P_n_dB - 30) / 10)  # Power values in linear scale
theta_c_k = 60 * (np.pi/180)  # Receiver field of view in radians (60 deg -> rad)
q = 1.5  # Refractive index of optical concentrator
B = 1e8 # Bandwidth in Hz
BER = 1e-3  # Bit Error Rate
ee = 1.60217662e-19  # Elementary charge
A_PDk = 1e-4  # Photo Detector area
xi = 10.93  # Ambient light photocurrent
rho_k = 0.4  # Photo Detector responsivity
i_amp = 5e-12  # Preamplifier noise density
m = 1  # mode number of Lambertian emission
A_k = q ** 2 * A_PDk / (np.sin(theta_c_k)) ** 2  # Collection area

#transmitters position
t = np.array([[1, 1, 0],
              [4, 1, 0],
              [1, 2.5, 0],
              [4, 2.5, 0],
              [1, 4, 0],
              [4, 4, 0]])

# Define the positions of users based on Table 3. UE with small separation
#For UE with small separation
u_s= np.array([
    [2.05, 2.20, 2.15],
    [2.05, 2.40, 2.15],
    [2.05, 2.60, 2.15],
    [2.05, 2.80, 2.15]
])

# For UE with big separation
u_b = np.array([
    [2.05, 1.60, 2.15],
    [2.15, 4.10, 2.15],
    [3.5, 3.50, 2.50],
    [4.2, 4.20, 2.50]
])

u=u_s


# Initialize channel matrix H with zeros to store channel informations
H = np.zeros((M, K))
# Calculate channel matrix H
for i in range(M): #Starts a loop over the transmitters.
    for j in range(K): #Starts a nested loop over the users.
        d1 = np.linalg.norm(u[j, :] - t[i, :]) #Calculates the Euclidean distance between the j-th user and the i-th transmitter.
        tt = t[i, :].copy() #Updates the z-coordinate of the transmitter position for distance calculation.
        tt[2] = u[j, 2]
        d2 = np.linalg.norm(u[j, :] - tt) #Calculates the Euclidean distance between the j-th user and the updated transmitter position.
        phi = np.arcsin(d2 / d1) #It represents the angle between the line connecting the user and the updated transmitter position and the horizontal plane.
        theta = np.arcsin(d2 / d1) #It represents the angle between the line connecting the user and the updated transmitter position, the line connecting the user and the original transmitter position. This angle helps in understanding the angular spread or deviation from the direct line between the user and the transmitter.
        if theta <= theta_c_k:
            H[i, j] = rho_k * A_k / d1 ** 2 * R_func(phi, m) * np.cos(theta)

# Initialize rate vectors
rate_ZF = np.zeros((len(Pn),1))
rate_OLP = np.zeros((len(Pn),1))

# Initialize rate vectors
rate_ZF = np.zeros(len(Pn))
# rate_ZF_test = np.zeros(len(Pn))
rate_OLP = np.zeros(len(Pn))

for n in range(len(Pn)):
    p = Pn[n] * np.ones(M)
    Ps = Pn[n] * np.sum(H, axis=0) #Calculates the total power received at each user by multiplying the transmitted power Pn(n) with the channel matrix H and summing over all transmitters.
    sigma = np.zeros(K) #%Initializes a vector sigma to store noise power values for each user.
    for i in range(K): #Starts a loop over the users.
        sigma[i] = 2 * ee * Ps[i] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B    

    # Zero Forcing 1
    C = np.dot(H, np.linalg.inv(np.dot(H.T, H)))
    A = np.dot(np.abs(C), np.diag(np.sqrt(sigma)))
    vec = Pn[n] / np.dot(A, np.ones((K, 1)))
    mu = np.min(vec)
    rate_ZF[n] = B  * np.sum(np.log2(1 + mu ** 2))

    # Zero Forcing 2
    gamma = mu * np.sqrt(sigma)
    W_ZF = np.dot(C, np.diag(gamma))
    SINR = np.zeros(K)

    for k in range(K):
        h_k = H[:, k]
        W_k = np.delete(W_ZF, k, axis=1)
        Sum = np.dot(W_k, W_k.T)
        SINR[k] = (np.dot(np.dot(h_k.T, W_ZF[:, k]), np.dot(W_ZF[:, k].T, h_k))) / (np.dot(np.dot(h_k.T, Sum), h_k) + sigma[k])
    # rate_ZF_test[n] = B/K  * np.sum(np.log2(1 + SINR))

    # OLP
    W = H
    x = np.zeros((M, 1))
    for ll in range(M):
        x[ll] = np.linalg.norm(W[ll, :], ord=1)
    val = np.max(x)
    W = W / val * Pn[n]
    it = 0
    t_aux = 1e-6
    t_works = 1e-10

    while abs(t_works - t_aux) / t_works >= 1e-3 or it <= 1:
        it += 1
        t_aux = t_works
        t_lower = t_works
        t_upper = 1e5
        tol = 0.01
        while (t_upper - t_lower) / t_lower > tol:
            t_test = (t_upper + t_lower) / 2
            Wz, d, status = cvx_func_bisection_2(t_test, M, K, sigma, p, H)
            if status != "Solved":
                t_upper = t_test
            else:
                W_works = Wz
                t_works = t_test
                t_lower = t_test
        W = W_works

    SINR1 = np.zeros((K, 1))
    for k in range(K):
        h_k = H[:, k]
        W_k = np.delete(W, k, axis=1)
        Sum = np.dot(W_k, W_k.T)
        SINR1[k] = (np.dot(h_k.T, W[:, k]) * np.dot(W[:, k].T, h_k)) / (np.dot(h_k.T, np.dot(Sum, h_k)) + sigma[k])

    rate_OLP[n] = (1 / K) * np.sum(B * np.log2(1 + SINR1))


print (H)
print(W)
# Plot results
plt.figure()
plt.semilogy(P_n_dB, rate_ZF, 'brown', label='Zero Forcing 1')
# plt.semilogy(P_n_dB, rate_ZF_test, 'b', label='Zero Forcing 2')
plt.semilogy(P_n_dB, rate_OLP, 'green', label='OLP')
plt.xlabel('Pn [dBm]')
plt.ylabel('rate [bits/sec]')
plt.grid(True)
plt.legend()
# plt.xlim(15, 30)
# plt.ylim(0, 1e9)
plt.show()