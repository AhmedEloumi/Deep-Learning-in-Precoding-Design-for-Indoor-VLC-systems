import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import kron, norm
from R_func import R_func
from cvx_func_bisection import cvx_func_bisection, cvx_func_bisection_2
import pandas as pd
import pickle 
import pyarrow as pa
import pyarrow.parquet as pq

# Implementation of the main code
K = 4  # number of UE
M = 6  # number of transmitter
P_n_dB = np.arange(15, 31, 1)  # Power values in dBm
Pn = 10 ** ((P_n_dB - 30) / 10)  # Power values in linear scale
theta_c_k = 60 * (np.pi / 180)  # Receiver field of view in radians (60 deg -> rad)
q = 1.5  # Refractive index of optical concentrator
B = 1e8  # Bandwidth in Hz
BER = 1e-3  # Bit Error Rate
ee = 1.60217662e-19  # Elementary charge
A_PDk = 1e-4  # Photo Detector area
xi = 10.93  # Ambient light photocurrent
rho_k = 0.4  # Photo Detector responsivity
i_amp = 5e-12  # Preamplifier noise density
m = 1  # mode number of Lambertian emission
A_k = q ** 2 * A_PDk / (np.sin(theta_c_k)) ** 2  # Collection area


def generate_unique_matrices(num_matrices, num_rows, num_columns, mean, std_dev, a=None):
    matrices = []
    while len(matrices) < num_matrices:
        # Generate a matrix of random values distributed according to a Gaussian distribution
        matrix = np.random.normal(loc=mean, scale=std_dev, size=(num_rows, num_columns))
        
        # Clip the values to ensure they are non-negative
        matrix = np.clip(matrix, a_min=0, a_max=a)
        
        # Check if the generated matrix is unique
        if all((matrix != existing).any() for existing in matrices):
            matrices.append(matrix)
    
    return matrices

#transmitters position
# Mean and standard deviation of the values
mean_t = 2.
std_dev_t = 1  # Larger standard deviation for more dispersed values
# Define the dimensions of the matrix
num_matrices_t = 10
num_rows_t = 6
num_columns_t = 3
# Generate unique matrices for transmitters
t_matrices = generate_unique_matrices(num_matrices_t, num_rows_t, num_columns_t, mean_t, std_dev_t, a=5)


#Users position
# Mean and standard deviation of the values
mean_u = 2.15
std_dev_u= 2  # Larger standard deviation for more dispersed values (0.1 for small separation)
# Define the dimensions of the matrix
num_matrices_u = 10
num_rows_u = 4
num_columns_u = 3

# Generate matrices
# Generate unique matrices for users
u_matrices = generate_unique_matrices(num_matrices_u, num_rows_u, num_columns_u, mean_u, std_dev_u, a=None)


# Display the generated matrices
# for i, matrix in enumerate(matrices):
#     print(f"Matrix {i+1}:")
#     print(matrix)
#     print()

for k, t in enumerate(t_matrices) :
    for l, u in enumerate(u_matrices) :

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
                theta = np.arccos(d2 / d1) #It represents the angle between the line connecting the user and the updated transmitter position, the line connecting the user and the original transmitter position. This angle helps in understanding the angular spread or deviation from the direct line between the user and the transmitter.
                if theta <= theta_c_k:
                    H[i, j] = rho_k * A_k / d1 ** 2 * R_func(phi, m) * np.cos(theta)

        # rate_ZF_test = np.zeros(len(Pn))
        rate_OLP = np.zeros(len(Pn))

        for n in range(len(Pn)):
            p = Pn[n] * np.ones(M)
            Ps = Pn[n] * np.sum(H,
                                    axis=0)  # Calculates the total power received at each user by multiplying the transmitted power Pn(n) with the channel matrix H and summing over all transmitters.
            sigma = np.zeros(K)  # %Initializes a vector sigma to store noise power values for each user.
            for i in range(K):  # Starts a loop over the users.
                sigma[i] = 2 * ee * Ps[i] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B

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

        # Load the existing DataFrame from the Parquet file
        table = pq.read_table('data.parquet')
        df = table.to_pandas()

        # Convert the matrices to strings with commas
        H_str = np.array2string(H, separator=',')
        W_str = np.array2string(W, separator=',')
        # t_str= np.array2string(t_matrices, separator=',')
        # u_str= np.array2string(u_matrices, separator=',')

        # Initialize an empty DataFrame
        df1 = pd.DataFrame(columns=['H', 'label'])

        df1 = df1._append({'H': H_str, 'label': W_str}, ignore_index=True)

        # Concatenate the existing DataFrame with the new DataFrame of data
        df = pd.concat([df, df1], ignore_index=True)

        # Convert the DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(df)

        # Write the updated DataFrame to the Parquet file
        pq.write_table(table, 'data.parquet')

print(df.shape)