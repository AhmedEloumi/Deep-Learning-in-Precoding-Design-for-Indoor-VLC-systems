from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np 
import matplotlib.pyplot as plt

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

# Define a function to calculate Mean Squared Error (MSE)
def calculate_mse(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mse = round(mse*100,3)
    return mse

# Define a function to calculate Mean Absolute Error (MAE)
def calculate_mae(true_values, predicted_values):
    mae= mean_absolute_error(true_values, predicted_values)
    mae= round(mae*100,3)
    return mae

# Define a function to calculate Root Mean Squared Error (RMSE)
def calculate_rmse(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    rmse = round(rmse*100,3)
    return rmse

# Define a function to calculate R-squared (R2)
def calculate_r2(true_values, predicted_values):
    r2 = r2_score(true_values, predicted_values)
    r2 =round(r2,3)
    return r2

def get_final_losses(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    final_train_loss = train_loss[-1]*100
    final_val_loss = val_loss[-1]*100
    return final_train_loss, final_val_loss

def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()# Load the saved model


# Function to apply the power constraint
def apply_power_constraint(predictions, Pn, K, M):
    for i in range(predictions.shape[0]):  
        for n in range(M): 
        #     start_idx = n * K
        #     end_idx = (n + 1) * K
            wn = predictions[i]  # Weights for transmitter n
            if np.sum(np.abs(wn)) > Pn[n]:
                # Normalize the weights to satisfy the constraint
                wn = wn / np.sum(np.abs(wn)) * Pn[n]
                predictions[i] = wn
    return predictions


# Function to calculate the rate
def calculate_rate_per_user(H, W, K=4, M=6):

    rate_proposed = np.zeros(len(Pn))

    for n in range(len(Pn)):
        p = Pn[n] * np.ones(M)
        Ps = Pn[n] * np.sum(H, axis=0)  # Total power received at each user
        sigma = np.zeros(K)  # Noise power values for each user

        SINR1 = np.zeros((K, 1))
        
        for k in range(K):
            sigma[k] = 2 * ee * Ps[k] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B
            h_k = H[:, k]
            W_k = np.delete(W, k, axis=1)
            Sum = np.dot(W_k, W_k.T)
            SINR1[k] = (np.dot(h_k.T, W[:, k]) * np.dot(W[:, k].T, h_k)) / (np.dot(h_k.T, np.dot(Sum, h_k)) + sigma[k])*Pn[n]
            print(f"Power level index {n}, User {k}, SINR: {SINR1[k]}")  

            rate_proposed[n] = (1/K) * np.sum(B * np.log2(1 + SINR1[k]))
       
        print(f"Power level index {n}, Rate: {rate_proposed[n]}")  

    return rate_proposed
    


# Function to calculate the rate
def calculate_min_SINR(H, W, K=4, M=6):

    min_SINR_proposed = np.zeros(len(Pn))

    for n in range(len(Pn)):
        p = Pn[n] * np.ones(M)
        Ps = Pn[n] * np.sum(H, axis=0)  # Total power received at each user
        sigma = np.zeros(K)  # Noise power values for each user
    
        SINR1 = np.zeros((K, 1))
        
        for k in range(K):
            sigma[k] = 2 * ee * Ps[k] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B
            h_k = H[:, k]
            W_k = np.delete(W, k, axis=1)
            Sum = np.dot(W_k, W_k.T)
            SINR1[k] = (np.dot(h_k.T, W[:, k]) * np.dot(W[:, k].T, h_k)) / (np.dot(h_k.T, np.dot(Sum, h_k)) + sigma[k])*Pn[n]
            print(f"Power level index {n}, User {k}, SINR: {SINR1[k]}")  

        min_SINR_proposed[n] = np.min(SINR1)

    return min_SINR_proposed
    

 # for i in range(K):
        #     sigma[i] = 2 * ee * Ps[i] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B
        #     print(f"Power level index {n}, User {i}, Sigma: {sigma[i]}")  

        # x = np.zeros((M, 1))
        # for ll in range(M):
        #     x[ll] = np.linalg.norm(W[ll, :], ord=1)
        # val = np.max(x)
        # W = W / val * Pn[n]