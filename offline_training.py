import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pickle
import pyarrow.parquet as pq


from sklearn.preprocessing import StandardScaler
from ast import literal_eval  # Import literal_eval to safely evaluate string literals
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
from tensorflow.keras.models import Model

from data_preparation import process_matrices
from data_process import split_data
from model_eval import calculate_mse, calculate_mae, calculate_rmse, calculate_r2, get_final_losses, plot_loss

df1 = pd.read_parquet('data.parquet')

with open('data1.pkl', 'rb') as f:
    df2 = pickle.load(f)


df = pd.concat([df1, df2], ignore_index=True)

# Write the merged DataFrame to a new Parquet file
df.to_parquet('VLC.parquet')
df = pd.read_parquet('VLC.parquet')

# Process 'H' column
flattened_matrices_H, normalized_matrices_H = process_matrices(df, 'H')
# Process 'label/W' column
flattened_matrices_W, normalized_matrices_W = process_matrices(df, 'label')


# Create a new DataFrame of processed data 
df3 = pd.DataFrame({
    'H' : df['H'],
    'H_f': flattened_matrices_H,
    # 'H_N' : normalized_matrices_H,
    'label' : df['label'],
    'label_f': flattened_matrices_W ,
    # 'label_N' : normalized_matrices_W
})

# # Apply the function to all rows in the 'H_N' column and create a new column 'H_f'
# df3['H_s'] = df1['H_N'].apply(string_operation)
# df3['label_s']=df1['label_N'].apply(string_operation)


X_train_reshaped, X_test_reshaped,x_test_scaled, y_train, y_train_scaled, y_test, y_test_scaled = split_data(df3, 'H_f', 'label_f')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(24, activation='linear')
])

# Train the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model with early stopping
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(x_test_scaled)

# evaluate model
mse = calculate_mse(y_test, predictions)
mae = calculate_mae(y_test, predictions)
rmse = calculate_rmse(y_test, predictions)
r2 = calculate_r2(y_test, predictions)
print("Mean Squared Error (MSE) on Test Set:", mse)
print("Mean Absolute Error (MAE) on Test Set:", mae)
print("Root Mean Squared Error (RMSE) on Test Set:", rmse)
print("R-squared (R2) on Test Set:", r2)

final_train_loss, final_val_loss = get_final_losses(history)
print("Final Training Loss:", final_train_loss)
print("Final Validation Loss:", final_val_loss)

plot_loss(history)


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


def rate_per_user (H,W):
    
    for n in range(len(Pn)):
        p = Pn[n] * np.ones(M)
        Ps = Pn[n] * np.sum(H, axis=0) #Calculates the total power received at each user by multiplying the transmitted power Pn(n) with the channel matrix H and summing over all transmitters.
        sigma = np.zeros(K) #%Initializes a vector sigma to store noise power values for each user.
        for i in range(K): #Starts a loop over the users.
            sigma[i] = 2 * ee * Ps[i] * B + 2 * ee * rho_k * xi * A_k * 2 * np.pi * (1 - np.cos(theta_c_k)) * B + i_amp ** 2 * B   
    
    rate_OLP = np.zeros((len(Pn),1))
    SINR1 = np.zeros((K, 1))

    for k in range(K):
        h_k = H[:, k]
        W_k = np.delete(W, k, axis=1)
        Sum = np.dot(W_k, W_k.T)
        SINR1[k] = (np.dot(h_k.T, W[:, k]) * np.dot(W[:, k].T, h_k)) / (np.dot(h_k.T, np.dot(Sum, h_k)) + sigma[k])
    
    rate_OLP[n] = (1 / K) * np.sum(B * np.log2(1 + SINR1))
    
    return rate_OLP

r=rate_per_user(df['H'][0],predictions)

# Plot results
plt.figure()
plt.semilogy(P_n_dB, r, 'green', label='OLP')
plt.xlabel('Pn [dBm]')
plt.ylabel('rate [bits/sec]')
plt.grid(True)
plt.legend()
plt.show()
