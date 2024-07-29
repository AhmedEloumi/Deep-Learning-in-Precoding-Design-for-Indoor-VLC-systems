import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pickle
import pyarrow.parquet as pq

import time

from sklearn.preprocessing import StandardScaler
from ast import literal_eval  # Import literal_eval to safely evaluate string literals
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
from tensorflow.keras.models import Model

from data_preparation import process_matrices, roundlist 
from data_process import split_data, divide_data
from model_eval import calculate_mse, calculate_mae, calculate_rmse, calculate_r2, get_final_losses, plot_loss
import cvxpy as cp
from scipy.linalg import kron, norm
from R_func import R_func
from cvx_func_bisection import cvx_func_bisection, cvx_func_bisection_2

np.random.seed(42)

with open('dataset.pkl', 'rb') as f:
    df = pickle.load(f)

# Process 'H : channel matrix' column
flattened_matrices_H = process_matrices(df, 'H')
# Process 'W : precoding matrix' column
flattened_matrices_W= process_matrices(df, 'label')

# Create a new DataFrame of processed data 
df3 = pd.DataFrame({
    'H' : df['H'],
    'H_vector': flattened_matrices_H,
    'W' : df['label'],
    'W_vector': flattened_matrices_W ,
})


# X_train, X_train_scaled, X_train_reshaped, X_test, X_test_scaled, X_test_reshaped, y_train, y_train_scaled, y_test, y_test_scaled= split_data(df3, 'H_vector', 'W_vector', test_size=0.1)
X_train, X_train_scaled, X_train_reshaped, X_valid, X_valid_scaled, X_valid_reshaped, \
X_test, X_test_scaled, X_test_reshaped, y_train, y_train_scaled, y_valid, y_valid_scaled, y_test, y_test_scaled = \
    divide_data(df3, 'H_vector', 'W_vector', test_size=0.2, valid_size=0.2, random_state=42, shuffle=True)


# Define callbacks
model_path = "best_model.weights.h5"
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-5, min_lr=1e-5)
epochs=100
# Define a fixed learning rate
lr_schedule = 0.001

# Parameters = (Kernel Size * Number of Channels + Bias) * Number of Filters 
# Parameters = (Input Size + Bias) * Number of Neurons

######## APPLICATION OF 1DCNN ##################
# Define an optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

def CNN_model (input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), output_shape=24) :
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])

    # Compile the model with the optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model

model1= CNN_model()

# # Measure training time
# start_time = time.time()
# # Train the model with callbacks
# history1 = model1.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=32, validation_split=0.2,
#                     callbacks=[checkpointer, reduce_lr, early_stopping])
# end_time = time.time()
# training_time = end_time - start_time
# print(f"Training Time: {training_time} seconds")


# Measure training time
start_time = time.time()

# Train the model with callbacks
history1 = model1.fit(
    X_train_reshaped, y_train, 
    epochs=epochs, 
    batch_size=32, 
    validation_data=(X_valid_reshaped, y_valid),
    callbacks=[checkpointer, reduce_lr, early_stopping]
)

end_time = time.time()
training_time = end_time - start_time

print(f"Training Time: {training_time} seconds")


# Measure prediction time
start_time = time.time()
prediction = model1.predict(X_test_reshaped)
end_time = time.time()
prediction_time = end_time - start_time
print(f"Prediction Time: {prediction_time} seconds")


print(prediction[0])
print("------------------------------------")
print(y_test[0])


r2=calculate_r2(y_test, prediction)
print('R2 :', r2)

# Plotting both sets of loss values in the same figure with different colors
plt.plot(history1.history['loss'], label='Train loss 1D CNN', color='blue')
plt.plot(history1.history['val_loss'], label='Validation loss 1D CNN', color='orange')


# Adding title, labels, and legend
# plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Display the plot
plt.show()


# plot results
final_train_loss, final_val_loss = get_final_losses(history1)
print("Final Training Loss:", final_train_loss)
print("Final Validation Loss:", final_val_loss)
plot_loss(history1)

# # Save the entire model
# model1.save("vlcmodel.h5")
# print("Model saved")
