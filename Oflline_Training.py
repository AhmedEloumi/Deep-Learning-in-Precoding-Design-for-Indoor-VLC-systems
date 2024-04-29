import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pickle

from sklearn.preprocessing import StandardScaler
from ast import literal_eval  # Import literal_eval to safely evaluate string literals
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping 

# Charger le DataFrame à partir du fichier pickle
with open('data.pkl', 'rb') as f:
    df = pickle.load(f)

# print(df.head())
# print(df.info())


# Initialize lists to store flattened and normalized matrices
flattened_matrices = []
normalized_matrices = []

# Process 'H' column
for matrix_string in df['H']:
    matrix = literal_eval(matrix_string)
    matrix = np.array(matrix)
    flattened_matrix = matrix.flatten()
    flattened_matrices.append(flattened_matrix)
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(flattened_matrix.reshape(-1, 1)).flatten()
    normalized_matrices.append(normalized_matrix)

# Create a new DataFrame with flattened and normalized matrices for 'H'
df1 = pd.DataFrame({
    'H': flattened_matrices,
    'H_N': normalized_matrices
})

# Initialize lists to store flattened and normalized matrices for 'label'
flattened_labels = []
normalized_labels = []

# Process 'label' column
for label_string in df['label']:
    label = literal_eval(label_string)
    label = np.array(label)
    flattened_label = label.flatten()
    flattened_labels.append(flattened_label)
    scaler = StandardScaler()
    normalized_label = scaler.fit_transform(flattened_label.reshape(-1, 1)).flatten()
    normalized_labels.append(normalized_label)

# Create a new DataFrame with flattened and normalized matrices for 'label'
df1['label'] = flattened_labels
df1['label_N'] = normalized_labels

# Display the processed DataFrame
# print(df1.head())

# Extract features (X) and labels (y)
X = np.stack(df1['H'].values)
y = np.stack(df1['label'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

## Print shapes or types to check
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)


# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(24, activation='elu'),  # Adjust the units to match the shape of your target labels
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model with early stopping
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) on test set
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE) on Test Set:", mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE) on Test Set:", mae)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error (RMSE) on Test Set:", rmse)

# Calculate R-squared (R2)
r2 = r2_score(y_test, predictions)
print("R-squared (R2) on Test Set:", r2)


# Get training and validation loss from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Final training and validation loss
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]

# print("Final Training Loss:", final_train_loss)
# print("Final Validation Loss:", final_val_loss)


# # Plot training and validation loss
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Save the trained model
model.save('my_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Example new input matrix
new_H_str = "[[3.43934628e-06,2.98965139e-06,2.57230977e-06,2.19688693e-06], \
             [1.81432094e-06,1.63717571e-06,1.46321509e-06,1.29759393e-06], \
             [5.22166319e-06,5.36835774e-06,5.36835774e-06,5.22166319e-06], \
             [2.43522441e-06,2.48163622e-06,2.48163622e-06,2.43522441e-06], \
             [2.19688693e-06,2.57230977e-06,2.98965139e-06,3.43934628e-06], \
             [1.29759393e-06,1.46321509e-06,1.63717571e-06,1.81432094e-06]]"

new_H = np.array(eval(new_H_str))

# Preprocess the new input matrix (similar to how the training data was preprocessed)
scaler_X = StandardScaler()
new_H_scaled = scaler_X.fit_transform(new_H.flatten().reshape(-1, 1)).flatten()

# Load the saved model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Make predictions using the loaded model
predicted_label_scaled = loaded_model.predict(new_H.reshape(1, -1))

# Inverse transform the predicted label to its original form
predicted_label = scaler_y.inverse_transform(predicted_label_scaled)

# Reshape the predicted label to match the original shape
predicted_label = predicted_label.reshape(new_H.shape[0], new_H.shape[1])

# Print the predicted label matrix
print("Predicted label matrix:")
print(predicted_label)