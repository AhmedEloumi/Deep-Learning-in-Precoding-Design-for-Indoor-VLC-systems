from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np 
import matplotlib.pyplot as plt


# Define a function to calculate Mean Squared Error (MSE)
def calculate_mse(true_values, predicted_values):
    return mean_squared_error(true_values, predicted_values)

# Define a function to calculate Mean Absolute Error (MAE)
def calculate_mae(true_values, predicted_values):
    return mean_absolute_error(true_values, predicted_values)

# Define a function to calculate Root Mean Squared Error (RMSE)
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))

# Define a function to calculate R-squared (R2)
def calculate_r2(true_values, predicted_values):
    return r2_score(true_values, predicted_values)


def get_final_losses(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    return final_train_loss, final_val_loss

def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

