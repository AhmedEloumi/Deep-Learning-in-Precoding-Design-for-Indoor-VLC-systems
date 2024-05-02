import numpy as np
import pandas as pd
import random
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

# The roundlist() function recursively rounds each element in a nested list to three decimal places and returns the rounded list.
def roundlist(l):
    new_l=[]
    for i in l :
        i=round(i,3)
        new_l.append(i)
    return new_l


# list -> string
def list2string(row):
    # Apply the operation to the row and return the result
    return ' '.join(map(str, row))

# string -> float
def string2float(row):
    # Apply the operation to the row and return the result
    return [float(value) for value in row.split()]

def process_matrices(df, column_name):

    flattened_matrices = []
    normalized_matrices = []
    
    for matrix_string in df[column_name]:
        matrix = literal_eval(matrix_string)
        matrix = np.array(matrix)
        flattened_matrix = matrix.flatten()
        flattened_matrices.append(flattened_matrix)
        
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(flattened_matrix.reshape(-1, 1)).flatten()
        normalized_matrix = roundlist(normalized_matrix)
        normalized_matrices.append(normalized_matrix)
    
    return flattened_matrices, normalized_matrices