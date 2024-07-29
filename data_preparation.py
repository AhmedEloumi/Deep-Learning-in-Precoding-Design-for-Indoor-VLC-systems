import numpy as np
import pandas as pd
import random
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

# The roundlist() function recursively rounds each element in a nested list to three decimal places and returns the rounded list.
def roundlist(l):
    new_l=[]
    for i in l :
        i=round(i,2)
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

def process_matrices(df, attribute):

    flattened_matrices = []
    
    for matrix in df[attribute]:
        
        # Flatten the matrix into a 1D array and append it to the list of flattened matrices
        flattened_matrix = matrix.flatten()
        flattened_matrices.append(flattened_matrix)
    
    return flattened_matrices