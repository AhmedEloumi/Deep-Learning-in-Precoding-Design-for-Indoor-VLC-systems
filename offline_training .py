import pandas as pd
import numpy as np
df=pd.read_csv('db.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head())
# Appliquer la fonction de conversion à chaque élément de la colonne "H"
df['H'] = df['H'].apply(lambda matrix: np.array(matrix).flatten())

# Vérifier le résultat
print(df['H'])