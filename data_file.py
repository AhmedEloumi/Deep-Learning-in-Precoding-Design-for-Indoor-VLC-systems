# Creatre pickle file to store the data
# Pickle file
import pickle
import pandas as pd
# Convertir les matrices en chaînes de caractères avec des virgules
H_str = np.array2string(H, separator=',')
W_str = np.array2string(W, separator=',')

# Initialize an empty DataFrame
df1 = pd.DataFrame(columns=['H', 'label'])

# Ajouter les matrices converties au DataFrame
df1 = df1._append({'H': H_str, 'label': W_str}, ignore_index=True)

# Sauvegarder le DataFrame dans un fichier pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(df1, f)

# Charger le DataFrame à partir du fichier pickle
with open('data.pkl', 'rb') as f:
    df = pickle.load(f)

# Afficher les informations sur la forme et les premières lignes du DataFrame
print(df.shape)
print(df.head())


# CSV file
# import pandas as pd

# # Convertir les matrices en chaînes de caractères avec des virgules
# H_str = np.array2string(H, separator=',')
# W_str = np.array2string(W, separator=',')

# # Initialize an empty DataFrame
# df1 = pd.DataFrame(columns=['H', 'label'])

# # Ajouter les matrices converties au DataFrame
# df1 = df1._append({'H': H_str, 'label': W_str}, ignore_index=True)

# # Sauvegarder le DataFrame dans un fichier CSV
# df1.to_csv('db.csv', index=False)  # Ne pas inclure l'index dans le fichier CSV

# # Lire le fichier CSV dans un DataFrame
# df = pd.read_csv('db.csv')

# # Afficher les informations sur la forme et les premières lignes du DataFrame
# print(df.shape)
# print(df.head())



# Initialize an empty DataFrame
df1 = pd.DataFrame(columns=['T','UE','H', 'label'])

# Add the converted matrices to the DataFrame
df1 = df1._append({'T':t_str, 'UE': u_str, 'H': H_str, 'label': W_str}, ignore_index=True)

# Save the DataFrame to a Parquet file
table = pa.Table.from_pandas(df1)
pq.write_table(table, 'db.parquet')

# Load the DataFrame from the Parquet file
table = pq.read_table('db.parquet')
df = table.to_pandas()

# Display information about the shape and the first few rows of the DataFrame
print(df.shape)
print(df.head())