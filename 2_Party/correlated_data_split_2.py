import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# Create directories - changed to 'splitted_data_weak' to indicate low-correlation features
os.makedirs('splitted_data_weak', exist_ok=True)
# Load the dataset
data = pd.read_csv('../../datasets/drive_cleaned.csv', header=None)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 1: Find bottom 20 least correlated features
correlations = []
for i in range(X.shape[1]):
    corr, _ = spearmanr(X[:, i], y)
    correlations.append(abs(corr))  # Use absolute correlation

# Get indices of bottom 20 features (least correlated)
top_n = 20
bottom_features = np.argsort(correlations)[:top_n]  # Indices of bottom 20 features

print("Bottom 20 features by correlation:")
print(bottom_features)
print("Correlation values:", [correlations[i] for i in bottom_features])

# Select only the bottom features
X_bottom = X[:, bottom_features]

# Step 2: Split the dataset with selected features (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X_bottom, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Distribute features among 2 clients
num_clients = 2
features_per_client = X_train.shape[1] // num_clients

for i in range(num_clients):
    start_idx = i * features_per_client
    end_idx = (i + 1) * features_per_client if i != num_clients - 1 else X_train.shape[1]

    # Save the features for each client in .npy format
    np.save(f'splitted_data_weak/client_{i+1}_train.npy', X_train[:, start_idx:end_idx])
    np.save(f'splitted_data_weak/client_{i+1}_val.npy', X_val[:, start_idx:end_idx])
    np.save(f'splitted_data_weak/client_{i+1}_test.npy', X_test[:, start_idx:end_idx])

    # Save the features for each client in .csv format
    pd.DataFrame(X_train[:, start_idx:end_idx]).to_csv(f'splitted_data_weak/client_{i+1}_train.csv', index=False, header=False)
    pd.DataFrame(X_val[:, start_idx:end_idx]).to_csv(f'splitted_data_weak/client_{i+1}_val.csv', index=False, header=False)
    pd.DataFrame(X_test[:, start_idx:end_idx]).to_csv(f'splitted_data_weak/client_{i+1}_test.csv', index=False, header=False)

# Save the labels for the active client (Client 1) in .npy format
np.save('splitted_data_weak/client_1_train_labels.npy', y_train)
np.save('splitted_data_weak/client_1_val_labels.npy', y_val)
np.save('splitted_data_weak/client_1_test_labels.npy', y_test)

# Save the labels for the active client (Client 1) in .csv format
pd.DataFrame(y_train, columns=['label']).to_csv('splitted_data_weak/client_1_train_labels.csv', index=False)
pd.DataFrame(y_val, columns=['label']).to_csv('splitted_data_weak/client_1_val_labels.csv', index=False)
pd.DataFrame(y_test, columns=['label']).to_csv('splitted_data_weak/client_1_test_labels.csv', index=False)

print("\nDataset with least correlated features split and saved successfully in splitted_data_weak folder")
print(f"Total features selected: {X_train.shape[1]}")
print(f"Features per client: {features_per_client}")
print("Client 1 gets features:", bottom_features[:features_per_client])
print("Client 2 gets features:", bottom_features[features_per_client:])

# Save feature selection info
with open('splitted_data_weak/feature_selection_info.txt', 'w') as f:
    f.write("Bottom 20 Feature Indices:\n")
    f.write(str(bottom_features.tolist()) + "\n\n")
    f.write("Correlation Values:\n")
    f.write(str([correlations[i] for i in bottom_features]) + "\n\n")
    f.write(f"Client 1 features: {bottom_features[:features_per_client].tolist()}\n")
    f.write(f"Client 2 features: {bottom_features[features_per_client:].tolist()}\n")
