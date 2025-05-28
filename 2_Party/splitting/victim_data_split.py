import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create directories
os.makedirs('../splitted_data', exist_ok=True)
victim_data = np.load('../shadow_model_data/victim_data.npy')
# Load the victim data
# victim_data = np.load('../shadow_model_data/victim_data.npy')

# Separate features and labels (assuming last column is label)
X = victim_data[:, :-1]
y = victim_data[:, -1]

# Split the dataset into training, validation, and testing sets (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Distribute features among 2 clients
num_clients = 2
features_per_client = (X_train.shape[1] + num_clients - 1) // num_clients

for i in range(num_clients):
    start_idx = i * features_per_client
    end_idx = min((i + 1) * features_per_client, X_train.shape[1])
    
    # Save the features for each client in .npy format
    np.save(f'../splitted_data/client_{i+1}_train.npy', X_train[:, start_idx:end_idx])
    np.save(f'../splitted_data/client_{i+1}_val.npy', X_val[:, start_idx:end_idx])
    np.save(f'../splitted_data/client_{i+1}_test.npy', X_test[:, start_idx:end_idx])

    # Save the features for each client in .csv format
    pd.DataFrame(X_train[:, start_idx:end_idx]).to_csv(f'../splitted_data/client_{i+1}_train.csv', index=False, header=False)
    pd.DataFrame(X_val[:, start_idx:end_idx]).to_csv(f'../splitted_data/client_{i+1}_val.csv', index=False, header=False)
    pd.DataFrame(X_test[:, start_idx:end_idx]).to_csv(f'../splitted_data/client_{i+1}_test.csv', index=False, header=False)

# Save the labels for the active client (Client 1) in .npy format
np.save('../splitted_data/client_1_train_labels.npy', y_train)
np.save('../splitted_data/client_1_val_labels.npy', y_val)
np.save('../splitted_data/client_1_test_labels.npy', y_test)

# Save the labels for the active client (Client 1) in .csv format
pd.DataFrame(y_train, columns=['label']).to_csv('../splitted_data/client_1_train_labels.csv', index=False)
pd.DataFrame(y_val, columns=['label']).to_csv('../splitted_data/client_1_val_labels.csv', index=False)
pd.DataFrame(y_test, columns=['label']).to_csv('../splitted_data/client_1_test_labels.csv', index=False)

print("Dataset split and saved successfully in both .npy and .csv formats.")
print(f"Total features: {X_train.shape[1]}")
print(f"Features per client: {features_per_client}")
print("Client 1 gets features: 0 to", features_per_client - 1)
print("Client 2 gets features:", features_per_client, "to", X_train.shape[1] - 1)
