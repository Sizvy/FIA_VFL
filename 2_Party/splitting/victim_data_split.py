import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--keep_target_feature', action='store_true',
                    help='Keep the target feature in victim data')
args = parser.parse_args()

os.makedirs('../splitted_data', exist_ok=True)
victim_data = np.load('../shadow_model_data/victim_data_initial.npy')
X = victim_data[:, :-1]
y = victim_data[:, -1]

# Split the dataset into training, validation, and testing sets (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

client1_features = 10
target_feature_idx = -9

client2_set_train = X_train[:, :client1_features]
client2_set_val = X_val[:, :client1_features]
client2_set_test = X_test[:, :client1_features]

if not args.keep_target_feature:
    client2_set_train =  np.delete(client2_set_train, target_feature_idx, axis=1)
    client2_set_val = np.delete(client2_set_val, target_feature_idx, axis=1)
    client2_set_test = np.delete(client2_set_test, target_feature_idx, axis=1)

    client2_set_train = np.delete(client2_set_train, target_feature_idx, axis=1)
    client2_set_val = np.delete(client2_set_val, target_feature_idx, axis=1)
    client2_set_test = np.delete(client2_set_test, target_feature_idx, axis=1)

np.save('../splitted_data/client_2_train.npy', client2_set_train)
np.save('../splitted_data/client_2_val.npy', client2_set_val)
np.save('../splitted_data/client_2_test.npy', client2_set_test)

np.save('../splitted_data/client_1_train.npy', X_train[:, client1_features:])
np.save('../splitted_data/client_1_val.npy', X_val[:, client1_features:])
np.save('../splitted_data/client_1_test.npy', X_test[:, client1_features:])

# Save the labels for the active client (Client 1)
np.save('../splitted_data/client_1_train_labels.npy', y_train)
np.save('../splitted_data/client_1_val_labels.npy', y_val)
np.save('../splitted_data/client_1_test_labels.npy', y_test)

print("Client 2 feature shape:", client2_set_train.shape)
