import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs('../splitted_data', exist_ok=True)
victim_data = np.load('../shadow_model_data/victim_data.npy')
X = victim_data[:, :-1]
y = victim_data[:, -1]

# Split the dataset into training, validation, and testing sets (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fixed number of features for Client 1
client1_features = 10

# Client 1 gets first 10 features
np.save('../splitted_data/client_1_train.npy', X_train[:, :client1_features])
np.save('../splitted_data/client_1_val.npy', X_val[:, :client1_features])
np.save('../splitted_data/client_1_test.npy', X_test[:, :client1_features])

# Client 2 gets all remaining features
np.save('../splitted_data/client_2_train.npy', X_train[:, client1_features:])
np.save('../splitted_data/client_2_val.npy', X_val[:, client1_features:])
np.save('../splitted_data/client_2_test.npy', X_test[:, client1_features:])

# Save the labels for the active client (Client 1)
np.save('../splitted_data/client_1_train_labels.npy', y_train)
np.save('../splitted_data/client_1_val_labels.npy', y_val)
np.save('../splitted_data/client_1_test_labels.npy', y_test)

print("Client 1 gets features: 0 to", client1_features - 1)
print("Client 2 gets features:", client1_features, "to", X_train.shape[1] - 1)
