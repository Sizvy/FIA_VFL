import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Load passive client's features and labels (from active client)
client_features = np.load('splitted_data/client_2_train.npy')
client_labels = np.load('splitted_data/client_1_train_labels.npy')

# Calculate correlation for each feature using SPEARMAN (like VFL split)
correlations = []
p_values = []

for i in range(client_features.shape[1]):
    corr, pval = spearmanr(client_features[:, i], client_labels)
    correlations.append(corr)
    p_values.append(pval)

# Create a DataFrame with the results
corr_df = pd.DataFrame({
    'feature_index': range(client_features.shape[1]),
    'correlation': correlations,
    'p_value': p_values,
    'abs_correlation': np.abs(correlations)  # Absolute value for ranking
})

# Sort by absolute correlation value (most correlated first)
corr_df = corr_df.sort_values('abs_correlation', ascending=False)

# Display top correlated features
print("Top correlated features for passive client (Spearman):")
print(corr_df.head(20))

# Save the results to CSV
corr_df.to_csv('splitted_data/passive_client_feature_correlations_spearman.csv', index=False)

# Select top N most correlated features
top_n = 10  # Change this to select more/less features
top_features = corr_df['feature_index'].values[:top_n]
print(f"\nTop {top_n} most correlated feature indices (Spearman):", top_features)
