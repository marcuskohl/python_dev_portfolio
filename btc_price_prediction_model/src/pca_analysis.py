import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def perform_pca(data, n_components=None):
    """Performing PCA on the dataset."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)

    return principal_components, pca

def plot_explained_variance(pca):
    """Plotting the explained variance by each principal component."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    #Loading dataset
    file_path = '../data/processed_BTCUSDT_Hourly.csv'
    btc_data = pd.read_csv(file_path)

    #Dropping non-numeric or non-relevant columns for PCA
    btc_data_processed = btc_data.drop(columns=['datetime'])

    #Performing PCA
    n_components = None
    principal_components, pca = perform_pca(btc_data_processed, n_components)

    #Saving principal components to new CSV file
    principal_components_df = pd.DataFrame(principal_components)
    output_path = '../data/btc_data_pca.csv' 
    principal_components_df.to_csv(output_path, index=False)
    print(f"PCA results saved to {output_path}")

    #Plotting explained variance
    plot_explained_variance(pca)

if __name__ == '__main__':
    main()
    
