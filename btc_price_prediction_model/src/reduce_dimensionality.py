import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_dimensionality(data, variance_threshold=0.95):
    """Reducing dimensionality of dataset to capture desired percentage of variance."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    pca = PCA()
    pca.fit(data_scaled)
    
    #Calculating number of components that capture the variance_threshold of variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cumulative_variance < variance_threshold).sum() + 1
    
    #Reducing dataset to these components
    pca_reduced = PCA(n_components=n_components)
    principal_components = pca_reduced.fit_transform(data_scaled)
    
    return principal_components, pca_reduced, n_components

def main():
    #Loading dataset
    file_path = '../data/processed_BTCUSDT_Hourly.csv'
    btc_data = pd.read_csv(file_path)
    
    #Dropping 'datetime'
    btc_data_for_pca = btc_data.drop(columns=['datetime'])

    #Reducing dimensionality
    principal_components, pca_reduced, n_components = reduce_dimensionality(btc_data_for_pca)
    
    #Creating DataFrame of the reduced data
    columns = [f'PC{i+1}' for i in range(n_components)]
    btc_data_reduced = pd.DataFrame(principal_components, columns=columns)
    
    #Saving dataset
    output_path = '../data/btc_data_reduced.csv'  
    btc_data_reduced.to_csv(output_path, index=False)
    
    print(f"Reduced dataset with {n_components} components saved to {output_path}")

if __name__ == '__main__':
    main()