import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- PROJECT SETTINGS ---
# Making sure the file path matches my folder
DATA_FILE = 'single_genre_artists.csv'

# Selecting features that actually describe the sound of the music
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

# Columns to keep for identifying the songs later
METADATA = ['id_songs', 'name_song', 'name_artists', 'genres']

def prepare_data(file_path):
    """
    Step 1: Load data, handle missing values, and scale features.
    Scaling is crucial because K-Means uses distance (Euclidean).
    """
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for missing values and fill with the average (Mean Imputation)
    if df[AUDIO_FEATURES].isnull().values.any():
        print("Filling missing values with feature means...")
        df[AUDIO_FEATURES] = df[AUDIO_FEATURES].fillna(df[AUDIO_FEATURES].mean())
    
    # Using StandardScaler to give all features a mean of 0 and variance of 1
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[AUDIO_FEATURES])
    
    return df, scaled_array

def find_best_clusters(scaled_data):
    """
    Step 2: Use the Elbow Method to find the optimal 'k'.
    (I ran this and found 5 clusters to be the 'elbow' point).
    """
    inertia = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    # Plotting the results to visualize the 'Elbow'
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method to find Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Error)')
    plt.show()

def build_final_model(df, scaled_data, k=5):
    """
    Step 3: Run the final K-Means model and evaluate results.
    """
    print(f"Fitting K-Means with {k} clusters...")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster_label'] = model.fit_predict(scaled_data)
    
    # Silhouette score helps us see how 'tight' our clusters are
    score = silhouette_score(scaled_data, df['cluster_label'])
    print(f"Evaluation: Silhouette Score = {score:.4f}")
    
    return df

def create_visualizations(df, scaled_data):
    """
    Step 4: Visualize the clusters using PCA and Heatmaps.
    """
    print("Generating visualizations...")
    
    # Using PCA to squeeze 10 features into 2D so we can plot them
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(scaled_data)
    df['pca_x'] = pca_points[:, 0]
    df['pca_y'] = pca_points[:, 1]

    # Cluster Scatter Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster_label', palette='bright')
    plt.title('Amazon Music Song Clusters (PCA Visualization)')
    plt.savefig('pca_cluster_chart.png')
    plt.show()

    # Heatmap of Audio Features
    # This helps me interpret what each cluster represents (e.g. 'Chill' vs 'Party')
    cluster_profile = df.groupby('cluster_label')[AUDIO_FEATURES].mean()
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profile, annot=True, cmap='YlGnBu')
    plt.title('Average Audio Characteristics per Cluster')
    plt.savefig('cluster_feature_heatmap.png')
    plt.show()

# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    try:
        # Phase 1: Preprocessing
        music_df, scaled_matrix = prepare_data(DATA_FILE)
        
        # Phase 2: Finding K (Optional: Uncomment to see Elbow Plot again)
        # find_best_clusters(scaled_matrix)
        
        # Phase 3: Final Clustering (using k=5 from our analysis)
        music_df = build_final_model(music_df, scaled_matrix, k=5)
        
        # Phase 4: Visualization & Interpretation
        create_visualizations(music_df, scaled_matrix)
        
        # Phase 5: Exporting results for the evaluation report
        output_name = 'Final_Clustered_Music_Data.csv'
        music_df.to_csv(output_name, index=False)
        
        print("-" * 30)
        print(f"SUCCESS: Project files generated!")
        print(f"1. CSV: {output_name}")
        print(f"2. Visuals: Saved as .png files")
        print("-" * 30)

    except FileNotFoundError:
        print(f"ERROR: Could not find '{DATA_FILE}'. Please check your folder.")
    except Exception as e:
        print(f"ERROR: Something went wrong: {e}")
