import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic Page Setup
st.set_page_config(page_title="My Music Clustering App", page_icon="🎧")

st.title("🎵 Amazon Music Cluster Explorer")
st.write("After running my K-Means model, I built this app to see which songs ended up in which groups.")

# 2. Function to load the data I exported from main.py
@st.cache_data
def get_data():
    # Loading the final CSV that main.py produced
    data = pd.read_csv('Final_Clustered_Music_Data.csv')
    return data

try:
    df = get_data()

    # 3. Sidebar for user interaction
    st.sidebar.header("Navigation")
    st.sidebar.write("Choose a cluster to see its musical profile.")
    
    # Let the user pick a cluster number
    selected_cluster = st.sidebar.selectbox("Select Cluster Number:", sorted(df['cluster_label'].unique()))

    # 4. Defining the 'Story' for each cluster based on my analysis
    # I manually mapped these after looking at the heatmap results
    if selected_cluster == 0:
        mood = "🌞 **Happy & Danceable** (High Valence/Energy)"
    elif selected_cluster == 1:
        mood = "🎹 **Instrumental & Study** (High Instrumentalness)"
    elif selected_cluster == 2:
        mood = "🎙️ **Talk / Podcasts** (High Speechiness/Short Duration)"
    elif selected_cluster == 3:
        mood = "🎸 **Acoustic & Relaxing** (High Acousticness)"
    else:
        mood = "⚡ **High Intensity / Rock** (High Tempo/Loudness)"

    st.subheader(f"Results for Cluster {selected_cluster}")
    st.success(f"**Musical Profile:** {mood}")

    # 5. Displaying the songs
    # Filtering the dataframe based on the user's choice
    filtered_songs = df[df['cluster_label'] == selected_cluster]
    
    st.write(f"I found {len(filtered_songs)} songs in this group. Here are the top tracks:")
    # Only showing relevant columns for the user to read
    st.table(filtered_songs[['name_song', 'name_artists', 'genres']].head(20))

    # 6. Visualizations Section (Bar Chart and Box Plot)
    st.write("---")
    st.write("### 📊 Cluster Analytics")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Average Audio Features**")
        # Bar chart for averages
        avg_features = filtered_songs[['danceability', 'energy', 'acousticness', 'valence']].mean()
        st.bar_chart(avg_features)

    with col2:
        st.write("**Energy Distribution (Spread)**")
        # Creating a boxplot to show the 'spread' of energy across all clusters
        # This helps show how distinct the selected cluster is from others
        fig, ax = plt.subplots()
        sns.boxplot(x='cluster_label', y='energy', data=df, palette='Set3', ax=ax)
        
        # Adding a title to make it clear for the demo
        plt.title(f"Comparing Cluster {selected_cluster} Energy to Others")
        st.pyplot(fig)

except FileNotFoundError:
    st.error("Wait! I can't find the 'Final_Clustered_Music_Data.csv' file. Did I run main.py yet?")
except Exception as e:
    st.error(f"Something went wrong: {e}")

# Footer for my project
st.markdown("---")
st.caption("Developed as part of the Amazon Music Clustering Project | Unsupervised Learning")

