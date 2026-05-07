# Amazon Music Clustering: Unsupervised Learning Project 🎧
"A hands-on Machine Learning project exploring the Amazon Music dataset. I used K-Means clustering to group songs by audio features like energy and acousticness. Includes data scaling, Elbow Method analysis, and 2D visualization using PCA."

### **Domain:** Music Analytics / Machine Learning  
### **Tools:** `Python` | `Scikit-Learn` | `Pandas` | `Streamlit`

---

## 📌 Project Overview
This project focuses on the automated categorization of songs based on their raw audio characteristics. Using the **Amazon Music dataset**, I implemented an unsupervised learning pipeline to group tracks into 5 distinct musical "moods" or "profiles." This eliminates the need for manual genre tagging and allows for data-driven playlist curation.

---

## 🛠️ Skills & Knowledge Gained
*   **Data Preprocessing:** Handled missing values and implemented `StandardScaler` to normalize features like `duration_ms` and `danceability` for distance-based clustering.
*   **Optimal Cluster Selection:** Used the **Elbow Method** to identify the most meaningful number of clusters ($k=5$).
*   **Advanced Clustering:** Applied **K-Means Clustering** and evaluated the model using the **Silhouette Score**.
*   **Dimensionality Reduction:** Utilized **PCA (Principal Component Analysis)** to visualize 10-dimensional audio data on a 2D scatter plot.
*   **Business Intelligence:** Interpreted cluster centroids to create meaningful "Musical Fingerprints."
*   **Deployment:** Developed a **Streamlit** dashboard to interactively explore the resulting clusters.

---

## 📊 Results & Interpretation
After analyzing the feature centroids (Heatmap), I identified the following 5 clusters:

*   **Cluster 0:** 🌞 **Feel-Good Hits** (High Valence, High Danceability)
*   **Cluster 1:** 🎹 **Instrumental & Chill** (Highest Instrumentalness)
*   **Cluster 2:** 🎙️ **Spoken Word / Podcasts** (Highest Speechiness, Short Duration)
*   **Cluster 3:** 🎸 **Soft Acoustic** (Highest Acousticness, Low Energy)
*   **Cluster 4:** ⚡ **High Intensity** (Highest Energy and Tempo)
Heatmap Screenshot :
<img width="1200" height="600" alt="cluster_feature_heatmap" src="https://github.com/user-attachments/assets/5a04f3e9-08e5-4bf2-91db-bd7755627c8f" />


---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |

| `main.py` | The core machine learning pipeline (Cleaning, Clustering, Visualization). |

| `app.py` | The interactive Streamlit dashboard (Bonus deliverable). |

<img width="947" height="484" alt="streamlit_dashboard" src="https://github.com/user-attachments/assets/ff48b272-ad69-4ccd-a155-3ac10ad641c5" />


 | `single_genre_artists.csv` | The raw dataset used for analysis. |
 
| `Final_Clustered_Music_Data.csv` | The final output with cluster labels assigned. |

| `cluster_visualization.png` | PCA 2D plot showing song distribution. |

<img width="1000" height="700" alt="pca_cluster_chart" src="https://github.com/user-attachments/assets/1a46dd7f-7496-49a0-a1d1-5511607f68a6" />

---

## 🚀 How to Run

1. **Clone the folder and install dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn streamlit
   ```

2. **Run the analysis:**
   ```bash
   python main.py
   ```

3. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 💼 Business Impact
This model can be used to:
*   **Automate Playlist Generation:** Create "Deep Focus" or "Energy Boost" playlists instantly.
*   **Enhance Song Discovery:** Suggest songs in the same audio cluster to improve user retention.
*   **Artist Benchmarking:** Help artists understand which audio "cluster" their new tracks fall into for better marketing.

---
*Developed as part of a Data Science learning journey. 🚀*
