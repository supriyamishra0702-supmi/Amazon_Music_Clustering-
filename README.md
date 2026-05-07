# Amazon Music Clustering: Unsupervised Learning Project 🎧

### **Domain:** Music Analytics / Machine Learning  
### **Tools:** `Python` | `Scikit-Learn` | `Pandas` | `Streamlit`

---

## 📌 Project Overview
This project focuses on the automated categorization of songs based on their raw audio characteristics. Using the **Amazon Music dataset**, I implemented an unsupervised learning pipeline to group tracks into 5 distinct musical "moods" or "profiles." This eliminates the need for manual genre tagging and allows for data-driven playlist curation.

---
## 📈 Model Optimization: The Elbow Method
To determine the optimal number of clusters, I used the **Elbow Method**. By plotting the "Inertia" (Sum of Squared Distances) against the number of clusters (K), I identified the point where the rate of decrease significantly slows down.

<img width="566" height="289" alt="Elbow_Plot" src="https://github.com/user-attachments/assets/7b18bf24-a8c0-44a2-93c1-7893f3b8f591" />


*Note: The "Elbow" is clearly visible at K=5, which is the point I selected for the final model to balance granularity and cluster stability.*

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

<img width="1200" height="600" alt="cluster_feature_heatmap" src="https://github.com/user-attachments/assets/72df1e3e-45d9-41a4-a047-134eebdb0a59" />

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |

| `main.py` | The core machine learning pipeline (Cleaning, Clustering, Visualization). |

| `app.py` | The interactive Streamlit dashboard (Bonus deliverable). |

<img width="948" height="488" alt="streamlit_dashboard1" src="https://github.com/user-attachments/assets/2eda4098-6999-4c7c-84a6-2f56c7a4b786" />


| `single_genre_artists.csv` | The raw dataset used for analysis. |

| `Final_Clustered_Music_Data.csv` | The final output with cluster labels assigned. |


<img width="944" height="247" alt="cluster_data" src="https://github.com/user-attachments/assets/08202576-4a68-4029-ab5a-06db19ce4ff5" />

| `cluster_visualization.png` | PCA 2D plot showing song distribution. |


<img width="1000" height="700" alt="pca_cluster_chart" src="https://github.com/user-attachments/assets/5284942e-3387-4597-a8ba-1fef3b804e2f" />

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
<img width="947" height="484" alt="streamlit_dashboard" src="https://github.com/user-attachments/assets/4ddc84bd-875c-4dcb-8ded-2e10a3fb308e" />

---

## 💼 Business Impact
This model can be used to:
*   **Automate Playlist Generation:** Create "Deep Focus" or "Energy Boost" playlists instantly.
*   **Enhance Song Discovery:** Suggest songs in the same audio cluster to improve user retention.
*   **Artist Benchmarking:** Help artists understand which audio "cluster" their new tracks fall into for better marketing.

---
*Developed as part of a Data Science learning journey. 🚀*
