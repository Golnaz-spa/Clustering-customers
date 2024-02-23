# Customer Clustering for Online Store

This script demonstrates how to perform customer segmentation using K-Means clustering based on an online store dataset. The goal is to classify customers into three levels of loyalty: Loyal, Not Loyal, and Neutral, using their purchase history data. The segmentation is based on the RFM (Recency, Frequency, Monetary) model, which is a common approach in customer value analysis.

## Features

- **Data Preprocessing**: Handles missing values, removes canceled orders, and converts invoice dates to a usable format.
- **Feature Engineering**: Constructs the RFM (Recency, Frequency, Monetary) features to analyze customer behavior.
- **Standardization**: Applies standard scaling to the RFM features to normalize the data for clustering.
- **K-Means Clustering**: Utilizes K-Means to cluster customers into three groups based on their RFM characteristics.
- **Optimal Cluster Number Determination**: Implements the elbow method to determine the optimal number of clusters if needed.
- **Cluster Interpretation**: Calculates the centroids of each cluster to interpret the characteristics of each customer group.
- **Silhouette Analysis**: Performs silhouette analysis to evaluate the consistency within clusters.

## Usage

1. **Load Data**: The dataset `Marketnc.csv` should be placed in the appropriate path or updated to match the dataset's location.
2. **Data Cleaning and Preparation**: Remove missing values and canceled orders to clean the dataset for analysis.
3. **Feature Engineering**: Create RFM features that reflect customer purchase patterns.
4. **Clustering**: Apply K-Means clustering to segment customers into loyalty levels based on RFM features.
5. **Evaluation**: Use silhouette scores to assess the clustering performance and interpret cluster centroids to understand the characteristics of each segment.

## Notes

- Ensure the dataset path is correctly specified before running the script.
- The script requires `numpy`, `pandas`, `matplotlib`, `sklearn`, and their dependencies.
- Adjust the number of clusters in the K-Means algorithm as needed, based on business requirements or further analysis such as the elbow method.

This script provides a foundation for performing customer segmentation in an e-commerce context, facilitating targeted marketing strategies and improved customer relationship management.
