# Customer Segmentation Using Clustering Models

This project applies various clustering algorithms to segment customers based on their demographic and financial attributes. The goal is to identify distinct customer groups for marketing and business strategies.

## Dataset

The dataset includes the following features:

- **CustomerID**: A unique identifier for each customer (not used for clustering).
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars.
- **Spending Score (1-100)**: A score assigned by the store based on customer spending behavior.

## Objective
The objective is to group customers into clusters based on their income, spending score, and other demographic attributes.

## Data Preprocessing

To prepare the data for clustering, the following preprocessing steps were applied:

1. **Feature Selection**: Only relevant features (Age, Annual Income, Spending Score) were used.
2. **Encoding Categorical Variables**: The Gender column was converted into numerical values (e.g., Male = 0, Female = 1).
3. **Feature Scaling**: Numerical features were standardized to ensure equal importance across attributes.


## Clustering Techniques

### K-Means Clustering

- Divides customers into k clusters by minimizing the variance within each cluster.
- The Elbow Method was used to determine the optimal number of clusters by analyzing the Within-Cluster Sum of Squares (WCSS).
- A scatter plot was generated to visualize customer clusters.


### Hierarchical Clustering

- Groups customers into a hierarchy of clusters using an Agglomerative Clustering approach.
- A dendrogram was used to determine the optimal number of clusters and analyze hierarchical relationships between customer groups.


## Results

### K-Means Clustering:

- Optimal number of clusters: 5 (determined using the elbow method).
- Customers were grouped based on similarities in spending behavior and income levels.
![kmeans_clustering](https://github.com/user-attachments/assets/66af270e-212e-4553-8d6c-e1c4b0caa952)

### Hierarchical Clustering:
- Produced a dendrogram that supports the clusters identified by K-Means.
![hc_clustering](https://github.com/user-attachments/assets/b2dd94dd-74bf-47d8-bd1b-8402dc5f5f28)

## Conclusion

Clustering models were used to segment customers into distinct groups based on their demographic and financial attributes. These insights can assist businesses in developing targeted marketing strategies and improving customer engagement.
