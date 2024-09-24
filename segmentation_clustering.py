import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the Excel file and the specific sheet into a DataFrame
xls = pd.ExcelFile(r'Projects/Python/Project1/Project2/Online Retail.xlsx')
data = pd.read_excel(xls, sheet_name='Online Retail')

# Data Cleaning
# Forward fill missing values and drop rows with missing 'CustomerID'
data.ffill(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)

# Feature Engineering
# Calculate total spend per customer by multiplying Quantity and UnitPrice
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Aggregate data to calculate total spend, total quantity, and total transactions per customer
customer_data = data.groupby('CustomerID').agg({
    'TotalSpend': 'sum',
    'Quantity': 'sum',
    'InvoiceNo': 'count'  # 'InvoiceNo' used to calculate the number of transactions
}).reset_index()

# Rename columns for clarity
customer_data.rename(columns={'TotalSpend': 'TotalSpend', 
                              'Quantity': 'TotalQuantity', 
                              'InvoiceNo': 'TotalTransactions'}, inplace=True)

# Standardize the features for clustering
features = ['TotalSpend', 'TotalQuantity', 'TotalTransactions']
X = customer_data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to determine the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means clustering with the optimal number of clusters (based on the Elbow Method)
optimal_clusters = 4  # Choose the appropriate number based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters based on Total Spend and Total Quantity
plt.figure(figsize=(12, 8))
scatter = plt.scatter(customer_data['TotalSpend'], customer_data['TotalQuantity'], 
                      c=customer_data['Cluster'], cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Segmentation')
plt.xlabel('Total Spend')
plt.ylabel('Total Quantity')
plt.show()

# Cluster summary statistics
for i in range(optimal_clusters):
    cluster_summary = customer_data[customer_data['Cluster'] == i]
    print(f"Cluster {i} Summary:")
    print(cluster_summary.describe(), '\n')

# Evaluate the clustering performance using the Silhouette Score
silhouette_avg = silhouette_score(X_scaled, customer_data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')