# Customer Segmentation Using K-Means Clustering

This project applies K-Means clustering to segment customers of an online retail store based on purchasing behavior. The analysis includes data cleaning, feature engineering, and clustering, with the goal of identifying distinct customer groups to enhance business strategies.

## Overview

Customer segmentation is a valuable process that helps businesses understand their customer base and tailor their services or marketing strategies accordingly. In this project, we cluster customers based on their total spend, quantity of products purchased, and the number of transactions made.

## Project Workflow

1. **Data Cleaning:**
   - Handled missing data by forward-filling missing values.
   - Removed rows where `CustomerID` was missing.
   
2. **Feature Engineering:**
   - Created new features such as `Total Spend` by multiplying the quantity purchased and unit price.
   - Aggregated data by `CustomerID` to calculate total spend, total quantity, and the number of transactions.

3. **Clustering:**
   - Standardized the features for clustering using `StandardScaler`.
   - Applied K-Means clustering to group customers into distinct segments.
   - Used the Elbow Method to determine the optimal number of clusters.
   
4. **Visualization:**
   - Plotted customer clusters to understand the relationship between `Total Spend` and `Total Quantity`.
   
5. **Evaluation:**
   - Evaluated the clustering performance using the Silhouette Score.

## Tools & Libraries Used

- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For clustering, scaling, and evaluation metrics.
- **Matplotlib:** For visualizations and plotting the clusters.
- **Jupyter Notebooks/VS Code:** For code execution and development.

## Installation & Setup

To run this project locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/Mr-Ayushh/Customer-Segmentation-Using-K.git
   ```

2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. Run the Python script:
   ```bash
   python customer_segmentation.py
   ```

## Project Features

- **Elbow Method:** Determines the optimal number of clusters by plotting inertia against the number of clusters.
- **Customer Segmentation:** Clusters customers into distinct groups based on their purchasing behavior.
- **Visualization:** Provides a scatter plot to visually understand the distribution of customers across clusters.

## Results

The K-Means algorithm grouped customers into four clusters, revealing distinct patterns in customer behavior. Cluster analysis helps in understanding different types of customers, such as high-spend customers or frequent buyers, which can be useful for targeted marketing.

## Evaluation

The clustering model achieved a **Silhouette Score** of `X.XX`, indicating the cohesion and separation of the customer segments.

## Contributing

Feel free to submit pull requests or open issues for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
