---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Clustering
Clustering groups similar samples together based on certain characteristics in order to find the hidden patterns within data to simplify and improve processing and analysis.

## K-means
- **What**: Iteratively group samples into $k$ clusters till the centroids stabilize
- **Why**: Find natural groupings within data.
- **How**:
    1. Initialize $k$ centroids.
    2. Repeat:
    	1. Assign each sample to its nearest centroid.
		2. Update the centroids based on the mean of all samples assigned to them.
    3. Break if required or until the centroids stabilize.
- **When**:
	- There are exactly $k$ clusters in the data.
	- The data is scale-independent.
	- The clusters are spherical, equally sized, and well-separated with hard margins.
- **Pros**:
    - Simple.
- **Cons**:
    - Requires prior knowledge of $k$.
	- Sensitive to centroid initialization.
	- Bad performance on non-spherical, overlapping, or varying-sized clusters.
	- Possible convergence to local minima.


## Gaussian Mixture Models
## Spectral
## Hierarchical
### Agglomerative
### Divisive
## DBSCAN

# Matrix Factorization
## Singular Value Decomposition
## Principal Component Analysis
## Independent Component Analysis
## Non-Negative Matrix Factorization
## Latent Dirichlet Allocation

# Manifold Learning
## t-SNE
## UMAP
## Factor Analysis

# Anomaly Detection
## Distance-Based Methods
## Density-Based Methods

# Association Rule Learning
## Apriori Algorithm
## Eclat Algorithm
## FP-Growth Algorithm