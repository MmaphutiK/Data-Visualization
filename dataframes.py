# Task 1: Load and Explore the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display the first few rows
print(df.head())

# Explore the structure
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Clean the dataset: No missing values in Iris, but for demonstration, we can drop or fill
# df = df.dropna()  # Drop missing values (if any)
# df = df.fillna(df.mean())  # Fill missing values with column mean (if any)

# Task 2: Basic Data Analysis

# Basic statistics
print(df.describe())

# Group by 'species' and compute the mean of numerical columns
grouped = df.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization

# Line chart (if you have a time-series, replace it with actual data)
# For demonstration, we'll plot a simple trend of a numerical column
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='species', y='sepal length (cm)')
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Bar chart: Comparing sepal length for each species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Histogram: Distribution of sepal lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], kde=True, bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Relationship between sepal length and petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
