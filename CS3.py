# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset directly from seaborn
iris = sns.load_dataset('iris')

# Display the first few rows
print(iris.head())

# Basic Information
print(iris.info())

# -----------------------
# 1️⃣ Pairplot: All features pairwise
# -----------------------
sns.pairplot(iris, hue='species')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# -----------------------
# 2️⃣ Boxplot: Feature distributions by species
# -----------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=iris)
plt.title("Boxplot of Petal Length by Species")
plt.show()

# -----------------------
# 3️⃣ Violin plot: Feature distributions
# -----------------------
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.title("Violin Plot of Sepal Width by Species")
plt.show()

# -----------------------
# 4️⃣ Heatmap: Correlation matrix
# -----------------------
plt.figure(figsize=(8, 6))
corr = iris.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------
# 5️⃣ Scatter Plot with regression line
# -----------------------
plt.figure(figsize=(10, 6))
sns.lmplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.title("Sepal Length vs Petal Length with Regression Lines")
plt.show()
