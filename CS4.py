# 📌 Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# To display plots nicely
%matplotlib inline

# Set seaborn style
sns.set(style="whitegrid")

# --------------------------------------------
# 📌 Step 2: Load the Titanic dataset
# --------------------------------------------
# Load directly from seaborn
df = sns.load_dataset('titanic')

# Show first few rows
print(df.head())

# Basic info
print("\nINFO:")
print(df.info())

# Basic stats
print("\nDESCRIPTIVE STATS:")
print(df.describe(include='all'))

# --------------------------------------------
# 📌 Step 3: Check missing values
# --------------------------------------------
print("\nMISSING VALUES:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.show()

# --------------------------------------------
# 📌 Step 4: Visualize numerical distributions
# --------------------------------------------

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Distribution of Age")
plt.show()

# Fare distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['fare'], bins=30, kde=True)
plt.title("Distribution of Fare")
plt.show()

# --------------------------------------------
# 📌 Step 5: Categorical feature counts
# --------------------------------------------
# Sex count
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', data=df)
plt.title("Count of Passengers by Sex")
plt.show()

# Class count
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df)
plt.title("Count of Passengers by Class")
plt.show()

# Survived count
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title("Count of Survived (0=No, 1=Yes)")
plt.show()

# --------------------------------------------
# 📌 Step 6: Boxplots to detect outliers
# --------------------------------------------

# Age by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='age', data=df)
plt.title("Boxplot of Age by Class")
plt.show()

# Fare by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='fare', data=df)
plt.title("Boxplot of Fare by Class")
plt.show()

# Fare by Embarked
plt.figure(figsize=(10, 6))
sns.boxplot(x='embarked', y='fare', data=df)
plt.title("Boxplot of Fare by Embarked")
plt.show()

# --------------------------------------------
# 📌 Step 7: Correlation & Heatmap
# --------------------------------------------

# Numerical correlation
numerical_df = df[['age', 'fare', 'pclass', 'sibsp', 'parch', 'survived']]
corr = numerical_df.corr()

print("\nCORRELATION MATRIX:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 📌 Step 8: Relationships between variables
# --------------------------------------------

# Survival by sex
plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()

# Survival by class
plt.figure(figsize=(6, 4))
sns.barplot(x='class', y='survived', data=df)
plt.title("Survival Rate by Class")
plt.show()

# Age vs Fare scatter, colored by Survived
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare by Survival")
plt.show()

# Pairplot for selected variables
sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived')
plt.suptitle("Pairplot: Age, Fare, Pclass vs Survived", y=1.02)
plt.show()
