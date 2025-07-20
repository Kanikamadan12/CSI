# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load('iris_model.pkl')
iris = load_iris()

# App title
st.title("🌸 Iris Flower Classification App")

st.write("This app predicts the **species** of an Iris flower using features you provide.")

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Display prediction
st.subheader("Prediction:")
st.success(f"**Predicted Species:** {iris.target_names[prediction].capitalize()}")

# Display prediction probabilities
st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(proba_df.T)

# Show Iris dataset plot
st.subheader("Iris Dataset Visualization")
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

fig, ax = plt.subplots()
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='species', ax=ax)
st.pyplot(fig)
