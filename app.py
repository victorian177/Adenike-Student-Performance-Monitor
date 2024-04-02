import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.title("Student Performance Predictor")
st.write("### CGPA Dataset")
cgpa_df = pd.read_csv("cgpa.csv")
st.write(cgpa_df.sample(5))
columns = st.multiselect(
    "Select columns for histogram",
    ["Past CGPA 1", "Past CGPA 2", "Current CGPA"],
)

if columns:
    st.write("### Histogram:")
    fig, ax = plt.subplots(figsize=(8, 6))
    for column in columns:
        ax.hist(cgpa_df[column], bins=20, alpha=0.7, label=column)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Selected Columns")
    ax.legend()
    st.pyplot(fig)

X = cgpa_df[["Past CGPA 1", "Past CGPA 2"]]
y = cgpa_df["Current CGPA"]
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=0.5,
    random_state=42,
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)

st.write(f"Mean Squared Error: {mse}")

y_test_preds = model.predict(X_test)
differences = y_test - y_test_preds

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.stem(np.arange(len(y_test)), (y_test_preds - y_test),)
ax1.set_xlabel("Index")
ax1.set_ylabel("Difference")
ax1.set_title("Difference between predictions and truth values")
ax1.axhline(0, color="black", linewidth=0.5)  # Add horizontal line at y=0
ax1.grid(True)
st.pyplot(fig1)
