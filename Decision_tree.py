import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Decision Tree & Random Forest Explorer", layout="wide")
st.title("🌳 Decision Trees and Random Forests - Task 5")

uploaded_file = st.file_uploader("Upload Heart Disease Dataset (heart.csv)", type=["csv"], key="heart_uploader")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using default Heart dataset.")
    df = pd.read_csv("heart.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

target_column = "target"
X = df.drop(target_column, axis=1)
y = df[target_column]

test_size = st.slider("Test Size (%)", 10, 50, 20, 5) / 100
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

st.sidebar.header("Decision Tree Parameters")
max_depth_dt = st.sidebar.slider("Max Depth (Decision Tree)", 1, 10, 3)
dt_model = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=random_state)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

st.sidebar.header("Random Forest Parameters")
n_estimators_rf = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100, 10)
max_depth_rf = st.sidebar.slider("Max Depth (Random Forest)", 1, 10, 5)
rf_model = RandomForestClassifier(n_estimators=n_estimators_rf, max_depth=max_depth_rf, random_state=random_state)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

st.subheader("📊 Model Accuracy")
col1, col2 = st.columns(2)
with col1:
    st.metric("Decision Tree Accuracy", f"{acc_dt:.2%}")
with col2:
    st.metric("Random Forest Accuracy", f"{acc_rf:.2%}")

st.subheader("Confusion Matrix - Decision Tree")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Confusion Matrix - Random Forest")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens", ax=ax)
st.pyplot(fig)

st.subheader("Feature Importances (Random Forest)")
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]
fig, ax = plt.subplots()
ax.bar(range(X.shape[1]), importances[indices])
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(X.columns[indices], rotation=90)
st.pyplot(fig)

st.subheader("Decision Tree Visualization (Matplotlib)")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True, fontsize=8)
st.pyplot(fig)

st.subheader("Cross-Validation Scores")
cv_dt = cross_val_score(dt_model, X, y, cv=5).mean()
cv_rf = cross_val_score(rf_model, X, y, cv=5).mean()
st.write(f"Decision Tree CV Accuracy: {cv_dt:.2%}")
st.write(f"Random Forest CV Accuracy: {cv_rf:.2%}")

st.subheader("Classification Reports")
st.write("Decision Tree")
st.text(classification_report(y_test, y_pred_dt))
st.write("Random Forest")
st.text(classification_report(y_test, y_pred_rf))
