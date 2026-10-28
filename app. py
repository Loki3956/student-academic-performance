import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
st.set_page_config(
    page_title="Student Performance Classification",
    page_icon="🎓"
)
st.title("🎓 Student Academic Performance Classification")
st.write("Machine Learning Based Student Performance Prediction")
data = pd.DataFrame({
    "Study_Hours": [1,2,3,4,5,6,7,8,2,5,3,6,8,1,4,7,2,6,9,3,5,8,4,7,1,9,6,3,5,8],
    "Attendance": [40,50,60,70,80,90,95,98,55,75,65,85,96,35,72,92,45,88,99,58,78,97,68,94,30,100,82,62,76,95],
    "Previous_Marks": [25,35,45,55,65,75,85,95,30,60,40,72,92,20,58,88,28,78,98,42,67,96,50,86,15,99,70,38,63,93]
})
def classify(row):
    score = (
        row["Previous_Marks"] * 0.6
        + row["Attendance"] * 0.25
        + row["Study_Hours"] * 2
    )
    if score < 40:
        return "Low"
    elif score < 65:
        return "Medium"
    return "High"
data["Performance"] = data.apply(classify, axis=1)
X = data[["Study_Hours", "Attendance", "Previous_Marks"]]
y = data["Performance"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.subheader("📊 Model Information")
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
st.write("Total Student Records:", len(data))
st.divider()
st.subheader("📝 Enter Student Details")
study_hours = st.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)
attendance = st.slider(
    "Attendance Percentage",
    0, 100, 80
)
previous_marks = st.slider(
    "Previous Marks",
    0, 100, 70
)
if st.button("🔮 Predict Performance"):
    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })
    result = model.predict(student)[0]
    st.subheader("🎯 Prediction Result")
    if result == "High":
        st.success("High Performance 🌟")
    elif result == "Medium":
        st.warning("Medium Performance 📘")
    else:
        st.error("Low Performance 📉")
    st.write("Study Hours:", study_hours)
    st.write("Attendance:", str(attendance) + "%")
    st.write("Previous Marks:", previous_marks)
st.divider()
st.caption("Student Academic Performance Classification | Mini Project")
st.caption("Educational demonstration using a small sample dataset.")
