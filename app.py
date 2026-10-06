
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA + TRAIN MODEL
# =========================================================

df = pd.read_csv("student_attendance.csv")

features = [
    "Attendance",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Result"
]

X = df[features]

encoder = LabelEncoder()
y = encoder.fit_transform(df["Risk_Level"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Student Dashboard")
st.sidebar.caption("AI-Based Student Risk Prediction")

page = st.sidebar.radio(
    "MENU",
    [
        "🏠 Home",
        "📊 Student Analysis",
        "🤖 Predict Risk",
        "👨‍🎓 All Students",
        "🌳 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** Decision Tree")
st.sidebar.write("**Students:**", len(df))

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🎓 Student Performance Dashboard")
    st.subheader("AI-Based Student Risk Prediction")

    st.write(
        "A simple machine-learning dashboard for analysing "
        "student performance and identifying academic risk."
    )

    st.markdown("---")

    total = len(df)
    low = (df["Risk_Level"] == "Low").sum()
    medium = (df["Risk_Level"] == "Medium").sum()
    high = (df["Risk_Level"] == "High").sum()

    st.subheader("Dashboard Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("👨‍🎓")
        st.write("**Total Students**")
        st.title(str(total))

    with col2:
        st.subheader("🟢")
        st.write("**Low Risk**")
        st.title(str(low))

    with col3:
        st.subheader("🟠")
        st.write("**Medium Risk**")
        st.title(str(medium))

    with col4:
        st.subheader("🔴")
        st.write("**High Risk**")
        st.title(str(high))

    st.markdown("---")

    st.subheader("📌 About This Project")

    st.write("""
This dashboard uses four student-performance factors:

- Attendance
- Internal Marks
- Assignment Score
- Previous Result

A **Decision Tree Classifier** uses these values to predict whether
a student belongs to **Low Risk, Medium Risk, or High Risk**.
""")

    st.info(
        "👈 Use the sidebar to explore the dashboard."
    )

# =========================================================
# STUDENT ANALYSIS
# =========================================================

elif page == "📊 Student Analysis":

    st.title("📊 Student Analysis")
    st.write("Analyse the whole class or select an individual student.")
    st.markdown("---")

    # -------------------------
    # CLASS ANALYSIS
    # -------------------------

    st.subheader("📈 Overall Class Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Risk Level Distribution**")

        risk_counts = (
            df["Risk_Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        ax1.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax1.set_xlabel("Risk Level")
        ax1.set_ylabel("Number of Students")
        ax1.set_title("Student Risk Distribution")

        st.pyplot(fig1)

    with col2:

        st.write("**Average Student Performance**")

        labels = [
            "Attendance",
            "Internal Marks",
            "Assignment",
            "Previous Result"
        ]

        averages = [
            df["Attendance"].mean(),
            df["Internal_Marks"].mean(),
            df["Assignment_Score"].mean(),
            df["Previous_Result"].mean()
        ]

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.bar(labels, averages)

        ax2.set_ylabel("Average Score")
        ax2.set_ylim(0, 100)
        ax2.set_title("Average Student Performance")
        ax2.tick_params(axis="x", rotation=20)

        st.pyplot(fig2)

    # -------------------------
    # INDIVIDUAL STUDENT
    # -------------------------

    st.markdown("---")

    st.subheader("👨‍🎓 Individual Student Analysis")

    st.write(
        "Select a student to view their performance "
        "and predict their academic risk."
    )

    selected_name = st.selectbox(
        "Select Student",
        df["Student_Name"].tolist()
    )

    student = df[
        df["Student_Name"] == selected_name
    ].iloc[0]

    st.markdown("---")

    st.subheader(f"📋 {selected_name}'s Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.write("**Attendance**")
        st.subheader(f"{student['Attendance']}%")

    with c2:
        st.write("**Internal Marks**")
        st.subheader(str(student["Internal_Marks"]))

    with c3:
        st.write("**Assignment Score**")
        st.subheader(str(student["Assignment_Score"]))

    with c4:
        st.write("**Previous Result**")
        st.subheader(str(student["Previous_Result"]))

    st.markdown("---")

    st.subheader("📊 Individual Performance Graph")

    student_labels = [
        "Attendance",
        "Internal Marks",
        "Assignment Score",
        "Previous Result"
    ]

    student_values = [
        student["Attendance"],
        student["Internal_Marks"],
        student["Assignment_Score"],
        student["Previous_Result"]
    ]

    fig_student, ax_student = plt.subplots(figsize=(8, 4))

    bars = ax_student.bar(
        student_labels,
        student_values
    )

    ax_student.set_ylim(0, 100)
    ax_student.set_ylabel("Score / Percentage")
    ax_student.set_title(
        f"{selected_name} - Performance Analysis"
    )

    ax_student.tick_params(
        axis="x",
        rotation=15
    )

    for bar, value in zip(bars, student_values):
        ax_student.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            str(value),
            ha="center"
        )

    st.pyplot(fig_student)

    # -------------------------
    # PREDICT SELECTED STUDENT
    # -------------------------

    st.markdown("---")

    if st.button(
        f"🔍 Predict Risk for {selected_name}",
        key="existing_student_prediction"
    ):

        selected_student_data = pd.DataFrame({
            "Attendance": [
                student["Attendance"]
            ],
            "Internal_Marks": [
                student["Internal_Marks"]
            ],
            "Assignment_Score": [
                student["Assignment_Score"]
            ],
            "Previous_Result": [
                student["Previous_Result"]
            ]
        })

        prediction = model.predict(
            selected_student_data
        )

        predicted_risk = encoder.inverse_transform(
            prediction
        )[0]

        st.subheader("🤖 Predicted Risk")

        if predicted_risk == "Low":

            st.success(
                f"🟢 {selected_name}: LOW RISK — "
                "Student performance is good."
            )

        elif predicted_risk == "Medium":

            st.warning(
                f"🟠 {selected_name}: MEDIUM RISK — "
                "Student needs some improvement."
            )

        else:

            st.error(
                f"🔴 {selected_name}: HIGH RISK — "
                "Student may need additional support."
            )

# =========================================================
# PREDICT NEW STUDENT
# =========================================================

elif page == "🤖 Predict Risk":

    st.title("🤖 Predict Risk for a New Student")

    st.write(
        "Enter the performance details of a new student "
        "and click Predict Risk."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=65
        )

        internal = st.number_input(
            "Internal Marks",
            min_value=0,
            max_value=100,
            value=55
        )

    with col2:

        assignment = st.number_input(
            "Assignment Score",
            min_value=0,
            max_value=100,
            value=60
        )

        previous = st.number_input(
            "Previous Result",
            min_value=0,
            max_value=100,
            value=58
        )

    st.markdown("---")

    if st.button(
        "🔍 Predict Risk",
        key="new_student_prediction"
    ):

        new_student = pd.DataFrame({
            "Attendance": [attendance],
            "Internal_Marks": [internal],
            "Assignment_Score": [assignment],
            "Previous_Result": [previous]
        })

        prediction = model.predict(new_student)

        risk = encoder.inverse_transform(
            prediction
        )[0]

        st.subheader("Prediction Result")

        if risk == "Low":

            st.success(
                "🟢 LOW RISK — Student performance is good."
            )

        elif risk == "Medium":

            st.warning(
                "🟠 MEDIUM RISK — Student needs some improvement."
            )

        else:

            st.error(
                "🔴 HIGH RISK — Student may need additional support."
            )

# =========================================================
# ALL STUDENTS
# =========================================================

elif page == "👨‍🎓 All Students":

    st.title("👨‍🎓 All Students")

    st.write(
        "Complete information for all students in the dataset."
    )

    st.markdown("---")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.write("**Total Students:**", len(df))

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🌳 Model Performance":

    st.title("🌳 Model Performance")

    st.write(
        "Performance of the Decision Tree classification model."
    )

    st.markdown("---")

    st.subheader("🌳 Model Used")
    st.success("Decision Tree Classifier")

    st.write("""
Decision Tree is used because the output is divided into three classes:

- 🟢 Low Risk
- 🟠 Medium Risk
- 🔴 High Risk
""")

    st.markdown("---")

    st.subheader("🎯 Model Accuracy")

    st.title(f"{accuracy * 100:.2f}%")

    st.caption(
        "Accuracy is calculated on the 20% test split "
        f"({len(X_test)} students) of this small demonstration dataset."
    )

    st.markdown("---")

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=range(len(encoder.classes_))
    )

    fig3, ax3 = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    disp.plot(
        ax=ax3,
        cmap="Blues",
        colorbar=False
    )

    ax3.set_title(
        "Decision Tree Confusion Matrix"
    )

    st.pyplot(fig3)

    st.write(
        "The confusion matrix compares the actual risk "
        "level with the predicted risk level."
    )

    st.markdown("---")

    st.subheader("✅ Conclusion")

    st.write("""
The Decision Tree model analyses Attendance, Internal Marks,
Assignment Score and Previous Result.

It then classifies students into **Low, Medium or High Risk**.

The dashboard can analyse the whole class, examine an individual
student, and predict the risk of a new student.
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "🎓 Student Performance Dashboard | Machine Learning Project"
)
