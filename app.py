import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Early-Warning System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "early_warning_model.keras"
    )

    preprocessor = joblib.load(
        "early_warning_preprocessor.pkl"
    )

    class_names = np.load(
        "class_names.npy"
    )

    return model, preprocessor, class_names


model, preprocessor, class_names = load_model()

# ============================================================
# HELPER FUNCTION
# ============================================================

def select_code(label, options, default=0):
    labels = [item[0] for item in options]
    values = [item[1] for item in options]

    selected = st.selectbox(
        label,
        labels,
        index=default
    )

    return values[labels.index(selected)]


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 Student Dropout / Performance Prediction")

st.subheader("AI-Based Early-Warning System")

st.write(
    "Predict a student's likely academic outcome using "
    "admission, demographic, financial and first-semester "
    "academic information."
)

st.info(
    "Possible outcomes: Dropout, Enrolled, or Graduate."
)

st.divider()

# ============================================================
# STUDENT & PERSONAL INFORMATION
# ============================================================

st.header("👤 Student & Personal Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=80,
        value=20
    )

with col2:

    gender = select_code(
        "Gender",
        [
            ("Female", 0),
            ("Male", 1)
        ]
    )

with col3:

    marital_status = select_code(
        "Marital Status",
        [
            ("Single", 1),
            ("Married", 2),
            ("Widower", 3),
            ("Divorced", 4),
            ("Facto Union", 5),
            ("Legally Separated", 6)
        ]
    )

# ============================================================
# ADMISSION INFORMATION
# ============================================================

st.header("🎓 Admission Information")

col1, col2 = st.columns(2)

with col1:

    application_mode = select_code(
        "Application Mode",
        [
            ("1 - General contingent", 1),
            ("2 - Ordinance No. 612/93", 2),
            ("5 - Special contingent - Azores", 5),
            ("7 - Other higher courses", 7),
            ("10 - Ordinance No. 854-B/99", 10),
            ("15 - International student", 15),
            ("16 - Special contingent - Madeira", 16),
            ("17 - 2nd phase - general", 17),
            ("18 - 3rd phase - general", 18),
            ("26 - Different plan", 26),
            ("27 - Other institution", 27),
            ("39 - Over 23 years old", 39),
            ("42 - Transfer", 42),
            ("43 - Change of course", 43),
            ("44 - Technological specialization", 44),
            ("51 - Change institution/course", 51),
            ("53 - Short cycle diploma", 53),
            ("57 - International institution/course", 57)
        ]
    )

with col2:

    application_order = st.selectbox(
        "Application Preference",
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ],
        index=1,
        format_func=lambda x:
            "First choice" if x == 0
            else f"Choice {x + 1}"
    )

col1, col2, col3 = st.columns(3)

with col1:

    course = select_code(
        "Course",
        [
            ("Biofuel Production Technologies", 33),
            ("Animation and Multimedia Design", 171),
            ("Social Service - Evening", 8014),
            ("Agronomy", 9003),
            ("Communication Design", 9070),
            ("Veterinary Nursing", 9085),
            ("Informatics Engineering", 9119),
            ("Equinculture", 9130),
            ("Management", 9147),
            ("Social Service", 9238),
            ("Tourism", 9254),
            ("Nursing", 9500),
            ("Oral Hygiene", 9556),
            ("Advertising and Marketing Management", 9670),
            ("Journalism and Communication", 9773),
            ("Basic Education", 9853),
            ("Management - Evening", 9991)
        ]
    )

with col2:

    attendance = select_code(
        "Class Attendance",
        [
            ("Daytime", 1),
            ("Evening", 0)
        ]
    )

with col3:

    previous_qualification = select_code(
        "Previous Qualification",
        [
            ("Secondary education", 1),
            ("Higher education - Bachelor's degree", 2),
            ("Higher education - Degree", 3),
            ("Higher education - Master's", 4),
            ("Higher education - Doctorate", 5),
            ("Frequency of higher education", 6),
            ("12th year - not completed", 9),
            ("11th year - not completed", 10),
            ("Other - 11th year", 12),
            ("10th year", 14),
            ("10th year - not completed", 15),
            ("Basic education - 3rd cycle", 19),
            ("Basic education - 2nd cycle", 38),
            ("Technological specialization", 39),
            ("Higher education - 1st cycle", 40),
            ("Professional higher technical course", 42),
            ("Higher education - Master's - 2nd cycle", 43)
        ]
    )

col1, col2, col3 = st.columns(3)

with col1:

    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

with col2:

    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0,
        max_value=200.0,
        value=125.0
    )

with col3:

    nationality = select_code(
        "Nationality",
        [
            ("Portuguese", 1),
            ("German", 2),
            ("Spanish", 6),
            ("Italian", 11),
            ("Dutch", 13),
            ("English", 14),
            ("Lithuanian", 17),
            ("Angolan", 21),
            ("Cape Verdean", 22),
            ("Guinean", 24),
            ("Mozambican", 25),
            ("Santomean", 26),
            ("Turkish", 32),
            ("Brazilian", 41),
            ("Romanian", 62),
            ("Moldova", 100),
            ("Mexican", 101),
            ("Ukrainian", 103),
            ("Russian", 105),
            ("Cuban", 108),
            ("Colombian", 109)
        ]
    )

# ============================================================
# FAMILY INFORMATION
# ============================================================

st.header("👨‍👩‍👧 Family Information")

st.caption(
    "Select the highest known educational qualification "
    "of the student's parents."
)

parent_qualification_options = [
    ("Secondary education", 1),
    ("Higher education - Bachelor's degree", 2),
    ("Higher education - Degree", 3),
    ("Higher education - Master's", 4),
    ("Higher education - Doctorate", 5),
    ("Frequency of higher education", 6),
    ("12th year - not completed", 9),
    ("11th year - not completed", 10),
    ("7th year - old system", 11),
    ("Other - 11th year", 12),
    ("10th year", 14),
    ("General commerce course", 18),
    ("Basic education - 3rd cycle", 19),
    ("Technical-professional course", 22),
    ("7th year of schooling", 26),
    ("2nd cycle general high school", 27),
    ("9th year - not completed", 29),
    ("8th year of schooling", 30),
    ("Unknown", 34),
    ("Cannot read or write", 35),
    ("Can read without 4th year", 36),
    ("Basic education - 1st cycle", 37),
    ("Basic education - 2nd cycle", 38),
    ("Technological specialization", 39),
    ("Higher education - 1st cycle", 40),
    ("Specialized higher studies", 41),
    ("Professional higher technical course", 42),
    ("Higher education - Master's", 43),
    ("Higher education - Doctorate", 44)
]

col1, col2 = st.columns(2)

with col1:

    mothers_qualification = select_code(
        "Mother's Education",
        parent_qualification_options
    )

with col2:

    fathers_qualification = select_code(
        "Father's Education",
        parent_qualification_options
    )

# ============================================================
# OCCUPATION
# ============================================================

occupation_options = [
    ("Student", 0),
    ("Managers / Executive managers", 1),
    ("Specialists - intellectual & scientific", 2),
    ("Technicians and professionals", 3),
    ("Administrative staff", 4),
    ("Personal services / Security / Sellers", 5),
    ("Agriculture / Fisheries / Forestry", 6),
    ("Industry / Construction / Crafts", 7),
    ("Machine operators / Assemblers", 8),
    ("Unskilled workers", 9),
    ("Armed forces", 10),
    ("Other situation", 90),
    ("Not specified", 99),
    ("Health professionals", 122),
    ("Teachers", 123),
    ("ICT specialists", 125),
    ("Science & Engineering technicians", 131),
    ("Health technicians", 132),
    ("Legal / Social / Sports / Cultural services", 134),
    ("Office workers", 141),
    ("Accounting / Financial services", 143),
    ("Administrative support", 144),
    ("Personal service workers", 151),
    ("Sellers", 152),
    ("Personal care workers", 153),
    ("Construction workers", 171),
    ("Printing / Precision / Jewellery workers", 173),
    ("Food / Wood / Clothing workers", 175),
    ("Cleaning workers", 191),
    ("Unskilled agriculture workers", 192),
    ("Unskilled industry / Construction workers", 193),
    ("Meal preparation assistants", 194)
]

col1, col2 = st.columns(2)

with col1:

    mothers_occupation = select_code(
        "Mother's Occupation",
        occupation_options
    )

with col2:

    fathers_occupation = select_code(
        "Father's Occupation",
        occupation_options
    )

# ============================================================
# FINANCIAL & SUPPORT INFORMATION
# ============================================================

st.header("💰 Financial & Support Information")

col1, col2, col3 = st.columns(3)

with col1:

    displaced = select_code(
        "Displaced Student",
        [
            ("No", 0),
            ("Yes", 1)
        ]
    )

with col2:

    special_needs = select_code(
        "Educational Special Needs",
        [
            ("No", 0),
            ("Yes", 1)
        ]
    )

with col3:

    international = select_code(
        "International Student",
        [
            ("No", 0),
            ("Yes", 1)
        ]
    )

col1, col2, col3 = st.columns(3)

with col1:

    debtor = select_code(
        "Outstanding Fees / Debtor",
        [
            ("No", 0),
            ("Yes", 1)
        ]
    )

with col2:

    tuition = select_code(
        "Tuition Fees Up to Date",
        [
            ("Yes", 1),
            ("No", 0)
        ]
    )

with col3:

    scholarship = select_code(
        "Scholarship Holder",
        [
            ("No", 0),
            ("Yes", 1)
        ]
    )

# ============================================================
# FIRST SEMESTER
# ============================================================

st.header("📚 First Semester Academic Information")

st.caption(
    "These values represent the student's academic activity "
    "during the first semester."
)

col1, col2, col3 = st.columns(3)

with col1:

    credited = st.number_input(
        "Credited Units",
        min_value=0,
        value=0
    )

with col2:

    enrolled = st.number_input(
        "Enrolled Units",
        min_value=0,
        value=6
    )

with col3:

    evaluations = st.number_input(
        "Evaluations",
        min_value=0,
        value=6
    )

col1, col2, col3 = st.columns(3)

with col1:

    approved = st.number_input(
        "Approved Units",
        min_value=0,
        value=5
    )

with col2:

    grade = st.number_input(
        "Average Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1
    )

with col3:

    without_evaluations = st.number_input(
        "Units Without Evaluation",
        min_value=0,
        value=0
    )

# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Student Outcome",
    use_container_width=True
):

    try:

        student_data = {

            "Marital status":
                marital_status,

            "Application mode":
                application_mode,

            "Application order":
                application_order,

            "Course":
                course,

            "Daytime/evening attendance":
                attendance,

            "Previous qualification":
                previous_qualification,

            "Previous qualification (grade)":
                previous_qualification_grade,

            "Nacionality":
                nationality,

            "Mother's qualification":
                mothers_qualification,

            "Father's qualification":
                fathers_qualification,

            "Mother's occupation":
                mothers_occupation,

            "Father's occupation":
                fathers_occupation,

            "Admission grade":
                admission_grade,

            "Displaced":
                displaced,

            "Educational special needs":
                special_needs,

            "Debtor":
                debtor,

            "Tuition fees up to date":
                tuition,

            "Gender":
                gender,

            "Scholarship holder":
                scholarship,

            "Age at enrollment":
                age,

            "International":
                international,

            "Curricular units 1st sem (credited)":
                credited,

            "Curricular units 1st sem (enrolled)":
                enrolled,

            "Curricular units 1st sem (evaluations)":
                evaluations,

            "Curricular units 1st sem (approved)":
                approved,

            "Curricular units 1st sem (grade)":
                grade,

            "Curricular units 1st sem (without evaluations)":
                without_evaluations
        }

        student_df = pd.DataFrame(
            [student_data]
        )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed = preprocessor.transform(
            student_df
        )

        processed = np.asarray(
            processed
        ).astype("float32")

        # ----------------------------------------------------
        # DEEP LEARNING PREDICTION
        # ----------------------------------------------------

        output = model(
            processed,
            training=False
        )

        probabilities = output.numpy()[0]

        predicted_class = np.argmax(
            probabilities
        )

        prediction = class_names[
            predicted_class
        ]

        # ----------------------------------------------------
        # DROPOUT RISK
        # ----------------------------------------------------

        dropout_index = list(
            class_names
        ).index("Dropout")

        dropout_probability = (
            probabilities[
                dropout_index
            ] * 100
        )

        if dropout_probability >= 70:

            risk = "HIGH RISK"

        elif dropout_probability >= 40:

            risk = "MODERATE RISK"

        else:

            risk = "LOW RISK"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Outcome",
                prediction
            )

        with col2:

            st.metric(
                "Dropout Probability",
                f"{dropout_probability:.2f}%"
            )

        with col3:

            st.metric(
                "Risk Level",
                risk
            )

        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk == "HIGH RISK":

            st.error(
                "⚠️ HIGH RISK — Early intervention recommended."
            )

        elif risk == "MODERATE RISK":

            st.warning(
                "⚠️ MODERATE RISK — Student should be monitored."
            )

        else:

            st.success(
                "✓ LOW RISK — Continue regular monitoring."
            )

        # ----------------------------------------------------
        # PROBABILITY TABLE
        # ----------------------------------------------------

        st.subheader(
            "Outcome Probabilities"
        )

        probability_df = pd.DataFrame({

            "Outcome":
                class_names,

            "Probability (%)":
                probabilities * 100

        })

        probability_df[
            "Probability (%)"
        ] = probability_df[
            "Probability (%)"
        ].round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probability"
        )

        st.bar_chart(
            probability_df.set_index(
                "Outcome"
            )
        )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Recommended Action"
        )

        if risk == "HIGH RISK":

            st.write(
                """
                **Immediate support recommended**

                • Faculty / mentor intervention

                • Academic counseling

                • Monitor attendance and academic progress

                • Provide additional academic support
                """
            )

        elif risk == "MODERATE RISK":

            st.write(
                """
                **Regular monitoring recommended**

                • Monitor academic performance

                • Encourage mentor interaction

                • Monitor attendance and assessments

                • Provide support where required
                """
            )

        else:

            st.write(
                """
                **Continue regular monitoring**

                • Maintain current academic performance

                • Continue normal academic support
                """
            )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.divider()

        st.caption(
            "This system is an academic early-warning tool. "
            "The prediction indicates model-estimated risk "
            "and should not be treated as a definitive "
            "decision about a student's future."
        )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Student Dropout / Performance Prediction | "
    "Deep Learning & Applications Project"
)