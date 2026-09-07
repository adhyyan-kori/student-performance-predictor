import streamlit as st
import pandas as pd
import joblib
import shap

st.title("Student Performance Predictor")

if "page" not in st.session_state:
    st.session_state.page = 1

@st.cache_resource
def load_model():
    model = joblib.load("model_encoder/random_forest_model.pkl")
    encoder = joblib.load("model_encoder/encoder.pkl")
    explainer = shap.TreeExplainer(model)

    return model, encoder, explainer


model, encoder, explainer = load_model()
explainer=shap.TreeExplainer(model)


if st.session_state.page == 1:

    st.header("Student Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    if gender == "Male":
        gender = "M"
    else:
        gender = "F"


    caste = st.selectbox(
        "Caste",
        [
            "General",
            "Scheduled Tribe (ST)",
            "Scheduled Caste (SC)",
            "Other Backward Class (OBC)",
            "More Other Backward Class (MOBC)"
        ]
    )

    if caste == "General":
        caste = "G"
    elif caste == "Scheduled Tribe (ST)":
        caste = "ST"
    elif caste == "Scheduled Caste (SC)":
        caste = "SC"
    elif caste == "Other Backward Class (OBC)":
        caste = "OBC"
    elif caste == "More Other Backward Class (MOBC)":
        caste = "MOBC"


    living_status = st.selectbox(
        "Living Status",
        ["Town", "Village"]
    )

    if living_status == "Town":
        living_status = "T"
    else:
        living_status = "V"


    admission_status = st.selectbox(
        "Admission Status",
        ["Free Admission", "Paid Admission"]
    )

    if admission_status == "Free Admission":
        admission_status = "Free"
    else:
        admission_status = "Paid"


    school_sector = st.selectbox(
        "School Sector",
        ["Government", "Private"]
    )

    if school_sector == "Government":
        school_sector = "Govt"
    else:
        school_sector = "Private"


    medium = st.selectbox(
        "Medium of Instruction",
        ["English", "Assamese", "Hindi", "Bengali"]
    )

    if medium == "English":
        medium = "Eng"
    elif medium == "Assamese":
        medium = "Asm"
    elif medium == "Hindi":
        medium = "Hin"
    elif medium == "Bengali":
        medium = "Ben"


    # =========================
    # ACADEMIC INFORMATION
    # =========================

    st.header("Academic Information")


    tenth_performance = st.selectbox(
        "10th Grade Performance",
        ["Best", "Very good", "Good", "Pass"]
    )

    tenth_value = (
        "Vg" if tenth_performance == "Very good"
        else tenth_performance
    )


    twelfth_performance = st.selectbox(
        "12th Grade Performance",
        ["Best", "Very good", "Good", "Pass"]
    )

    twelfth_value = (
        "Vg" if twelfth_performance == "Very good"
        else twelfth_performance
    )


    arrears = st.selectbox(
        "History of Arrears",
        ["Yes", "No"]
    )

    if arrears == "Yes":
        arrears = "Y"
    else:
        arrears = "N"


    study_habits = st.selectbox(
        "Study Habits",
        [
            "Good (Regular Study)",
            "Average (Occasional Study)",
            "Poor (Rarely Studies)"
        ]
    )

    if study_habits == "Good (Regular Study)":
        study_habits = "Good"
    elif study_habits == "Average (Occasional Study)":
        study_habits = "Average"
    else:
        study_habits = "Poor"


    attendance = st.selectbox(
        "Attendance",
        [
            "Good (Regular)",
            "Average (Sometimes Absent)",
            "Poor (Frequently Absent)"
        ]
    )

    if attendance == "Good (Regular)":
        attendance = "Good"
    elif attendance == "Average (Sometimes Absent)":
        attendance = "Average"
    else:
        attendance = "Poor"


    internal_assessment = st.selectbox(
        "Internal Assessment",
        ["Best", "Very good", "Good", "Pass"]
    )

    internal_assessment_value = (
        "Vg" if internal_assessment == "Very good"
        else internal_assessment
    )


    travel_time = st.selectbox(
        "Travel Time",
        [
            "Up to 30 minutes",
            "30–60 minutes",
            "More than 60 minutes"
        ]
    )

    if travel_time == "Up to 30 minutes":
        travel_time = "Small"
    elif travel_time == "30–60 minutes":
        travel_time = "Average"
    else:
        travel_time = "Large"


    # =========================
    # FAMILY INFORMATION
    # =========================

    st.header("Family Information")


    family_income = st.selectbox(
        "Family Monthly Income",
        ["Very high", "High", "Average", "Medium", "Low"]
    )

    if family_income == "Very high":
        family_income = "Vh"
    elif family_income == "Average":
        family_income = "Am"


    father_qualification = st.selectbox(
        "Father's Qualification",
        [
            "Illiterate",
            "Under Matric",
            "10th",
            "12th",
            "Degree",
            "Post graduate"
        ]
    )

    if father_qualification == "Illiterate":
        father_qualification = "Il"
    elif father_qualification == "Under Matric":
        father_qualification = "Um"
    elif father_qualification == "Post graduate":
        father_qualification = "Pg"


    mother_qualification = st.selectbox(
        "Mother's Qualification",
        [
            "Illiterate",
            "Under Matric",
            "10th",
            "12th",
            "Degree",
            "Post graduate"
        ]
    )

    if mother_qualification == "Illiterate":
        mother_qualification = "Il"
    elif mother_qualification == "Under Matric":
        mother_qualification = "Um"
    elif mother_qualification == "Post graduate":
        mother_qualification = "Pg"


    father_occupation = st.selectbox(
        "Father's Occupation",
        ["Service", "Business", "Retired", "Farmer", "Others"]
    )


    mother_occupation = st.selectbox(
        "Mother's Occupation",
        ["Service", "Business", "Retired", "Housewife", "Others"]
    )


    family_members_number = st.number_input(
        "Number of Family Members",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    st.caption("Enter a number between 1 and 20")


    if family_members_number <= 3:
        family_members = "Small"
        family_size = "Small"
    elif family_members_number <= 5:
        family_members = "Average"
        family_size = "Average"
    else:
        family_members = "Large"
        family_size = "Large"


    # =========================
    # PREDICT
    # =========================

    if st.button("Predict Performance"):

        input_data = pd.DataFrame({
            "ge": [gender],
            "cst": [caste],
            "tnp": [tenth_value],
            "twp": [twelfth_value],
            "iap": [internal_assessment_value],
            "arr": [arrears],
            "ls": [living_status],
            "as": [admission_status],
            "fmi": [family_income],
            "fs": [family_size],
            "fq": [father_qualification],
            "mq": [mother_qualification],
            "fo": [father_occupation],
            "mo": [mother_occupation],
            "nf": [family_members],
            "sh": [study_habits],
            "ss": [school_sector],
            "me": [medium],
            "tt": [travel_time],
            "atd": [attendance]
        })

        encoded_input = encoder.transform(input_data)

        prediction = model.predict(encoded_input)

        st.session_state.prediction = prediction[0]
        st.session_state.student_data=input_data

        st.session_state.page = 2

        st.rerun()


# =========================
# PAGE 2 - RESULT
# =========================
# =========================
# PAGE 2 - RESULT
# =========================

if st.session_state.page == 2:

    result = st.session_state.prediction

    if result == "Vg":
        result = "Very Good"

    st.header("🎯 Prediction Result")

    st.success(f"Predicted Performance: {result}")

    # Get student's data
    student_data = st.session_state.student_data

    # Encode student's data
    encoded_student = encoder.transform(student_data)
    encoded_student = encoded_student.toarray().astype(float)

    # Calculate SHAP values
    shap_values = explainer.shap_values(encoded_student)

    # Find predicted class
    predicted_class = st.session_state.prediction
    class_index = list(model.classes_).index(predicted_class)

    # SHAP values for predicted class
    student_shap = shap_values[0, :, class_index]

    # Feature names
    feature_names = encoder.get_feature_names_out()

    student_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": student_shap
    })

    student_importance["abs_shap"] = (
        student_importance["shap_value"].abs()
    )

    # Find student's actual categories
    actual_features = []

    for feature in feature_names:

        if "_" in feature:

            original_feature, category = feature.split("_", 1)

            if original_feature in student_data.columns:

                actual_value = str(
                    student_data.iloc[0][original_feature]
                )

                if actual_value == category:
                    actual_features.append(feature)

    # Keep only actual student features
    student_importance_actual = student_importance[
        student_importance["feature"].isin(actual_features)
    ].copy()

    # Sort by influence
    student_importance_actual = student_importance_actual.sort_values(
        "abs_shap",
        ascending=False
    )

    # Top 5
    top_factors = student_importance_actual.head(5)


    # =========================
    # LABELS
    # =========================

    feature_labels = {
        "tnp": "10th Grade Performance",
        "twp": "12th Grade Performance",
        "iap": "Internal Assessment",
        "ls": "Living Status",
        "fo": "Father's Occupation",
        "mo": "Mother's Occupation",
        "sh": "Study Habits",
        "atd": "Attendance",
        "arr": "History of Arrears",
        "as": "Admission Status",
        "ge": "Gender",
        "cst": "Caste",
        "fmi": "Family Monthly Income",
        "fs": "Family Size",
        "fq": "Father's Qualification",
        "mq": "Mother's Qualification",
        "nf": "Number of Family Members",
        "ss": "School Sector",
        "me": "Medium of Instruction",
        "tt": "Travel Time"
    }


    category_labels = {
        "Vg": "Very Good",
        "G": "General",
        "ST": "Scheduled Tribe (ST)",
        "SC": "Scheduled Caste (SC)",
        "OBC": "Other Backward Class (OBC)",
        "MOBC": "More Other Backward Class (MOBC)",
        "T": "Town",
        "V": "Village",
        "Free": "Free Admission",
        "Paid": "Paid Admission",
        "Y": "Yes",
        "N": "No",
        "Il": "Illiterate",
        "Um": "Under Matric",
        "Pg": "Post graduate",
        "Am": "Average",
        "Vh": "Very High"
    }


    # =========================
    # MOST INFLUENTIAL FACTOR
    # =========================

    first_factor = top_factors.iloc[0]

    feature = first_factor["feature"]
    shap_value = first_factor["shap_value"]

    original_feature, category = feature.split("_", 1)

    label = feature_labels.get(
        original_feature,
        original_feature
    )

    category_display = category_labels.get(
        category,
        category
    )

    st.subheader("⭐ Most Influential Factor")

    if shap_value > 0:

        st.success(
            f"**{label}: {category_display}**\n\n"
            "🟢 Positively influenced this prediction."
        )

    else:

        st.error(
            f"**{label}: {category_display}**\n\n"
            "🔴 Negatively influenced this prediction."
        )


    # =========================
    # OTHER FACTORS
    # =========================

    st.subheader("📊 Other Factors Influencing This Prediction")

    for _, row in top_factors.iloc[1:].iterrows():

        feature = row["feature"]
        shap_value = row["shap_value"]

        original_feature, category = feature.split("_", 1)

        label = feature_labels.get(
            original_feature,
            original_feature
        )

        category_display = category_labels.get(
            category,
            category
        )

        if shap_value > 0:

            st.write(
                f"🟢 **{label}: {category_display}** — "
                "Positive influence"
            )

        else:

            st.write(
                f"🔴 **{label}: {category_display}** — "
                "Negative influence"
            )


    # =========================
    # BACK BUTTON
    # =========================

    st.divider()

    if st.button(
        "← Back to Student Information",
        key="back_to_student"
    ):
        st.session_state.page = 1
        st.rerun()