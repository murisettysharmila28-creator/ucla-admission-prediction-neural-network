import streamlit as st
from src.predict import predict_admission

st.set_page_config(page_title="UCLA Admission Prediction App", layout="centered")

st.title("UCLA Admission Prediction App")
st.markdown(
    "Predict whether an applicant is likely to have a strong admission chance using a neural network model."
)

st.divider()

gre_score = st.slider("GRE Score", 260, 340, 320)
toefl_score = st.slider("TOEFL Score", 0, 120, 105)
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.slider("Statement of Purpose Strength (SOP)", 1.0, 5.0, 3.5, step=0.5)
lor = st.slider("Letter of Recommendation Strength (LOR)", 1.0, 5.0, 3.5, step=0.5)
cgpa = st.slider("CGPA", 0.0, 10.0, 8.5, step=0.1)
research = st.selectbox(
    "Research Experience",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

if st.button("Predict Admission Outcome"):
    input_data = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
    }

    prediction, probability = predict_admission(input_data)

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.success("Likely to Have a Strong Admission Chance")
    else:
        st.error("Unlikely to Have a Strong Admission Chance")

    st.markdown(
        f"**Model-estimated probability of being in the high-admission-chance group:** {probability:.2%}"
    )

    st.markdown("### Interpretation")

    if probability >= 0.8:
        st.write("This profile shows a high likelihood of being in the strong admission chance group.")
    elif probability >= 0.6:
        st.write("This profile has a moderate outlook, but stronger academic or research indicators may improve the result.")
    else:
        st.write("This profile may need stronger academic or research credentials to improve the likelihood of being in the strong admission chance group.")