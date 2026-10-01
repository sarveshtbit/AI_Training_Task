import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import gradio as gr


# ==========================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ==========================================

model = tf.keras.models.load_model(
    "bank_marketing_ann.keras"
)

scaler = joblib.load(
    "bank_marketing_scaler.pkl"
)

feature_columns = joblib.load(
    "bank_marketing_features.pkl"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_subscription(
    age,
    job,
    marital,
    education,
    default,
    housing,
    loan,
    contact,
    month,
    day_of_week,
    campaign,
    pdays,
    previous,
    poutcome,
    emp_var_rate,
    cons_price_idx,
    cons_conf_idx,
    euribor3m,
    nr_employed
):

    # Create dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "day_of_week": [day_of_week],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "emp.var.rate": [emp_var_rate],
        "cons.price.idx": [cons_price_idx],
        "cons.conf.idx": [cons_conf_idx],
        "euribor3m": [euribor3m],
        "nr.employed": [nr_employed]
    })


    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    input_data["previously_contacted"] = (
        input_data["pdays"] != 999
    ).astype(int)


    input_data["multiple_contacts"] = (
        input_data["campaign"] > 1
    ).astype(int)


    input_data["age_group"] = pd.cut(
        input_data["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[
            "Young",
            "Adult",
            "Middle_Aged",
            "Senior",
            "Elderly"
        ]
    )


    input_data["previous_success"] = (
        input_data["poutcome"] == "success"
    ).astype(int)


    # ==========================================
    # ONE-HOT ENCODING
    # ==========================================

    categorical_columns = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
        "age_group"
    ]


    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True
    )


    # ==========================================
    # MATCH TRAINING FEATURES
    # ==========================================

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    input_data = input_data.astype(float)


    # ==========================================
    # SCALE DATA
    # ==========================================

    input_scaled = scaler.transform(
        input_data
    )


    # ==========================================
    # PREDICTION
    # ==========================================

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    if probability >= 0.5:

        prediction = (
            "Customer is likely to subscribe"
        )

    else:

        prediction = (
            "Customer is unlikely to subscribe"
        )


    return (
        prediction,
        f"{probability:.2%}"
    )


# ==========================================
# GRADIO INTERFACE
# ==========================================

demo = gr.Interface(

    fn=predict_subscription,

    inputs=[

        gr.Number(
            label="Age",
            value=35
        ),

        gr.Dropdown(
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ],
            label="Job",
            value="admin."
        ),

        gr.Dropdown(
            [
                "divorced",
                "married",
                "single",
                "unknown"
            ],
            label="Marital Status",
            value="single"
        ),

        gr.Dropdown(
            [
                "basic.4y",
                "basic.6y",
                "basic.9y",
                "high.school",
                "illiterate",
                "professional.course",
                "university.degree",
                "unknown"
            ],
            label="Education",
            value="university.degree"
        ),

        gr.Dropdown(
            ["no", "yes", "unknown"],
            label="Credit Default",
            value="no"
        ),

        gr.Dropdown(
            ["no", "yes", "unknown"],
            label="Housing Loan",
            value="no"
        ),

        gr.Dropdown(
            ["no", "yes", "unknown"],
            label="Personal Loan",
            value="no"
        ),

        gr.Dropdown(
            ["cellular", "telephone"],
            label="Contact Type",
            value="cellular"
        ),

        gr.Dropdown(
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ],
            label="Month",
            value="may"
        ),

        gr.Dropdown(
            [
                "mon",
                "tue",
                "wed",
                "thu",
                "fri"
            ],
            label="Day of Week",
            value="mon"
        ),

        gr.Number(
            label="Campaign Contacts",
            value=1
        ),

        gr.Number(
            label="Days Since Previous Contact",
            value=999
        ),

        gr.Number(
            label="Previous Contacts",
            value=0
        ),

        gr.Dropdown(
            [
                "failure",
                "nonexistent",
                "success"
            ],
            label="Previous Campaign Outcome",
            value="nonexistent"
        ),

        gr.Number(
            label="Employment Variation Rate",
            value=1.1
        ),

        gr.Number(
            label="Consumer Price Index",
            value=93.994
        ),

        gr.Number(
            label="Consumer Confidence Index",
            value=-36.4
        ),

        gr.Number(
            label="Euribor 3 Month Rate",
            value=4.857
        ),

        gr.Number(
            label="Number of Employees",
            value=5191
        )
    ],

    outputs=[
        gr.Textbox(
            label="Prediction"
        ),

        gr.Textbox(
            label="Subscription Probability"
        )
    ],

    title="Bank Marketing Term Deposit Prediction",

    description=(
        "Enter customer information to predict "
        "whether the customer is likely to subscribe "
        "to a bank term deposit."
    )
)


# ==========================================
# LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        share=False
    )