"""
==============================================================================
Playground: Political Knowledge & Affective Polarization
==============================================================================

Research Question: How does political knowledge affect affective polarization?
H1: Higher political knowledge leads to higher affective polarization.

Execute line by line using Shift+Enter in VS Code.
==============================================================================
"""

import numpy as np
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col


# ==============================================================================
# Step 1: Load, Select, and Rename Variables
# ==============================================================================
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

vars_map = {
    "V201033": "vote_intention",
    "V201507x": "age",
    "V201600": "sex",
    "V201511x": "education",
    "V201617x": "income",
    "V201228": "party_id",
    "V201231x": "party_id_str",
    "V201232": "party_id_imp",
    "V201200": "ideology",
    "V201156": "feeling_democrat",
    "V201157": "feeling_republican",
    "V201642": "political_knowledge_catch1",
    "V201644": "political_knowledge_senate_term",
    "V201645": "political_knowledge_least_spending",
    "V201646": "political_knowledge_house_majority",
    "V201647": "political_knowledge_senate_majority",
    "V202406": "political_interest",
    "V202407": "follow_politics_media",
    "V202408": "understand_issues",
}

df = anes_data[list(vars_map.keys())].rename(columns=vars_map).copy()


# ==============================================================================
# Step 2: Clean Variables & Build Political Knowledge Scale
# ==============================================================================
def clean_knowledge_variable(series, correct_values):
    series_cleaned = series.replace([-9, -5, -4, -1], np.nan)
    return series_cleaned.apply(lambda x: 1 if x in correct_values else 0)


df["political_knowledge_senate_term"] = clean_knowledge_variable(
    df["political_knowledge_senate_term"], [6]
)
df["political_knowledge_least_spending"] = clean_knowledge_variable(
    df["political_knowledge_least_spending"], [1]
)
df["political_knowledge_house_majority"] = clean_knowledge_variable(
    df["political_knowledge_house_majority"], [1]
)
df["political_knowledge_senate_majority"] = clean_knowledge_variable(
    df["political_knowledge_senate_majority"], [2]
)

# Apply awareness check filter:
df = df[df["political_knowledge_catch1"].between(1000, 2000)].copy()

pk_vars = [
    "political_knowledge_senate_term",
    "political_knowledge_least_spending",
    "political_knowledge_house_majority",
    "political_knowledge_senate_majority",
]
df["political_knowledge_scale"] = df[pk_vars].sum(axis=1)
df["pk_dummy"] = (df["political_knowledge_scale"] == 4).astype(int)

# Filter valid respondent demographics and attitudes:
df = df[
    (df["age"] >= 18)
    & df["sex"].between(1, 2)
    & (df["education"] > 0)
    & (df["income"] > 0)
    & df["ideology"].between(1, 7)
    & df["party_id"].between(1, 3)
    & df["party_id_imp"].between(1, 5)
    & df["vote_intention"].between(1, 2)
    & (df["political_interest"] > 0)
    & (df["follow_politics_media"] > 0)
    & (df["understand_issues"] > 0)
    & (df["feeling_democrat"] >= 0)
    & (df["feeling_republican"] >= 0)
].copy()

df["sex"] = df["sex"].map({1: "Male", 2: "Female"})


# ==============================================================================
# Step 3: Build Quantity of Interest & Visualize Distributions
# ==============================================================================
df["affective_polarization"] = np.abs(
    df["feeling_democrat"] - df["feeling_republican"]
)

# DV distribution (histogram):
dv_chart = (
    alt.Chart(df)
    .mark_bar(color="#4c78a8")
    .encode(
        x=alt.X(
            "affective_polarization",
            type="quantitative",
            bin=alt.Bin(maxbins=15),
            title="Affective Polarization (0-100)",
        ),
        y=alt.Y("count()", type="quantitative", title="Number of Respondents"),
    )
    .properties(
        title="Distribution of Affective Polarization (DV)",
        width=450,
        height=300,
    )
)
dv_chart

# IV distribution (bar chart):
iv_chart = (
    alt.Chart(df)
    .mark_bar(color="#e45756")
    .encode(
        x=alt.X(
            "political_knowledge_scale",
            type="ordinal",
            title="Political Knowledge Scale (0-4)",
        ),
        y=alt.Y("count()", type="quantitative", title="Number of Respondents"),
    )
    .properties(
        title="Distribution of Political Knowledge Scale (IV)",
        width=450,
        height=300,
    )
)
iv_chart


# ==============================================================================
# Step 4: Modeling with statsmodels Formula API
# ==============================================================================
formula_base = "affective_polarization ~ political_knowledge_scale"
model_base = smf.ols(formula=formula_base, data=df).fit()

formula_full = (
    "affective_polarization ~ political_knowledge_scale + age + C(sex) "
    "+ education + ideology"
)
model_full = smf.ols(formula=formula_full, data=df).fit()

model_full.summary()

# Compare models side by side:
summary_col([model_base, model_full], stars=True)


# ==============================================================================
# Step 5: Visualize Regression Results in Altair
# ==============================================================================
coef_df = pd.DataFrame({
    "coef": model_full.params,
    "lower_ci": model_full.conf_int()[0],
    "upper_ci": model_full.conf_int()[1],
}).drop("Intercept")

plot_df = coef_df.reset_index()
plot_df.columns = ["variable", "coef", "lower_ci", "upper_ci"]

whiskers = (
    alt.Chart(plot_df)
    .mark_rule(color="#4c78a8", strokeWidth=2)
    .encode(
        x=alt.X(
            "lower_ci",
            type="quantitative",
            title="Estimated Coefficient (95% CI)",
        ),
        x2=alt.X2("upper_ci"),
        y=alt.Y("variable", type="nominal", title="Variable", sort=None),
    )
)

points = (
    alt.Chart(plot_df)
    .mark_circle(size=70, color="#4c78a8")
    .encode(
        x=alt.X("coef", type="quantitative"),
        y=alt.Y("variable", type="nominal", sort=None),
    )
)

zero_rule = (
    alt.Chart()
    .mark_rule(color="red", strokeDash=[4, 4])
    .encode(x=alt.datum(0))
)

(zero_rule + whiskers + points).properties(
    title="Regression Coefficients with 95% Confidence Intervals",
    width=450,
    height=280,
)

# Actual vs. Predicted polarization across sample:
df["predicted"] = model_full.predict(df)
df_sample = df.sample(min(400, len(df)), random_state=42)

pred_chart = (
    alt.Chart(df_sample)
    .mark_circle(size=50, opacity=0.6, color="#4c78a8")
    .encode(
        x=alt.X(
            "predicted",
            type="quantitative",
            title="Predicted Affective Polarization",
        ),
        y=alt.Y(
            "affective_polarization",
            type="quantitative",
            title="Actual Affective Polarization",
        ),
    )
    .properties(
        title="Actual vs. Predicted Affective Polarization",
        width=450,
        height=300,
    )
)
pred_chart
