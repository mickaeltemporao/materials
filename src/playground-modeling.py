"""
==============================================================================
Playground: Statistical Modeling & Linear Regression
==============================================================================

Execute line by line using Shift+Enter in VS Code.
==============================================================================
"""

import numpy as np
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf


# ==============================================================================
# Step 1: Load, Select, and Clean Variables (ANES 2020)
# ==============================================================================
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

clean_data = (
    anes_data[["V201507x", "V201200"]]
    .rename(columns={"V201507x": "age", "V201200": "ideology"})
)
clean_data = clean_data[
    clean_data["ideology"].between(1, 7) & (clean_data["age"] >= 18)
].copy()


# ==============================================================================
# Step 2: Baseline Model (The Mean) & Random Lines
# ==============================================================================
# Generating random candidate lines:
np.random.seed(42)
models = pd.DataFrame({
    "a1": np.random.uniform(-5, 5, 50),
    "a2": np.random.uniform(-0.25, 0.25, 50),
})

# Sample commute time data (minutes to Sciences Po Bordeaux):
my_data = pd.DataFrame({
    "time_to_iep": [
        16.93, 19.49, 18.21, 19.09, 17.67, 18.48, 16.37, 17.57, 19.18, 18.74,
        17.15, 17.76, 17.20, 19.78, 18.34, 17.93, 18.09, 17.14, 19.41, 17.99,
        16.54, 18.42, 16.65, 19.83, 18.32, 18.13, 16.72, 18.05, 18.50, 19.45,
        17.22, 17.32, 19.48, 18.93, 18.69, 18.78, 18.58, 18.80, 18.28, 20.06,
        18.12, 18.64, 18.16, 17.44, 18.96, 17.55, 19.09, 17.95, 21.01, 18.19,
    ]
})

mean_value = my_data["time_to_iep"].mean()

# Simple Altair histogram + mean reference line:
hist_chart = (
    alt.Chart(my_data)
    .mark_bar(color="#4c78a8")
    .encode(
        x=alt.X(
            "time_to_iep",
            type="quantitative",
            bin=alt.Bin(maxbins=10),
            title="Time to IEP (minutes)",
        ),
        y=alt.Y("count()", type="quantitative", title="Frequency"),
    )
)

mean_line = (
    alt.Chart(pd.DataFrame({"mean_val": [mean_value]}))
    .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
    .encode(x=alt.X("mean_val", type="quantitative"))
)

(hist_chart + mean_line).properties(
    title=f"Histogram of Time to IEP (Mean = {mean_value:.2f})",
    width=450,
    height=300,
)


# ------------------------------------------------------------------------------
# Hack-Time: Explore Commute Errors
# ------------------------------------------------------------------------------

...


# ==============================================================================
# Step 3: Scatter Plot & Regression Line with Altair
# ==============================================================================
vars_map = {
    "V201507x": "age",
    "V201200": "ideology",
    "V201228": "party_id",
    "V201511x": "education",
    "V201600": "gender",
}

tb = anes_data[list(vars_map.keys())].rename(columns=vars_map).copy()
tb = tb[
    (tb["age"] >= 18)
    & tb["ideology"].between(1, 7)
    & tb["party_id"].between(1, 3)
    & (tb["education"] > 0)
    & tb["gender"].between(1, 2)
].copy()

tb["party_id"] = tb["party_id"].map({
    1: "Democrat",
    2: "Republican",
    3: "Independent",
})
tb["gender"] = tb["gender"].map({1: "Male", 2: "Female"})

# Always inspect the data first:
tb.head()

# Sample 400 rows for readable scatter plotting:
tb_sample = tb.sample(400, random_state=42)

scatter_chart = (
    alt.Chart(tb_sample)
    .mark_circle(size=45, opacity=0.5, color="#4c78a8")
    .encode(
        x=alt.X("age", type="quantitative", title="Age"),
        y=alt.Y("ideology", type="quantitative", title="Ideology (1-7)"),
    )
    .properties(title="Ideology vs. Age (Sample)", width=450, height=300)
)
scatter_chart

# Fit OLS line and overlay on the scatter plot:
model_age = smf.ols(formula="ideology ~ age", data=tb).fit()
tb_sample["predicted"] = model_age.predict(tb_sample)

line_chart = (
    alt.Chart(tb_sample)
    .mark_line(color="red", strokeWidth=2.5)
    .encode(
        x=alt.X("age", type="quantitative"),
        y=alt.Y("predicted", type="quantitative"),
    )
)

(scatter_chart + line_chart).properties(
    title="Ideology vs. Age with OLS Line of Best Fit",
    width=450,
    height=300,
)


# ==============================================================================
# Step 4: Linear Models with statsmodels Formula API
# ==============================================================================
# 1. Ideology as a function of age:
model_age.summary()

# 2. Effect of party identification on ideology:
model_party = smf.ols(formula="ideology ~ C(party_id)", data=tb).fit()
model_party.summary()

# 3. Effect of education on ideology:
model_edu = smf.ols(formula="ideology ~ education", data=tb).fit()
model_edu.summary()

# 4. Effect of gender on ideology:
model_gender = smf.ols(formula="ideology ~ C(gender)", data=tb).fit()
model_gender.summary()

# 5. Multiple regression controlling for all predictors:
full_formula = "ideology ~ age + C(party_id) + education + C(gender)"
model_full = smf.ols(formula=full_formula, data=tb).fit()
model_full.summary()
