"""
==============================================================================
08: Multiple Linear Regression, Controls, and Diagnostic Visualization
==============================================================================

Learning Objectives:
  - Understand the logic of statistical control and confounding variables
  - Specify multiple regression models using statsmodels formula syntax
  - Incorporate categorical predictors and reference levels using C()
  - Estimate nested models: Baseline -> Demographics -> Political Controls
  - Compare model performance using summary_col, R², and adjusted R²
  - Export publication-ready regression tables to Typst using mmisc.typst
  - Visualize regression results using dot-and-whisker coefficient plots
  - Interpret substantive effects and confidence intervals in political research
==============================================================================
"""

# ==============================================================================
# The Logic of Multiple Regression & Statistical Control
# ==============================================================================
# In observational political science, we cannot run laboratory experiments
# where we randomly assign political knowledge or ideological beliefs.
#
# Bivariate relationships can be spurious or confounded:
#   - An apparent relationship between X and Y may be driven by a third
#     variable (Z) that causes both X and Y.
#
# Multiple regression allows us to estimate the partial effect of X on Y
# while holding confounding variables constant:
#   Y_i = beta_0 + beta_1 * X_1i + beta_2 * X_2i + ... + beta_k * X_ki + e_i
#
# beta_1 represents the expected change in Y for a one-unit increase in X_1,
# holding X_2 through X_k constant (ceteris paribus).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from mmisc.typst import make_table


# ==============================================================================
# Research Question: Political Knowledge and Affective Polarization
# ==============================================================================
# Affective polarization is the tendency of citizens to distrust, dislike,
# and avoid members of the opposing political party.
#
# Hypotheses:
#   - H1 (Engagement/Polarization): Highly knowledgeable citizens hold stronger
#     partisan identities, leading to HIGHER affective polarization.
#   - H0 (Null): Knowledgeable citizens are more rational and open-minded,
#     leading to LOWER affective polarization.
#
# Confounders:
#   - Age, gender, education, income, and ideological extremity might predict
#     both political knowledge and partisan hostility.

data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Select key survey items:
vars_map = {
    "V201156": "feeling_democrat",       # Dem feeling thermometer (0-100)
    "V201157": "feeling_republican",     # Rep feeling thermometer (0-100)
    "V201644": "knowledge_senate_term",  # Senate term length
    "V201645": "knowledge_spending",     # Least federal spending area
    "V201646": "knowledge_house_maj",    # House majority party
    "V201647": "knowledge_senate_maj",   # Senate majority party
    "V201507x": "age",                   # Respondent age
    "V201600": "sex",                    # 1=Male, 2=Female
    "V201511x": "education",             # Education level (1 to 5)
    "V201617x": "income",                # Income brackets (1 to 22)
    "V201200": "ideology",               # Ideological self-placement (1 to 7)
    "V201228": "party_id",               # 1=Democrat, 2=Republican, 3=Indep
}

df = anes_data[list(vars_map.keys())].rename(columns=vars_map).copy()

# Step 1: Clean Dependent Variable (Affective Polarization)
ft_valid = (
    df["feeling_democrat"].between(0, 100)
    & df["feeling_republican"].between(0, 100)
)
df = df[ft_valid].copy()
df["affective_polarization"] = (
    df["feeling_democrat"] - df["feeling_republican"]
).abs()

# Step 2: Construct Objective Political Knowledge Scale (0 to 4)
df["k_senate_term"] = (df["knowledge_senate_term"] == 6).astype(int)
df["k_spending"] = (df["knowledge_spending"] == 1).astype(int)
df["k_house_maj"] = (df["knowledge_house_maj"] == 1).astype(int)
df["k_senate_maj"] = (df["knowledge_senate_maj"] == 2).astype(int)

knowledge_items = [
    "k_senate_term", "k_spending", "k_house_maj", "k_senate_maj"
]
df["political_knowledge"] = df[knowledge_items].sum(axis=1)

# Step 3: Clean Control Variables
df = df[df["age"] >= 18]
df = df[df["sex"].isin([1, 2])]
df["female"] = (df["sex"] == 2).astype(int)
df = df[df["education"] > 0]
df = df[df["income"] > 0]
df = df[df["ideology"].between(1, 7)]
df = df[df["party_id"].between(1, 3)]
df["party"] = df["party_id"].map({
    1: "Democrat",
    2: "Republican",
    3: "Independent",
})

# In the REPL, check cleaned sample size and variables:
df.shape
df.head()


# ==============================================================================
# Model 1: Bivariate Baseline Model
# ==============================================================================
# First, estimate the unadjusted bivariate relationship:

formula_m1 = "affective_polarization ~ political_knowledge"
model_1 = smf.ols(formula=formula_m1, data=df).fit()

# In the REPL, evaluate the baseline model:
model_1.summary()


# ------------------------------------------------------------------------------
# Hack-Time: Evaluating the Baseline Relationship
# ------------------------------------------------------------------------------
# Tasks:
#   1. In the REPL, inspect model_1.params["political_knowledge"].
#   2. Is the relationship positive or negative?
#   3. Does this initial bivariate evidence support H1 or H0?


# ==============================================================================
# Model 2: Adding Demographic Controls
# ==============================================================================
# Does the relationship between knowledge and polarization survive controlling
# for demographic characteristics (age, gender, education, income)?

formula_m2 = (
    "affective_polarization ~ political_knowledge + age + female + "
    "education + income"
)
model_2 = smf.ols(formula=formula_m2, data=df).fit()

# In the REPL, evaluate model 2:
model_2.summary()


# ==============================================================================
# Categorical Predictors in Patsy: The C() Operator
# ==============================================================================
# In political science, many key explanatory variables are qualitative:
# party affiliation, religious tradition, region, or race.
#
# If a categorical variable has k categories, a regression model requires
# k - 1 dummy variables. One category is omitted as the reference category
# against which other categories are compared.
#
# In statsmodels formula syntax, wrap categorical variables in C():
#   C(party)
#
# Statsmodels automatically chooses a reference category (e.g. Democrat)
# and estimates dummy coefficients:
#   - C(party)[T.Independent]: difference in Y between Independents and Dems.
#   - C(party)[T.Republican]: difference in Y between Republicans and Dems.


# ------------------------------------------------------------------------------
# Intentional Error Challenge: The Raw Numeric Categorical Fallacy
# ------------------------------------------------------------------------------
# In survey data, party_id is coded as 1=Democrat, 2=Republican, 3=Independent.
# What happens if you include party_id as a continuous number without C()?
#
# 1. Uncomment the model below and run it in the REPL with Shift+Enter.
# 2. Inspect the coefficient: what does a single slope for party_id assume?
# 3. Why is treating unordered parties as a continuous slope flawed?
# naive_model = smf.ols("affective_polarization ~ party_id", data=df).fit()
# naive_model.summary()


# ==============================================================================
# Model 3: Full Model with Political Controls
# ==============================================================================
# Now estimate the full model including demographic and political controls:

formula_m3 = (
    "affective_polarization ~ political_knowledge + age + female + "
    "education + income + ideology + C(party)"
)
model_3 = smf.ols(formula=formula_m3, data=df).fit()

# In the REPL, evaluate the full model:
model_3.summary()


# ==============================================================================
# Nested Model Comparison Tables with summary_col
# ==============================================================================
# Social scientists present regression models side by side to demonstrate
# how coefficients evolve as confounders are introduced:

table_models = [model_1, model_2, model_3]
model_names = ["Baseline", "Demographics", "Full Model"]

comparison_table = summary_col(
    table_models,
    model_names=model_names,
    stars=True,
    float_format="%.3f",
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: f"{x.rsquared:.3f}",
        "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}",
    },
)

# In the REPL, evaluate the comparison table:
print(comparison_table)


# ==============================================================================
# Exporting Regression Tables to Typst with mmisc
# ==============================================================================
# For term papers and milestone submissions written in Typst, mmisc.typst
# converts your statsmodels objects directly into native Typst markup:

make_table(table_models)


# ==============================================================================
# Visualizing Results: Dot-and-Whisker Coefficient Plots
# ==============================================================================
# While regression tables are detailed, coefficient plots are vastly more
# intuitive for conference presentations, posters, and executive summaries.
#
# A coefficient plot shows:
#   - Point: Estimated coefficient (beta).
#   - Horizontal bar: 95% Confidence Interval (beta +/- 1.96 * SE).
#   - Vertical reference line at 0: If the CI crosses 0, the effect is
#     not statistically significant at p < 0.05.

# Extract non-intercept coefficients and confidence intervals:
coef_df = pd.DataFrame({
    "coef": model_3.params,
    "lower_ci": model_3.conf_int()[0],
    "upper_ci": model_3.conf_int()[1],
    "pvalue": model_3.pvalues,
}).drop("Intercept")

# Label variables descriptively for presentation:
var_labels = {
    "political_knowledge": "Political Knowledge (0-4)",
    "age": "Respondent Age (years)",
    "female": "Female (vs. Male)",
    "education": "Education Level (1-5)",
    "income": "Household Income",
    "ideology": "Ideological Conservatism (1-7)",
    "C(party)[T.Independent]": "Independent (vs. Dem)",
    "C(party)[T.Republican]": "Republican (vs. Dem)",
}

coef_df.index = [var_labels.get(idx, idx) for idx in coef_df.index]

# Generate coefficient plot:
plt.figure(figsize=(9, 6))
plt.errorbar(
    coef_df["coef"],
    coef_df.index,
    xerr=(
        coef_df["coef"] - coef_df["lower_ci"],
        coef_df["upper_ci"] - coef_df["coef"],
    ),
    fmt="o",
    color="steelblue",
    ecolor="darkslategray",
    elinewidth=2,
    capsize=4,
)
plt.axvline(x=0, color="crimson", linestyle="--", linewidth=1.2)
plt.title(
    "Predictors of Affective Polarization (Full OLS Model)",
    fontsize=12, fontweight="bold"
)
plt.xlabel("Estimated Coefficient (Change in Polarization)")
plt.ylabel("Predictor Variable")
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# Hack-Time: Substantive Interpretation of Regression Results
# ------------------------------------------------------------------------------
# Tasks:
#   1. Inspect the coefficient for 'political_knowledge' in model_3.
#      Holding demographics, party, and ideology constant, what is the
#      estimated change in affective polarization between a citizen with
#      0 correct answers and a citizen with 4 correct answers?
#   2. Inspect the coefficient for 'C(party)[T.Independent]'.
#      How do Independents differ in polarization from Democrats?
#      Is this difference statistically significant?
#   3. Does education have a statistically significant effect once political
#      knowledge and ideology are accounted for?


