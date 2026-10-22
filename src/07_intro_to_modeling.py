"""
==============================================================================
07: Introduction to Statistical Modeling and Bivariate Regression
==============================================================================

Learning Objectives:
  - Conceptualize models as simplifications, starting with the mean
  - Understand OLS geometry, residuals, and Sum of Squared Errors
  - Estimate and interpret bivariate regression models with statsmodels
  - Layer regression lines and residual drop-lines in Altair
==============================================================================
"""

# ==============================================================================
# What is a Statistical Model?
# ==============================================================================
# In empirical political science, the social world is infinitely complex.
# We cannot observe every interaction, attitude, or idiosyncratic shock.
#
# A model is a simplified representation of reality that allows us to:
#   1. Summarize systematic relationships between variables.
#   2. Separate signal (underlying pattern) from noise (random error).
#   3. Make predictions for unobserved cases.
#   4. Test theoretical hypotheses about causal mechanisms.
#
# As statistician George Box famously stated:
# "All models are wrong, but some are useful."
#
# A model is useful if it captures the essential structure of a phenomenon
# better than a naive guess while remaining interpretable.

import numpy as np
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf


# ==============================================================================
# The Mean as the Simplest Baseline Model
# ==============================================================================
# Before adding complex predictors, what is the simplest model we can build?
# The sample mean (constant model):
#   Y_hat = Y_bar
#
# Consider a real-world example: Commute times (in minutes) to Sciences Po
# Bordeaux recorded over 50 consecutive trips:

commute_df = pd.DataFrame({
    "time_to_iep": [
        16.93, 19.49, 18.21, 19.09, 17.67, 18.48, 16.37, 17.57, 19.18, 18.74,
        17.15, 17.76, 17.20, 19.78, 18.34, 17.93, 18.09, 17.14, 19.41, 17.99,
        16.54, 18.42, 16.65, 19.83, 18.32, 18.13, 16.72, 18.05, 18.50, 19.45,
        17.22, 17.32, 19.48, 18.93, 18.69, 18.78, 18.58, 18.80, 18.28, 20.06,
        18.12, 18.64, 18.16, 17.44, 18.96, 17.55, 19.09, 17.95, 21.01, 18.19
    ]
})

# Run the code to calculate the sample mean commute time:
mean_commute = commute_df["time_to_iep"].mean()
mean_commute

# If someone asks: "How long will tomorrow's commute take?", our best baseline
# prediction in the absence of any other information is the mean (~18.29 min).
#
# For each individual trip (i), the prediction error (residual) is:
#   e_i = Y_i - Y_bar

residuals = commute_df["time_to_iep"] - mean_commute

# Sum of Squared Errors (SSE):
# Key principle: The sample mean mathematically minimizes squared errors!
# Any other constant prediction would yield a larger SSE.
sse_baseline = (residuals ** 2).sum()
sse_baseline

# Visualize the baseline model with Altair:
base_hist = (
    alt.Chart(commute_df)
    .mark_bar(color="skyblue")
    .encode(
        x=alt.X(
            "time_to_iep",
            type="quantitative",
            bin=alt.Bin(maxbins=12),
            title="Commute Time (minutes)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Students",
        ),
    )
)

mean_rule = (
    alt.Chart(pd.DataFrame({"mean_val": [mean_commute]}))
    .mark_rule(color="crimson", strokeWidth=2, strokeDash=[5, 5])
    .encode(
        x=alt.X("mean_val", type="quantitative"),
    )
)

baseline_chart = (
    (base_hist + mean_rule)
    .properties(
        title=alt.TitleParams(
            text="Baseline Model: Commute Times to Sciences Po Bordeaux",
            subtitle=[
                f"Sample mean = {mean_commute:.2f} minutes (minimizes SSE)",
            ],
            anchor="start",
        ),
        width=450,
        height=280,
    )
)
baseline_chart


# ==============================================================================
# From Constant to Line: Adding an Explanatory Variable (X)
# ==============================================================================
# Can we improve upon the baseline mean?
# Yes, by using information from an independent variable (X).
#
# A linear regression model assumes a straight-line relationship:
#   Y_i = beta_0 + beta_1 * X_i + e_i
#
# Where:
#   - Y_i is the observed outcome (Dependent Variable).
#   - X_i is the predictor (Independent Variable).
#   - beta_0 is the Intercept (predicted Y when X = 0).
#   - beta_1 is the Slope (expected change in Y for a 1-unit increase in X).
#   - e_i is the Residual / error term (Y_i - Y_hat_i).


# ==============================================================================
# The Geometry of Ordinary Least Squares (OLS)
# ==============================================================================
# Why is OLS called "Ordinary Least Squares"?
# Infinitely many straight lines can be drawn through a scatter plot.
# How do we choose the "best" line?
#
# Ordinary Least Squares mathematically finds the unique line (beta_0, beta_1)
# that minimizes the Sum of Squared Residuals:
#   SSR = sum( (Y_i - Y_hat_i)^2 )
#
# Let us demonstrate this empirically using random candidate lines:

np.random.seed(42)
x_vals = commute_df.index.values
y_vals = commute_df["time_to_iep"].values

# Evaluate 5 random candidate lines:
candidate_lines = [
    (17.0, 0.05),
    (19.0, -0.04),
    (16.5, 0.08),
    (18.0, 0.01),
    (18.29, 0.00),  # Flat mean line
]

print("Candidate Line Comparisons:")
for b0, b1 in candidate_lines:
    y_pred = b0 + b1 * x_vals
    ssr = ((y_vals - y_pred) ** 2).sum()
    print(f"Line (b0={b0:.2f}, b1={b1:+.2f}) -> SSR = {ssr:.2f}")


# ==============================================================================
# Estimating Bivariate OLS on Survey Data (ANES 2020)
# ==============================================================================
# Let us test a core hypothesis in political behavior:
# Are older citizens systematically more conservative than younger citizens?
#
# Theoretical mechanism:
# Socialization, life-cycle wealth accumulation, and cohort generational
# differences are often hypothesized to shift voters toward conservatism.
#
# Variables:
#   - Dependent Variable (Y): Ideology (1=Extr Lib to 7=Extr Con)
#   - Independent Variable (X): Respondent Age (years)

data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Clean and subset variables:
clean_data = (
    anes_data[["V201507x", "V201200"]]
    .rename(columns={"V201507x": "age", "V201200": "ideology"})
)

clean_data = clean_data[
    (clean_data["ideology"].between(1, 7)) & (clean_data["age"] >= 18)
].copy()

# Fit bivariate regression using statsmodels formula API:
# Formula syntax: "dependent_variable ~ independent_variable"
bivariate_model = smf.ols(formula="ideology ~ age", data=clean_data).fit()

# Run the code to inspect the full estimation summary:
bivariate_model.summary()


# ==============================================================================
# Interpreting the Regression Summary Table
# ==============================================================================
# When reading the statsmodels output table, focus on four key statistics:
#
# 1. Intercept (beta_0 = 3.27):
#    The expected ideological score for someone with age = 0.
#    Substantively, age = 0 is outside our survey age range (18 to 80),
#    so the intercept serves primarily as an anchor for the regression line.
#
# 2. Slope (beta_1 = +0.0156):
#    For every additional year of age, expected conservatism increases by
#    0.0156 units on the 1-7 scale.
#    Over a 30-year life-cycle difference, expected ideology shifts by:
#    0.0156 * 30 = +0.47 units toward conservatism.
#
# 3. p-value (P > |t| < 0.001):
#    The probability of observing a slope this large if the true population
#    relationship were zero. Since p < 0.05, we reject the null hypothesis
#    and conclude the age-ideology relationship is statistically significant.
#
# 4. R-squared (R² = 0.0256):
#    Age accounts for 2.56% of the variance in ideological self-placement.
#    While the effect is statistically reliable, 97.4% of ideological variation
#    is explained by factors other than age (education, race, party ID, etc.).


# ==============================================================================
# Visualizing the Regression Line and Residuals
# ==============================================================================
# Visualizing fitted models builds intuition for what regression achieves.

# Draw a small sample of 40 respondents to visualize residual drop-lines:
sample_viz = clean_data.sample(40, random_state=42).sort_values("age")
sample_model = smf.ols("ideology ~ age", data=sample_viz).fit()
sample_viz["predicted"] = sample_model.fittedvalues

# Visualize regression line, observed points, and residual drop-lines:
scatter_pts = (
    alt.Chart(sample_viz)
    .mark_circle(size=60, color="steelblue")
    .encode(
        x=alt.X(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
        y=alt.Y(
            "ideology",
            type="quantitative",
            title="Ideological Self-Placement (1=Lib to 7=Con)",
        ),
    )
)

reg_line = (
    alt.Chart(sample_viz)
    .mark_line(color="crimson", strokeWidth=2.5)
    .encode(
        x=alt.X("age", type="quantitative"),
        y=alt.Y("predicted", type="quantitative"),
    )
)

residual_rules = (
    alt.Chart(sample_viz)
    .mark_rule(color="gray", strokeDash=[3, 3], opacity=0.7)
    .encode(
        x=alt.X("age", type="quantitative"),
        y=alt.Y("ideology", type="quantitative"),
        y2=alt.Y2("predicted"),
    )
)

reg_chart = (
    (residual_rules + reg_line + scatter_pts)
    .properties(
        title=alt.TitleParams(
            text="OLS Bivariate Regression: Ideology ~ Age",
            subtitle=[
                "Red line = OLS fit; gray dashed lines = residual errors (e_i)",
            ],
            anchor="start",
        ),
        width=500,
        height=320,
    )
)
reg_chart


# ------------------------------------------------------------------------------
# Intentional Error Challenge: The Formula Operator
# ------------------------------------------------------------------------------
# In formula syntax, the relationship between DV and IV is specified using
# the tilde '~' (read as "modeled as a function of"), NOT the equals sign '='.
#
# 1. Uncomment the line below and run it with Shift+Enter.
# 2. Read the error message: what error did Python raise?
# 3. Fix the syntax and run it again.
# broken_model = smf.ols(formula="ideology = age", data=clean_data).fit()


# ------------------------------------------------------------------------------
# Hack-Time: Comparing Baseline Error with Regression Error
# ------------------------------------------------------------------------------
# Tasks:
#   1. Calculate the baseline Sum of Squared Errors using only the sample mean:
#      mean_y = clean_data["ideology"].mean()
#      sse_baseline = ((clean_data["ideology"] - mean_y) ** 2).sum()
#   2. Inspect the Sum of Squared Residuals from the fitted regression model
#      using bivariate_model.ssr.
#   3. Calculate R² manually:
#      r_squared = 1 - (bivariate_model.ssr / sse_baseline)
#   4. Verify that this manual calculation matches bivariate_model.rsquared.

...


# ------------------------------------------------------------------------------
# Hack-Time: Modeling Political Trust as a Function of Ideology
# ------------------------------------------------------------------------------
# Theoretical question: Are conservatives or liberals more trusting of the
# federal government in 2020?
#
# Tasks:
#   1. Extract 'V201233' (trust in government: 1=Always to 5=Never) from
#      anes_data and merge it with 'clean_data'.
#   2. Clean the trust variable by replacing negative non-responses with NaN.
#   3. Estimate a bivariate regression:
#      formula = "trust_gov ~ ideology"
#      trust_model = smf.ols(formula, data=df_trust).fit()
#   4. Run trust_model.summary() to inspect the results.
#   5. Is the slope positive or negative? Is it statistically significant?
#      What does this imply about trust under the Trump presidency?

...
