"""
==============================================================================
09: Categorical and Non-Linear Regression Models (Logit, Ordered, Multinomial)
==============================================================================

Learning Objectives:
  - Understand why OLS is unsuitable for qualitative dependent variables
  - Estimate binary logistic regression models using statsmodels.formula.api
  - Transform log-odds coefficients into intuitive Odds Ratios (OR = exp(beta))
  - Calculate and plot predicted probabilities (the S-curve) in Altair
  - Estimate Ordered Logistic Regression using statsmodels OrderedModel
  - Interpret threshold cutpoints and the proportional odds assumption
  - Estimate Multinomial Logit models using smf.mnlogit for multi-party vote
  - Compare substantive model predictions across political science applications
==============================================================================
"""

# ==============================================================================
# The Limits of OLS for Qualitative Outcomes
# ==============================================================================
# In political science, many critical dependent variables are discrete:
#   - Binary: Turnout (Vote vs. Abstain), Vote choice (Biden vs. Trump).
#   - Ordinal: Policy approval (Strongly Agree to Strongly Disagree).
#   - Nominal: Multi-party selection (Democrat, Republican, Third Party).
#
# Why can't we simply use Ordinary Least Squares (Linear Probability Model)?
#   1. Nonsensical Predictions: OLS can predict probabilities < 0 or > 1.
#   2. Non-linear Reality: The effect of a predictor tends to level off near
#      the extremes (0 and 1) rather than changing linearly forever.
#   3. Heteroskedasticity: The error variance of a binary outcome depends
#      on the probabilities (Var = P * (1 - P)), violating OLS assumptions.
#
# Generalized Linear Models (GLMs) solve this by using link functions that
# map predictions strictly within the valid mathematical bounds of probability.

import numpy as np
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ==============================================================================
# Binary Logistic Regression: Foundations & Estimation
# ==============================================================================
# The logistic regression model uses the logit link function:
#   logit(P) = ln( P / (1 - P) ) = beta_0 + beta_1 * X_1 + ... + beta_k * X_k
#
# Where:
#   - P is the probability of the event occurring (Y = 1).
#   - P / (1 - P) is the Odds of the event.
#   - ln( P / (1 - P) ) is the Log-Odds (logit).
#
# By estimating on the log-odds scale, predicted probabilities are guaranteed
# to remain bounded strictly between 0 and 1:
#   P = 1 / (1 + exp( -(beta_0 + beta_1 * X) ))

data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Select key variables from ANES 2020:
vars_map = {
    "V201033": "vote_int",   # 1=Biden, 2=Trump, 3=Jorgensen, 4=Hawkins
    "V201032": "turnout",    # 1=Intend to vote, 2=Do not intend
    "V201507x": "age",       # Respondent age
    "V201600": "sex",        # 1=Male, 2=Female
    "V201511x": "educ",      # Education level (1-5)
    "V201200": "ideology",   # Ideology (1=Extr Lib to 7=Extr Con)
    "V202406": "interest",   # 1=Very interested to 4=Not at all
}

df = anes_data[list(vars_map.keys())].rename(columns=vars_map).copy()

# Clean sample:
df = df[(df["age"] >= 18) & (df["ideology"].between(1, 7))].copy()
df["female"] = (df["sex"] == 2).astype(int)

# Create two-party presidential vote subset:
# 1 = Joe Biden, 0 = Donald Trump
two_party = df[df["vote_int"].isin([1, 2])].copy()
two_party["vote_biden"] = (two_party["vote_int"] == 1).astype(int)

# Fit binary logistic regression:
formula_logit = "vote_biden ~ ideology + age + female + educ"
logit_model = smf.logit(formula=formula_logit, data=two_party).fit()

# Run the code to evaluate the model summary:
logit_model.summary()


# ==============================================================================
# Interpreting Log-Odds vs. Odds Ratios
# ==============================================================================
# Logistic regression coefficients (beta) represent changes in LOG-ODDS.
# Because humans do not think in logarithms, social scientists exponentiate
# coefficients to obtain ODDS RATIOS (OR = exp(beta)):
#
#   - OR = 1.0 : No relationship between X and Y.
#   - OR > 1.0 : Positive relationship (higher odds of Y=1).
#   - OR < 1.0 : Negative relationship (lower odds of Y=1).
#
# Percentage change in odds = (OR - 1) * 100%

odds_ratios = pd.DataFrame({
    "Log-Odds (beta)": logit_model.params,
    "Odds Ratio (exp(b))": np.exp(logit_model.params),
    "p-value": logit_model.pvalues,
})

# Run the code to inspect odds ratios:
odds_ratios.round(3)

# Substantive interpretation:
# For ideology (beta ~ -1.61, OR ~ 0.20):
# Holding other factors constant, each 1-unit increase in conservatism
# multiplies the odds of voting for Biden by 0.20 (an 80% decrease in odds!).


# ==============================================================================
# Calculating and Plotting Predicted Probabilities (The S-Curve)
# ==============================================================================
# The most intuitive way to communicate logistic regression results to
# policymakers and the public is through predicted probabilities:
#   "What is the probability of voting for Biden across the ideological scale?"

# Construct a synthetic prediction grid:
ideo_range = np.linspace(1, 7, 100)
pred_grid = pd.DataFrame({
    "ideology": ideo_range,
    "age": two_party["age"].mean(),
    "female": 1,                     # Fixed at female respondent
    "educ": two_party["educ"].median(),
})

# Generate predicted probabilities:
pred_grid["prob_biden"] = logit_model.predict(pred_grid)

# Visualize the characteristic S-curve using Altair:
scurve_line = (
    alt.Chart(pred_grid)
    .mark_line(color="#2b5c8f", strokeWidth=3)
    .encode(
        x=alt.X(
            "ideology",
            type="quantitative",
            title="Ideological Self-Placement (1=Extr Lib to 7=Extr Con)",
        ),
        y=alt.Y(
            "prob_biden",
            type="quantitative",
            title="Predicted Probability of Voting for Biden",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
    )
)

threshold_rule = (
    alt.Chart()
    .mark_rule(color="gray", strokeDash=[4, 4], strokeWidth=1.5)
    .encode(y=alt.datum(0.5))
)

scurve_chart = (
    (scurve_line + threshold_rule)
    .properties(
        title=alt.TitleParams(
            text="Predicted Probability of Voting for Biden (2020 ANES)",
            subtitle=[
                "Logistic regression S-curve (median female respondent)",
                "Dashed line = 50% threshold",
            ],
            anchor="start",
        ),
        width=500,
        height=300,
    )
)
scurve_chart


# ------------------------------------------------------------------------------
# Intentional Error Challenge: The String Category Trap in Logit
# ------------------------------------------------------------------------------
# In statsmodels, passing unmapped string columns into binary logit will
# raise an error if the dependent variable is not numeric (0 or 1).
#
# 1. Uncomment the model below and run it with Shift+Enter.
# 2. Inspect the error message: what did Python raise?
# 3. Why must binary dependent variables be encoded as 0 and 1?
# broken_df = two_party.copy()
# broken_df["vote_str"] = broken_df["vote_biden"].map({1: "Yes", 0: "No"})
# broken_logit = smf.logit("vote_str ~ ideology", data=broken_df).fit()


# ------------------------------------------------------------------------------
# Hack-Time: Modeling Voter Turnout Intention
# ------------------------------------------------------------------------------
# In the ANES, 'turnout' is coded 1 = Intends to vote, 2 = Does not intend.
#
# Tasks:
#   1. Create binary indicator 'vote_turnout' (1 if turnout==1, 0 if 2).
#   2. Estimate a binary logit model predicting 'vote_turnout' from 'age',
#      'educ', and 'ideology'.
#   3. Exponentiate the coefficients to compute Odds Ratios.
#   4. Which predictor has the largest multiplicative effect on the odds
#      of turning out to vote?

...


# ==============================================================================
# Ordered Logistic Regression: Ranked Survey Responses
# ==============================================================================
# In political science, attitudes are often measured on Likert scales:
#   Political Interest: 1 = Very, 2 = Somewhat, 3 = Slightly, 4 = Not at all.
#
# OLS assumes the distance between "Very" and "Somewhat" equals the distance
# between "Slightly" and "Not at all". This assumption is rarely justified.
#
# Ordered logit estimates latent continuous propensities divided by
# threshold cutpoints (thresholds theta_1, theta_2, theta_3):

df_ord = df[df["interest"].between(1, 4)].copy()

# Fit Ordered Logit using statsmodels OrderedModel:
ord_model = OrderedModel.from_formula(
    "interest ~ age + ideology",
    data=df_ord,
    distr="logit",
).fit(disp=False)

# Run the code to evaluate the ordered logit model:
ord_model.summary()

# Interpretation:
#   - Coefficients show the shift in latent political interest.
#   - Cutpoints (e.g. 1.0/2.0, 2.0/3.0) represent estimated category thresholds.


# ==============================================================================
# Multinomial Logistic Regression: Multi-Party Electoral Choice
# ==============================================================================
# In multi-party democracies, or in elections with third-party alternatives,
# the outcome is categorical and unordered:
#   0 = Joe Biden (Democrat) - Reference Category
#   1 = Donald Trump (Republican)
#   2 = Third Party / Other (Libertarian, Green, Independent)
#
# Multinomial Logit estimates J - 1 sets of log-odds relative to the baseline:
#   ln( P(Trump) / P(Biden) ) = beta_Trump * X
#   ln( P(Other) / P(Biden) ) = beta_Other * X

df_mn = df[df["vote_int"].between(1, 5)].copy()
# Encode integer categories: 0=Biden, 1=Trump, 2=Other
df_mn["vote_code"] = df_mn["vote_int"].map({
    1: 0,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
})

mn_model = smf.mnlogit(
    formula="vote_code ~ ideology + age",
    data=df_mn,
).fit(disp=False)

# Run the code to evaluate the multinomial model:
mn_model.summary()


# ------------------------------------------------------------------------------
# Hack-Time: Predicting Multi-Party Probabilities
# ------------------------------------------------------------------------------
# Using the fitted multinomial model, predict the probability distribution
# over candidate choices (Biden, Trump, Other) for two hypothetical voters:
#   - Voter A: 21-year-old liberal (age = 21, ideology = 2)
#   - Voter B: 68-year-old conservative (age = 68, ideology = 6)
#
# Tasks:
#   1. Create a DataFrame containing profiles for Voter A and Voter B.
#   2. Use mn_model.predict(profiles_df) to compute predicted probabilities.
#   3. Label columns as ["P(Biden)", "P(Trump)", "P(Other)"].
#   4. Which candidate is the predicted winner for each voter profile?

...
