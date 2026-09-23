"""
==============================================================================
10: Data Visualization, Declarative Graphics, and Communication
==============================================================================

Learning Objectives:
  - Translate regression results into intuitive, communicative visual narratives
  - Construct reusable functions to generate dot-and-whisker coefficient plots
  - Visualize segmented distributions using grouped kernel density estimation
  - Master declarative graphics in Python using Altair's grammar of graphics
  - Map variables to visual channels using Altair data types (:Q, :N, :O, :T)
  - Build interactive charts with linked selections, tooltips, and brushing
  - Prepare publication-ready graphics for Typst papers and presentations
==============================================================================
"""

# ==============================================================================
# The Visual Narrative in Empirical Political Science
# ==============================================================================
# In academic papers and policy reports, statistical tables are useful for
# archiving complete numbers, but graphs are essential for human communication.
#
# As Kastellec & Leoni (2007) argue in "Using Graphs Instead of Tables in
# Political Science":
#   "Readers perceive differences in position and length far more accurately
#    than differences in numerical values printed in large tables."
#
# This script covers three professional visualization workflows:
#   1. Functionalizing Matplotlib/Seaborn for reproducible coefficient plots.
#   2. Grouped kernel density estimation to inspect distribution shapes.
#   3. Modern declarative graphics with Altair for interactive dashboards.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import altair as alt


# ==============================================================================
# Loading ANES Data and Estimating Regression Models
# ==============================================================================
# We load the 2020 ANES survey to prepare a complete empirical pipeline:

data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Select key variables:
vars_map = {
    "V201156": "feeling_democrat",
    "V201157": "feeling_republican",
    "V201644": "knowledge_senate_term",
    "V201645": "knowledge_spending",
    "V201646": "knowledge_house_maj",
    "V201647": "knowledge_senate_maj",
    "V201507x": "age",
    "V201600": "sex",
    "V201511x": "education",
    "V201200": "ideology",
    "V201228": "party_id",
    "V202406": "interest",
}

df = anes_data[list(vars_map.keys())].rename(columns=vars_map).copy()

# Clean sample:
ft_valid = (
    df["feeling_democrat"].between(0, 100)
    & df["feeling_republican"].between(0, 100)
)
df = df[ft_valid].copy()
df["affective_polarization"] = (
    df["feeling_democrat"] - df["feeling_republican"]
).abs()

# Construct 0-4 political knowledge scale:
df["k_term"] = (df["knowledge_senate_term"] == 6).astype(int)
df["k_spend"] = (df["knowledge_spending"] == 1).astype(int)
df["k_house"] = (df["knowledge_house_maj"] == 1).astype(int)
df["k_senate"] = (df["knowledge_senate_maj"] == 2).astype(int)
df["political_knowledge"] = df[
    ["k_term", "k_spend", "k_house", "k_senate"]
].sum(axis=1)

# Clean demographics and politics:
df = df[df["age"] >= 18]
df = df[df["sex"].isin([1, 2])]
df["female"] = (df["sex"] == 2).astype(int)
df = df[df["education"] > 0]
df = df[df["ideology"].between(1, 7)]
df = df[df["party_id"].between(1, 3)]
df["party"] = df["party_id"].map({
    1: "Democrat",
    2: "Republican",
    3: "Independent",
})
df = df[df["interest"].between(1, 4)]
df["interest_label"] = df["interest"].map({
    1: "Very Interested",
    2: "Somewhat Interested",
    3: "Slightly Interested",
    4: "Not Interested",
})

# Estimate multiple regression model:
formula = (
    "affective_polarization ~ political_knowledge + age + female + "
    "education + ideology + C(party)"
)
full_model = smf.ols(formula=formula, data=df).fit()


# ==============================================================================
# Functionalizing the Coefficient Plot
# ==============================================================================
# In scientific research, you estimate dozens of models. Writing plotting code
# from scratch each time violates the DRY (Don't Repeat Yourself) principle.
#
# Let us encapsulate the coefficient plot into a reusable function:

def plot_coefficients(model, title="Model Coefficients (95% CI)"):
    """
    Extracts regression coefficients and 95% confidence intervals from a
    fitted statsmodels object and displays a dot-and-whisker plot.
    """
    coef_df = pd.DataFrame({
        "coef": model.params,
        "lower_ci": model.conf_int()[0],
        "upper_ci": model.conf_int()[1],
        "pval": model.pvalues,
    }).drop("Intercept")

    plt.figure(figsize=(8, 5))
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
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Estimated Coefficient")
    plt.ylabel("Predictor Variable")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

# In the REPL, test our reusable function:
plot_coefficients(full_model, "Predictors of Affective Polarization")


# ==============================================================================
# Grouped Distributions: Kernel Density Estimation (KDE)
# ==============================================================================
# Histograms can look blocky and depend heavily on arbitrary bin choices.
# Kernel Density Estimation (KDE) smooths observations to show underlying shape.
#
# Comparing affective polarization distributions by political party:

plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=df,
    x="affective_polarization",
    hue="party",
    fill=True,
    common_norm=False,
    palette={"Democrat": "blue", "Republican": "red", "Independent": "green"},
    alpha=0.25,
    linewidth=2,
)
plt.title(
    "Distribution of Affective Polarization by Party Identification",
    fontsize=12, fontweight="bold"
)
plt.xlabel("Affective Polarization (|Dem Thermometer - Rep Thermometer|)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.show()


# ==============================================================================
# Declarative Visualization with Altair
# ==============================================================================
# Altair is based on the "Grammar of Graphics" (Vega-Lite specification).
# Instead of writing imperative drawing instructions (draw line, set color),
# you declare the relationship between data columns and visual channels.
#
# Core data types in Altair:
#   :Q -> Quantitative (continuous numerical measurements, e.g. age, score)
#   :N -> Nominal (unordered discrete categories, e.g. party, state)
#   :O -> Ordinal (ranked discrete categories, e.g. education, Likert scale)
#   :T -> Temporal (dates and timestamps)

# Sample 500 respondents for smooth interactive rendering:
sample_df = df.sample(500, random_state=42)

# Declarative scatter chart with tooltips:
scatter_chart = (
    alt.Chart(sample_df, title="Age vs. Polarization by Party")
    .mark_circle(size=60, opacity=0.7)
    .encode(
        x=alt.X("age:Q", title="Respondent Age (years)"),
        y=alt.Y("affective_polarization:Q", title="Affective Polarization"),
        color=alt.Color(
            "party:N",
            scale=alt.Scale(
                domain=["Democrat", "Republican", "Independent"],
                range=["#2b5c8f", "#d95f02", "#7570b3"],
            ),
            title="Party Identification",
        ),
        tooltip=[
            alt.Tooltip("age:Q", title="Age"),
            alt.Tooltip("affective_polarization:Q", title="Polarization"),
            alt.Tooltip("party:N", title="Party"),
            alt.Tooltip("interest_label:N", title="Interest"),
        ],
    )
    .properties(width=500, height=350)
)

# In the REPL or interactive window, evaluate the chart:
scatter_chart


# ==============================================================================
# Interactive Dashboards: Linked Brushing in Altair
# ==============================================================================
# Altair makes it straightforward to link multiple views through selections.
# An interval selection ("brush") allows users to highlight a subset of points
# in one plot and watch other plots update in real time:

# Step 1: Create an interval selection over the scatter plot:
brush = alt.selection_interval()

# Step 2: Scatter plot that highlights points within the brush:
interactive_scatter = (
    alt.Chart(sample_df)
    .mark_circle(size=50)
    .encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("affective_polarization:Q", title="Affective Polarization"),
        color=alt.condition(
            brush,
            alt.Color("party:N", legend=None),
            alt.value("lightgray"),
        ),
    )
    .properties(width=350, height=300)
    .add_params(brush)
)

# Step 3: Bar chart showing counts of brushed respondents by interest:
brushed_bars = (
    alt.Chart(sample_df)
    .mark_bar()
    .encode(
        y=alt.Y("interest_label:N", title="Political Interest"),
        x=alt.X("count():Q", title="Number of Respondents"),
        color="party:N",
    )
    .transform_filter(brush)
    .properties(width=250, height=300)
)

# Combine the two charts side by side using the '|' operator:
dashboard = interactive_scatter | brushed_bars
dashboard


# ------------------------------------------------------------------------------
# Intentional Error Challenge: The Type Encoding Suffix
# ------------------------------------------------------------------------------
# In Altair, specifying the wrong data type alters how the chart is rendered.
# For example, encoding 'party' as Quantitative (:Q) will raise a Schema error
# because text strings cannot be mapped to a continuous numerical scale.
#
# 1. Uncomment the chart below and evaluate it in the REPL.
# 2. Inspect the error message: what schema validation failed?
# 3. Fix the type suffix to ':N' and re-run.
# broken_chart = (
#     alt.Chart(sample_df).mark_bar().encode(x="party:Q", y="count()")
# )


# ------------------------------------------------------------------------------
# Hack-Time: Enhancing the Coefficient Plot Function
# ------------------------------------------------------------------------------
# Tasks:
#   1. Modify plot_coefficients() to accept an optional dictionary of custom
#      variable labels (e.g. {"political_knowledge": "Knowledge Score (0-4)"}).
#   2. Add a visual distinction: plot coefficients that are statistically
#      significant (p < 0.05) in a different color from non-significant ones.
#   3. Test your enhanced function on full_model.


# ------------------------------------------------------------------------------
# Hack-Time: Building an Interactive Exploration of Ideology
# ------------------------------------------------------------------------------
# Tasks:
#   1. Using sample_df, build an Altair box plot or strip plot comparing
#      'feeling_democrat' across levels of 'ideology:O'.
#   2. Add an interactive dropdown binding allowing the user to filter
#      the chart by 'party:N'.
#   3. What pattern do you observe in how partisans evaluate Democrats as
#      their ideological self-placement becomes more conservative?


