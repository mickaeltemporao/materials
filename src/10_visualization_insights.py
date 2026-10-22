"""
==============================================================================
10: Data Visualization, Declarative Graphics, and Communication
==============================================================================

Learning Objectives:
  - Translate regression results into intuitive, communicative visual narratives
  - Construct reusable functions to generate dot-and-whisker coefficient plots
  - Visualize segmented distributions using grouped kernel density estimation
  - Master declarative graphics in Python using Altair's grammar of graphics
  - Map variables to visual channels using Altair long-form data types
  - Construct faceted small-multiple charts to unpack subgroup differences
  - Apply publication-ready typography, labels, custom palettes, and themes
  - Export high-resolution figures for integration into Typst documents
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
#   3. Publication-ready declarative graphics with Altair: faceting, polishing,
#      and Typst export.

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

# Run the code to test our reusable function:
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
# Core data types in Altair (strictly specified using long-form syntax):
#   - type="quantitative": Continuous numerical values (e.g. age, score)
#   - type="nominal": Unordered discrete categories (e.g. party, state)
#   - type="ordinal": Ordered discrete categories (e.g. education, Likert)
#   - type="temporal": Dates and timestamps
#
# CRITICAL: Always use explicit method-based syntax and long-form type
# arguments. Never use colon shorthand notation (e.g. :Q, :N, :O).

# Sample 500 respondents for smooth interactive rendering:
sample_df = df.sample(500, random_state=42)

# Declarative scatter chart with tooltips:
scatter_chart = (
    alt.Chart(sample_df, title="Age vs. Polarization by Party")
    .mark_circle(size=60, opacity=0.7)
    .encode(
        x=alt.X(
            "age",
            type="quantitative",
            title="Respondent Age (years)",
        ),
        y=alt.Y(
            "affective_polarization",
            type="quantitative",
            title="Affective Polarization",
        ),
        color=alt.Color(
            "party",
            type="nominal",
            scale=alt.Scale(
                domain=["Democrat", "Republican", "Independent"],
                range=["#2b5c8f", "#d95f02", "#7570b3"],
            ),
            title="Party Identification",
        ),
        tooltip=[
            alt.Tooltip("age", type="quantitative", title="Age"),
            alt.Tooltip(
                "affective_polarization",
                type="quantitative",
                title="Polarization",
            ),
            alt.Tooltip("party", type="nominal", title="Party"),
            alt.Tooltip(
                "interest_label",
                type="nominal",
                title="Interest",
            ),
        ],
    )
    .properties(width=500, height=350)
)

# Run the code to display the chart:
scatter_chart


# ==============================================================================
# Faceted Visualizations: Edward Tufte's Small Multiples
# ==============================================================================
# In political science, we often want to compare how an empirical relationship
# behaves across distinct subgroups (e.g. across parties or cohorts).
#
# Overlaying multiple groups onto a single chart frequently produces visual
# clutter. The small multiples technique (Tufte 1983) solves this by splitting
# data into a grid of identical, aligned sub-charts using the same scale.
#
# In Altair, faceting is implemented via column=alt.Column(...) or .facet():

faceted_polarization = (
    alt.Chart(sample_df)
    .mark_circle(size=50, opacity=0.6)
    .encode(
        x=alt.X(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
        y=alt.Y(
            "affective_polarization",
            type="quantitative",
            title="Affective Polarization",
        ),
        color=alt.Color(
            "party",
            type="nominal",
            scale=alt.Scale(
                domain=["Democrat", "Republican", "Independent"],
                range=["#2b5c8f", "#d95f02", "#7570b3"],
            ),
            legend=None,
        ),
        column=alt.Column(
            "party",
            type="nominal",
            title="Party Identification",
        ),
    )
    .properties(width=220, height=220)
)

faceted_polarization


# ==============================================================================
# Publication-Ready Finishes: Typography, Reference Lines, and Themes
# ==============================================================================
# Exploratory graphics are quick and rough; publication figures require
# meticulous attention to visual hierarchy, typography, and clean framing:
#
# 1. Title and Subtitle: Use alt.TitleParams() to formulate a clear finding
#    in the title and cite data sources in the subtitle.
# 2. Reference Lines: Use alt.Chart().mark_rule() to anchor substantive
#    benchmarks (e.g. the sample mean or a neutral score).
# 3. Canvas Configuration: Use .configure_*() methods to remove default chart
#    borders, adjust font sizes, and mute distracting gridlines.

# Calculate the sample mean of affective polarization:
mean_polarization = df["affective_polarization"].mean()

# Reference line indicating overall sample mean:
polarization_mean_rule = (
    alt.Chart(pd.DataFrame({"mean_val": [mean_polarization]}))
    .mark_rule(color="#555555", strokeDash=[4, 4], strokeWidth=1.5)
    .encode(y=alt.Y("mean_val", type="quantitative"))
)

# Polished composite visualization:
polished_figure = (
    (scatter_chart + polarization_mean_rule)
    .properties(
        title=alt.TitleParams(
            text="Affective Polarization Across Respondent Age",
            subtitle=[
                f"Dashed line = sample mean ({mean_polarization:.1f})",
                "Source: American National Election Studies (ANES 2020)",
            ],
            anchor="start",
            fontSize=15,
            subtitleFontSize=11,
            subtitleColor="#666666",
        ),
        width=520,
        height=320,
    )
    .configure_axis(
        labelFontSize=11,
        titleFontSize=12,
        gridColor="#f0f0f0",
    )
    .configure_legend(
        titleFontSize=11,
        labelFontSize=10,
        orient="bottom",
    )
    .configure_view(stroke=None)
)

polished_figure


# ==============================================================================
# Exporting Figures for Typst Papers and Presentations
# ==============================================================================
# When writing articles, theses, or policy memos in Typst or LaTeX, you need
# to export your figures to standalone image files.
#
# Altair supports vector (SVG/PDF) and raster (PNG) exports via .save():
#
#   - PNG (raster): High compatibility, set scale_factor=2.0 for 300 DPI print.
#   - SVG (vector): Infinite resolution, ideal for publication layouts.
#
# To save to disk, uncomment the lines below:
# polished_figure.save("figures/polarization_by_party.png", scale_factor=2.0)
# polished_figure.save("figures/polarization_by_party.svg")
#
# Embedding your exported figure in a Typst document:
# ------------------------------------------------------------------------------
# In your Typst paper (`paper.typ`), insert the figure using:
#
#   #figure(
#     image("figures/polarization_by_party.png", width: 85%),
#     caption: [Affective Polarization by Age and Party Identification.],
#   ) <fig-polarization>
#
# You can then reference it in your text using:
#   "As shown in @fig-polarization, affective hostility is elevated..."


# ------------------------------------------------------------------------------
# Intentional Error Challenge: Incompatible Altair Data Type
# ------------------------------------------------------------------------------
# In Altair, specifying an incompatible data type raises a SchemaValidationError
# because text strings cannot be mapped to a continuous numerical scale.
#
# 1. Uncomment the chart below and run it with Shift+Enter.
# 2. Inspect the error message: what schema validation failed?
# 3. Fix the data type to type="nominal" and re-run.
# broken_chart = (
#     alt.Chart(sample_df)
#     .mark_bar()
#     .encode(
#         x=alt.X("party", type="quantitative"),
#         y=alt.Y("count()", type="quantitative"),
#     )
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

...


# ------------------------------------------------------------------------------
# Hack-Time: Building a Publication-Ready Faceted Visualization
# ------------------------------------------------------------------------------
# Tasks:
#   1. Using sample_df, build a faceted Altair chart comparing feelings
#      toward Democrats ('feeling_democrat', type="quantitative") across
#      levels of 'ideology' (type="ordinal"), faceted by 'party'
#      (type="nominal").
#   2. Apply publication-ready typography using alt.TitleParams() with a
#      substantive finding title, subtitle, and data source.
#   3. Polish the layout using .configure_axis() and
#      .configure_view(stroke=None).
#   4. Add a commented-out .save() command exporting your figure to PNG.

...
