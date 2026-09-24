"""
==============================================================================
04: Data Exploration - Columns, Summary Statistics, and Altair
==============================================================================

Learning Objectives:
  - Understand the political data science pipeline
  - Distinguish between pandas Series and DataFrames
  - Select single and multiple columns from survey datasets
  - Rename columns to create transparent, self-documenting data
  - Calculate univariate summary statistics: mean, standard deviation, describe
  - Compute frequency and percentage distributions using value_counts()
  - Master declarative graphics using Altair's grammar of graphics
  - Visualize continuous and discrete distributions (histograms, bar charts)
==============================================================================
"""

# ==============================================================================
# The Political Data Science Pipeline
# ==============================================================================
# Empirical research typically progresses through four stages:
#   1. Acquisition: Loading raw survey data, electoral records, or text.
#   2. Exploration: Inspecting variables, missingness, distributions, shapes.
#   3. Modeling: Testing hypotheses using regression and statistical controls.
#   4. Communication: Presenting results in clear publication-ready figures.
#
# Before building statistical models, you must deeply understand your variables.

import pandas as pd
import altair as alt

# Load the American National Election Studies (ANES 2020) dataset:
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)


# ------------------------------------------------------------------------------
# Hack-Time: Inspecting Raw Survey Data
# ------------------------------------------------------------------------------
# Tasks:
#   1. Check the type of anes_data in the REPL.
#   2. Inspect the number of rows and columns using the .shape attribute.




# ==============================================================================
# DataFrames and Series
# ==============================================================================
# In pandas:
#   - A DataFrame is a 2-dimensional tabular dataset with rows and columns.
#   - A Series is a single 1-dimensional column from a DataFrame.
#
# Let us construct a small DataFrame from scratch to understand its anatomy:

turnout = [66.8, 55.7, 54.9, 58.2, 56.7, 51.2, 49.0]
year = [2020, 2016, 2012, 2008, 2004, 2000, 1996]
party = ["Dem", "Rep", "Dem", "Dem", "Rep", "Rep", "Dem"]

my_data = pd.DataFrame({
    "turnout": turnout,
    "year": year,
    "party": party,
})

# In the REPL, evaluate my_data directly to view the table:
my_data

# Use .info() to inspect data types and non-null counts:
my_data.info()

# Use .head() to view the first few rows of a large dataset:
my_data.head()


# ------------------------------------------------------------------------------
# Hack-Time: Exploring the Full ANES Dataset
# ------------------------------------------------------------------------------
# Tasks:
#   1. Evaluate anes_data.head() in the REPL to view the first 5 rows.
#   2. Check the column names of anes_data using anes_data.columns.




# ==============================================================================
# Selecting Variables (Columns)
# ==============================================================================
# In survey analysis, raw datasets contain hundreds of questions.
# You need to isolate only the variables relevant to your research hypothesis.
#
# 1. Selecting a single column returns a Series:
#    df["column_name"]

vote_intent = anes_data["V201033"]
type(vote_intent)
vote_intent.head()

# 2. Selecting multiple columns requires passing a list of column names:
#    df[["col1", "col2", "col3"]]

my_vars = [
    "V201032",   # Turnout: did respondent intend to vote?
    "V201033",   # Vote choice: candidate intended to vote for
    "V201507x",  # Age of respondent
    "V201200",   # Ideology: 7-point scale (1=Liberal to 7=Conservative)
]

# Create a clean subset DataFrame:
my_df = anes_data[my_vars].copy()
my_df.head()


# ==============================================================================
# Renaming Columns for Clarity
# ==============================================================================
# Raw survey codes (e.g. V201507x) make analysis error-prone.
# Always rename variables to intuitive, descriptive names:

my_df.columns = ["vote", "vote_int", "age", "ideology"]
my_df.head()


# ==============================================================================
# Univariate Summary Statistics
# ==============================================================================
# pandas Series provide built-in methods to summarize variables:
#   .mean(), .std(), .min(), .max(), .count(), .describe(), .value_counts()

# Frequencies of vote intention:
my_df["vote"].value_counts()

# Proportions (relative frequencies):
my_df["vote"].value_counts(normalize=True)

# Percentages rounded to nearest whole number:
vote_pct = my_df["vote"].value_counts(normalize=True) * 100
vote_pct.round()


# ------------------------------------------------------------------------------
# Hack-Time: Exploring Age and Ideology
# ------------------------------------------------------------------------------
# Tasks:
#   1. Calculate the mean age of respondents in my_df.
#   2. Use .describe() on my_df["age"] to view all summary statistics.
#   3. Use .describe() on my_df["ideology"] to inspect the ideology scale.




# ==============================================================================
# Declarative Data Visualization with Altair
# ==============================================================================
# In empirical political science, data visualization is essential for uncovering
# distributions, spotting anomalies, and communicating substantive patterns.
# Rather than imperative plotting, we use Altair—a declarative statistical
# visualization library based on the Grammar of Graphics (Vega-Lite).
#
# Declarative visualization means you declare the statistical relationship
# between variables and visual channels (x, y, color), and Altair handles the
# rendering.
#
# Altair's Grammar of Graphics follows a three-step structure:
#   1. Chart: The data table (a pandas DataFrame).
#   2. Mark: The geometric mark representing data (bar, circle, line, boxplot).
#   3. Encode: The visual channel mapping (x, y, color, size, shape).
#
# Core Altair Data Types (strictly specified using long-form syntax):
#   - type="quantitative": Continuous numerical values (e.g. age, turnout %)
#   - type="nominal": Unordered discrete categories (e.g. party, state)
#   - type="ordinal": Ordered discrete categories (e.g. education level)
#   - type="temporal": Dates and timestamps

# Continuous variable: Histogram of respondent age
# For histograms, bin=alt.Bin(maxbins=30) groups continuous values:
df_age = my_df[my_df["age"] >= 18].copy()

age_histogram = (
    alt.Chart(df_age)
    .mark_bar(color="steelblue")
    .encode(
        x=alt.X(
            "age",
            type="quantitative",
            bin=alt.Bin(maxbins=30),
            title="Respondent Age (years)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Respondent Age (ANES 2020)",
        width=500,
        height=300,
    )
)

# In the REPL, sending the chart name evaluates and renders the figure:
age_histogram

# Discrete variable: Bar chart of voter turnout intentions
# Step 1: Use pandas to compute a clean frequency table:
vote_valid = my_df[my_df["vote"].isin([1, 2])].copy()
vote_freq = (
    vote_valid["vote"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .reset_index()
)
vote_freq.columns = ["vote_code", "share"]
vote_freq["label"] = vote_freq["vote_code"].map({
    1: "Intends to Vote",
    2: "Does Not Intend",
})

# Step 2: Build declarative bar chart with explicit types:
turnout_bar = (
    alt.Chart(vote_freq)
    .mark_bar(color="#2b5c8f")
    .encode(
        x=alt.X(
            "label",
            type="nominal",
            title="Turnout Intention",
            sort=None,
        ),
        y=alt.Y(
            "share",
            type="quantitative",
            title="Percentage of Respondents (%)",
        ),
    )
    .properties(
        title="2020 Voter Turnout Intentions (ANES)",
        width=400,
        height=300,
    )
)

turnout_bar


# ------------------------------------------------------------------------------
# Hack-Time: Visualizing Ideological Distribution with Altair
# ------------------------------------------------------------------------------
# In ANES 2020, ideology is measured on a 1-7 scale:
#   1 = Extremely Liberal, 4 = Moderate, 7 = Extremely Conservative
#   (Negative values like -9 and -8 represent non-responses and refusal)
#
# Tasks:
#   1. Create a filtered DataFrame keeping only valid ideology scores (1 to 7):
#      valid_ideo = my_df[my_df["ideology"].between(1, 7)]
#   2. Build an Altair bar chart displaying the frequency of each score:
#      - Use alt.Chart(valid_ideo).mark_bar()
#      - Map x to alt.X("ideology", type="ordinal", title="Ideology (1-7)")
#      - Map y to alt.Y("count()", type="quantitative", title="Count")
#      - Add properties: title="Ideology Distribution", width=450, height=280
#   3. Evaluate the chart in the REPL. Are voters polarized or centrist?


# ==============================================================================
# Polished Visualizations: Figure Anatomy & Communication
# ==============================================================================
# In scientific research, every figure must communicate clearly:
#   - Title stating the main substantive finding
#   - Subtitle providing context or sample details
#   - Data source attribution
#   - Informative axis titles and readable categories
#
# Let us visualize 2020 presidential candidate vote intentions:

mask_cand = my_df["vote_int"].between(1, 4)
candidate_names = {
    1: "Joe Biden (Dem)",
    2: "Donald Trump (Rep)",
    3: "Jo Jorgensen (Lib)",
    4: "Howie Hawkins (Grn)",
}

candidate_df = (
    my_df.loc[mask_cand, "vote_int"]
    .map(candidate_names)
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .reset_index()
)
candidate_df.columns = ["candidate", "share"]

# Horizontal bar chart sorted by vote share with informative subtitles:
candidate_chart = (
    alt.Chart(candidate_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "share",
            type="quantitative",
            title="Percentage of Valid Responses (%)",
        ),
        y=alt.Y(
            "candidate",
            type="nominal",
            title="Candidate",
            sort="-x",
        ),
        color=alt.Color(
            "candidate",
            type="nominal",
            legend=None,
            scale=alt.Scale(
                domain=[
                    "Joe Biden (Dem)",
                    "Donald Trump (Rep)",
                    "Jo Jorgensen (Lib)",
                    "Howie Hawkins (Grn)",
                ],
                range=["#2b5c8f", "#d95f02", "#e6ab02", "#2ca02c"],
            ),
        ),
    )
    .properties(
        title=alt.TitleParams(
            text="2020 Presidential Vote Intentions",
            subtitle=[
                "Share of respondents intending to vote for each candidate",
                "Source: American National Election Studies (ANES 2020)",
            ],
            anchor="start",
        ),
        width=500,
        height=250,
    )
)

candidate_chart
