"""
==============================================================================
04: Data Exploration - Columns and Summary Statistics
==============================================================================

Learning Objectives:
  - Understand the political data science pipeline
  - Distinguish between pandas Series and DataFrames
  - Select single and multiple columns from survey datasets
  - Rename columns to create transparent, self-documenting data
  - Calculate univariate summary statistics: mean, standard deviation, describe
  - Compute frequency and percentage distributions using value_counts()
  - Visualize continuous and discrete distributions using the pandas plot API
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
import matplotlib.pyplot as plt

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
# Data Visualization with pandas
# ==============================================================================
# Plotting distributions is critical for detecting outliers and data errors.
# In social science research, choice of plot depends on measurement level:
#
#   1. Continuous variables (infinite possible values, e.g. age, income):
#      Use histograms to visualize the shape of the distribution.
#
#   2. Discrete / Categorical variables (finite categories, e.g. vote choice):
#      Use bar plots to compare category frequencies.

# Continuous variable: Histogram of respondent age
my_df["age"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Distribution of Age in ANES 2020")
plt.xlabel("Age")
plt.ylabel("Number of Respondents")
plt.tight_layout()
plt.show()

# Discrete variable: Bar plot of voter turnout intentions
vote_pct.plot(kind="bar", color="steelblue")
plt.title("Voter Turnout Intention (ANES 2020)")
plt.xlabel("Intends to Vote (1 = Yes, 2 = No)")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# Hack-Time: Visualizing Ideological Distribution
# ------------------------------------------------------------------------------
# In ANES 2020, ideology is measured on a 1-7 scale:
#   1 = Extremely Liberal, 4 = Moderate, 7 = Extremely Conservative
#   (Negative values like -9 and -8 represent non-responses and refusal)
#
# Tasks:
#   1. Create a boolean mask keeping only valid ideology scores between 1 and 7:
#      valid_ideology = my_df["ideology"].between(1, 7)
#   2. Plot a histogram or bar plot of valid ideology scores.
#   3. Are American voters polarized or concentrated in the middle?




# ==============================================================================
# Polished Visualizations & Figure Anatomy
# ==============================================================================
# Before presenting graphics in presentations or papers, recode numeric codes
# into human-readable labels and add clear titles and axes:

mask = my_df["vote_int"].between(1, 4)
candidate_labels = {
    1: "Joe Biden",
    2: "Donald Trump",
    3: "Jo Jorgensen",
    4: "Howie Hawkins",
}

candidate_shares = (
    my_df.loc[mask, "vote_int"]
    .replace(candidate_labels)
    .value_counts(normalize=True) * 100
)

candidate_shares.plot(
    kind="bar",
    title="2020 Presidential Vote Intentions (ANES)",
    ylabel="Percentage of Valid Responses (%)",
    color=["blue", "red", "gold", "green"],
)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
