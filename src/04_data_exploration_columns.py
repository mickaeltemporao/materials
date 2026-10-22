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

...


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

...


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

...


# ==============================================================================
# First Visualizations with Altair
# ==============================================================================
# In empirical research, looking at tables of numbers can be dry and slow.
# Visualizing your variables makes patterns and distributions immediately clear.
#
# In Python, we use Altair for data visualization. Altair follows a simple
# three-step logic:
#   1. Chart: The data table (a pandas DataFrame).
#   2. Mark: The visual shape to draw (e.g. .mark_bar()).
#   3. Encode: Which column goes to the x-axis and which goes to the y-axis.
#
# Core workflow:
#   First, prepare the summary table using pandas.
#   Second, inspect the table with .head() to verify column names.
#   Third, create the plot in two lines of Altair code!

# Example 1: Distribution of respondent age
# Step 1: Use pandas to create a frequency table:
df_age = my_df[my_df["age"] >= 18]["age"].value_counts().reset_index()

# Always inspect the DataFrame before plotting:
df_age.head()

# Step 2: Plot the frequencies:
alt.Chart(df_age).mark_bar().encode(x="age", y="count")


# Example 2: Voter turnout intentions
# Step 1: Filter and compute frequencies with pandas:
df_vote = (
    my_df[my_df["vote"].isin([1, 2])]["vote"]
    .value_counts()
    .reset_index()
)

# Inspect the table:
df_vote.head()

# Step 2: Plot turnout frequencies:
alt.Chart(df_vote).mark_bar().encode(x="vote", y="count")


# ------------------------------------------------------------------------------
# Hack-Time: Visualizing Ideology Frequencies
# ------------------------------------------------------------------------------
# In ANES 2020, ideology is measured on a 1-7 scale:
#   1 = Extremely Liberal, 4 = Moderate, 7 = Extremely Conservative
#   (Negative values represent non-responses)
#
# Tasks:
#   1. Filter my_df for valid ideology scores between 1 and 7.
#   2. Compute frequencies with .value_counts().reset_index() as df_ideo.
#   3. Inspect df_ideo.head() in the REPL.
#   4. Use alt.Chart(df_ideo).mark_bar().encode(...) to visualize it.


# ==============================================================================
# Labeling Categories Before Plotting
# ==============================================================================
# Unlabeled numeric codes (like 1, 2, 3) make charts difficult to interpret.
# By replacing numeric codes with candidate names in pandas first, Altair
# automatically uses those clear labels on the chart axes:

mask_cand = my_df["vote_int"].between(1, 4)
candidate_names = {
    1: "Joe Biden",
    2: "Donald Trump",
    3: "Jo Jorgensen",
    4: "Howie Hawkins",
}

df_cand = (
    my_df.loc[mask_cand, "vote_int"]
    .replace(candidate_names)
    .value_counts()
    .reset_index()
)
df_cand.columns = ["candidate", "votes"]

# Inspect the prepared table:
df_cand.head()

# Plot horizontal bars by putting categories on y and counts on x:
alt.Chart(df_cand).mark_bar().encode(x="votes", y="candidate")
