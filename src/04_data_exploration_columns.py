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
# Concept Diagram: Series vs DataFrame
# https://storage.googleapis.com/lds-media/images/
#   series-and-dataframe.width-1200.png
#
#       DataFrame ('my_df')                      Series ('age')
#       +---------+----------+-------+-------+     +-------+
#       | vote    | vote_int | age   | ideo  |     | age   |
#       +---------+----------+-------+-------+     +-------+
#     0 | 1       | 1        | 45    | 2     |   0 | 45    |
#     1 | 1       | 2        | 62    | 6     |   1 | 62    |
#     2 | 2       | -1       | 21    | 4     |   2 | 21    |
#       +---------+----------+-------+-------+     +-------+
#       Columns (Variables) ──^                    1D Array
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
# Concept Diagram: Selecting Columns
# https://pandas.pydata.org/docs/_images/03_subset_columns.svg
#
#   DataFrame (anes_data)         Select my_vars               my_df
#   +-----+-----+-----+-----+     ["vote", "age", ...]         +-----+-----+
#   | V1  | V2  | V3  | ... |  ─────────────────────────>      |vote | age |
#   +-----+-----+-----+-----+                                  +-----+-----+
#   | ... | ... | ... | ... |                                  | ... | ... |
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
# Answering Political Science Questions with Summary Statistics
# ==============================================================================
# In empirical political research, summary statistics answer foundational
# descriptive questions before jumping into complex causal models.
#
# pandas Series provide built-in methods to summarize variables:
#   .mean(), .std(), .min(), .max(), .count(), .describe(), .value_counts()
#
# Question: How many people in the sample intend to vote?
# In ANES 2020: 1 = Intend to vote, 2 = Do not intend to vote.
my_df["vote"].value_counts()

# Question: What percentage of respondents intend to vote?
vote_pct = my_df["vote"].value_counts(normalize=True) * 100
vote_pct.round(1)


# ------------------------------------------------------------------------------
# Hack-Time: What is the Average Age and Ideology of the Electorate?
# ------------------------------------------------------------------------------
# Tasks:
#   1. What is the average age of respondents in the ANES dataset?
#      Calculate my_df["age"].mean() in the REPL.
#   2. Use .describe() on my_df["age"] to view all summary statistics
#      (mean, standard deviation, quartiles, min, max).
#   3. What is the average ideology of respondents?
#      Use .describe() on my_df["ideology"] to inspect the scale.

...


# ==============================================================================
# First Visualizations with Altair
# ==============================================================================
# In empirical research, looking at summary numbers alone can deceive you.
# Datasets with identical means and standard deviations can have completely
# different real-world distributions!
#
# Concept: Same stats, different graphs (Datasaurus Dozen animation):
# https://damassets.autodesk.net/content/dam/autodesk/
#   research/publications-assets/gifs/
#   same-stats-different-graphs/DinoSequentialSmaller.gif
#
# In Python, we use Altair for statistical visualization. Altair follows a
# simple three-step declarative workflow:
#
#       Data Table        ──>       Mark        ──>       Encode
#    (pandas DataFrame)        (Geometric Shape)     (x and y Channels)
#      alt.Chart(df)      ──>    .mark_bar()    ──>  .encode(x=..., y=...)
#
# Core workflow:
#   First, prepare the summary table using pandas
#   (.value_counts().reset_index()).
#   Second, inspect the table with .head() to verify column names.
#   Third, create the plot in two lines of Altair code!

# ------------------------------------------------------------------------------
# Question: What is the Age Distribution of the Electorate?
# ------------------------------------------------------------------------------
# Step 1: Use pandas to create a frequency table across all observations:
df_age = my_df["age"].value_counts().reset_index()

# Always inspect the DataFrame before plotting:
df_age.head()

# Step 2: Plot the frequencies:
alt.Chart(df_age).mark_bar().encode(x="age", y="count")


# ------------------------------------------------------------------------------
# Question: How Many Citizens Intend to Vote?
# ------------------------------------------------------------------------------
# Step 1: Compute frequencies across all observations:
df_vote = my_df["vote"].value_counts().reset_index()

# Inspect the table:
df_vote.head()

# Step 2: Plot turnout frequencies:
alt.Chart(df_vote).mark_bar().encode(x="vote", y="count")


# ------------------------------------------------------------------------------
# Hack-Time: Are US Citizens Polarized?
# ------------------------------------------------------------------------------
# Political scientists debate whether the American public is polarized into
# ideological extremes (liberals vs. conservatives) or clustered in the middle.
#
# The ANES measures ideological self-placement on a 7-point scale:
#
#      1        2         3         4         5         6         7
#   [---|--------|---------|---------|---------|---------|---------|---]
#     Extr.     Lib.     Slight.  Moderate   Slight.   Cons.     Extr.
#     Lib.               Lib.                Cons.               Cons.
#   (Negative values represent survey non-responses)
#
# Tasks:
#   1. Compute frequencies across all observations of my_df["ideology"]
#      using .value_counts().reset_index() and save it as df_ideo.
#   2. Inspect df_ideo.head() in the REPL.
#   3. Visualize the distribution with:
#      alt.Chart(df_ideo).mark_bar().encode(x="ideology", y="count")
#   4. Look at the chart: do respondents cluster around 4 (moderate) or do
#      they concentrate at the extreme ends (1 and 7)?

...


# ==============================================================================
# Question: Who Would Win the Popular Vote According to the ANES?
# ==============================================================================
# In ANES 2020, 'vote_int' records which presidential candidate respondents
# intended to vote for:
#   1 = Joe Biden, 2 = Donald Trump, 3 = Jo Jorgensen, 4 = Howie Hawkins
#
# Unlabeled numeric codes make charts difficult to interpret. By replacing
# numeric codes with candidate names in pandas first, Altair automatically
# uses those clear names on the chart axes:

candidate_names = {
    1: "Joe Biden (Dem)",
    2: "Donald Trump (Rep)",
    3: "Jo Jorgensen (Lib)",
    4: "Howie Hawkins (Grn)",
}

df_cand = (
    my_df["vote_int"]
    .replace(candidate_names)
    .value_counts()
    .reset_index()
)
df_cand.columns = ["candidate", "votes"]

# Inspect the prepared table:
df_cand.head()

# Plot horizontal bars by putting candidate on y and vote counts on x:
alt.Chart(df_cand).mark_bar().encode(x="votes", y="candidate")

# Anatomy of a Figure Reference:
# https://matplotlib.org/3.1.1/_images/anatomy.png
