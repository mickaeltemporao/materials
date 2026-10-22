"""
==============================================================================
06: Data Management, Variable Transformation, and Scale Construction
==============================================================================

Learning Objectives:
  - Recode survey variables using boolean masks and .replace() dictionaries
  - Understand the methodological tradeoff between row filtering and recoding
  - Create, modify, and drop DataFrame columns without modifying raw inputs
  - Discretize continuous variables into demographic bins using pd.cut()
  - Understand the theoretical rationale for composite scales in social science
  - Construct an Additive Political Trust Index and reverse-code survey items
  - Build an Objective Political Knowledge Index from institutional questions
  - Construct an Affective Polarization Index from feeling thermometers
  - Validate scale reliability, distributions, and item correlations
==============================================================================
"""

# ==============================================================================
# The Logic of Data Management
# ==============================================================================
# Survey research rarely gives you variables ready for immediate analysis.
# Raw files use arbitrary numeric codes, reverse scales, and multiple items
# tapping the same underlying concept.
#
# Data management encompasses three core tasks:
#   1. Recoding: Translating survey codes into meaningful text or binary tags.
#   2. Discretization: Binning continuous data into theoretical cohorts.
#   3. Scale Construction: Combining multiple items into composite indices.

import numpy as np
import pandas as pd
import altair as alt

# Load the American National Election Studies (ANES 2020) dataset:
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Select key survey variables:
var_dict = {
    "V201033": "vote_int",
    "V201507x": "age",
    "V201200": "ideology",
    "V201144x": "covid",
    "V201233": "trust_gov",
    "V201236": "trust_corrupt",
    "V201237": "trust_people",
    "V201156": "feeling_democrat",
    "V201157": "feeling_republican",
    "V201644": "knowledge_senate_term",
    "V201645": "knowledge_least_spending",
    "V201646": "knowledge_house_majority",
    "V201647": "knowledge_senate_majority",
}

df = anes_data[list(var_dict.keys())].copy()
df.columns = list(var_dict.values())

# Run the code to inspect the raw dataset:
df.shape
df.head()


# ==============================================================================
# Recoding Variables: Boolean Masks vs. .replace()
# ==============================================================================
# In ANES 2020, 'vote_int' records presidential vote intention:
#   1 = Joe Biden, 2 = Donald Trump, 3 = Jo Jorgensen, 4 = Howie Hawkins
#   Negative values (-9, -8, -1) represent non-responses and refusals.
#
# Concept Diagram: Creating New Columns in pandas
# https://pandas.pydata.org/docs/_images/05_newcolumn_1.svg
#
#   Existing DataFrame                      Transform & Assign
#   +----------+-------+                    df["candidate"] = ...
#   +----------+-------+                    +----------+-------+-------------+
#   | vote_int | age   |  ───────────────>  | vote_int | age   | candidate   |
#   +----------+-------+                    +----------+-------+-------------+
# 0 | 1        | 45    |                  0 | 1        | 45    | Joe Biden   |
# 1 | 2        | 62    |                  1 | 2        | 62    | Donald Trump|
#   +----------+-------+                    +----------+-------+-------------+

# Strategy 1: Recoding using a boolean mask and .loc
# To assign strings into a numeric column, create a new object column:
df["candidate"] = pd.Series(np.nan, index=df.index, dtype=object)
df.loc[df["vote_int"] == 1, "candidate"] = "Joe Biden"
df.loc[df["vote_int"] == 2, "candidate"] = "Donald Trump"
df.loc[df["vote_int"].isin([3, 4, 5]), "candidate"] = "Third Party / Other"

# Run the code to inspect the frequency of recoded candidate preferences:
df["candidate"].value_counts(dropna=False)

# Strategy 2: Recoding using .replace() and a dictionary
# The 'covid' variable asks: "Do you approve or disapprove of how the federal
# government handled the COVID-19 pandemic?"
#   1 = Strongly approve, 2 = Somewhat approve
#   3 = Somewhat disapprove, 4 = Strongly disapprove
#   Negative values (-2, -9) represent 'Don't know' and refusal.

covid_labels = {
    1: "Strongly Approve",
    2: "Somewhat Approve",
    3: "Somewhat Disapprove",
    4: "Strongly Disapprove",
}

# Create a clean, labeled covid evaluation column:
df["covid_label"] = df["covid"].replace(covid_labels)
df["covid_label"].value_counts()


# ------------------------------------------------------------------------------
# Intentional Error Challenge: Forgetting Assignment with .replace()
# ------------------------------------------------------------------------------
# 1. Uncomment the lines below and run them with Shift+Enter.
# 2. Inspect the output: why did the underlying column remain unchanged?
# 3. Fix the code by assigning the returned Series or creating a column.
# df["covid"].replace(covid_labels)
# df["covid"].head()


# ==============================================================================
# Methodological Tradeoff: Filtering vs. Recoding
# ==============================================================================
# When cleaning survey data, you face a critical methodological decision:
# Do you DROP non-matching respondents (filtering), or do you RECODE them
# into an explicit residual category ("Other", "Non-voter")?
#
# Tradeoff Analysis:
#   - Filtering drops rows, reducing sample size and statistical power.
#     If excluded respondents differ systematically from included ones,
#     filtering introduces selection bias!
#   - Recoding retains all valid respondents and preserves population weights,
#     giving a more realistic estimate of the entire electorate.

# Compare two-party filtered share vs. full electorate recoded share:
# Option A: Filtered (dropping all third-party and undecided voters):
two_party_share = (
    df.loc[df["vote_int"].isin([1, 2]), "candidate"]
    .value_counts(normalize=True) * 100
)

# Option B: Recoded (retaining third-party and other selections):
electorate_share = (
    df.loc[df["vote_int"] > 0, "candidate"]
    .value_counts(normalize=True) * 100
)

# Run the code to compare both estimates:
two_party_share.round(1)
electorate_share.round(1)


# ------------------------------------------------------------------------------
# Hack-Time: Creating Binary Dummy Variables
# ------------------------------------------------------------------------------
# In empirical modeling, dummy (indicator) variables take the value 1 when a
# condition is met, and 0 otherwise.
#
# Tasks:
#   1. Create a binary dummy variable named 'vote_trump' in df that takes
#      the value 1 if candidate == "Donald Trump", and 0 otherwise.
#   2. Run the code to compute the mean of 'vote_trump'.
#   3. What does this mean represent substantively?

...


# ==============================================================================
# Discretizing Continuous Variables with pd.cut()
# ==============================================================================
# Social scientists often group continuous variables (such as age or income)
# into demographic cohorts to analyze life-cycle and generational patterns.
#
# While manual boolean masking is tedious and error-prone, pandas provides
# pd.cut() to segment continuous data into bins:
#   - bins: List of edge values defining category boundaries.
#   - labels: Descriptive names for the resulting intervals.

# Clean respondent age by removing non-substantive negative codes:
df["age_clean"] = df["age"].where(df["age"] >= 18, np.nan)

# Create four generational cohorts:
#   (17, 35] -> 18 to 35 (Young voters)
#   (35, 50] -> 36 to 50 (Early-to-mid career)
#   (50, 65] -> 51 to 65 (Late career)
#   (65, 80] -> 66+ (Retirement age)
age_bins = [17, 35, 50, 65, 80]
age_cohorts = ["18-35", "36-50", "51-65", "66+"]

df["age_cohort"] = pd.cut(df["age_clean"], bins=age_bins, labels=age_cohorts)

# Run the code to inspect the distribution of age cohorts:
df["age_cohort"].value_counts().sort_index()


# ==============================================================================
# Customizing Visualizations: Properties, Titles, and Colors
# ==============================================================================
# In Lesson 05, we focused on raw marks and data types. Now that our charts
# are assigned to Python objects, we can begin customizing their presentation:
#   1. Dimensions & Titles: .properties(title="...", width=..., height=...)
#      controls the canvas dimensions and main heading.
#   2. Axis Labels: Passing title="..." inside alt.X() and alt.Y() replaces
#      raw variable names with readable survey question labels.
#   3. Colors: Setting color="..." directly inside the mark method applies a
#      custom aesthetic hue (e.g. "darkcyan", "steelblue").

# Visualize age cohort distribution using Altair:
cohort_chart = (
    alt.Chart(df.dropna(subset=["age_cohort"]))
    .mark_bar(color="darkcyan")
    .encode(
        x=alt.X(
            "age_cohort",
            type="ordinal",
            title="Age Cohort",
            sort=age_cohorts,
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Age Cohorts (ANES 2020)",
        width=450,
        height=280,
    )
)
cohort_chart

# Cross-tabulate candidate preference by age cohort:
cohort_vote = (
    pd.crosstab(
        df["age_cohort"],
        df["candidate"],
        normalize="index",
    )
    * 100
)
cohort_vote.round(1)


# ------------------------------------------------------------------------------
# Hack-Time: Discretizing COVID-19 Evaluations
# ------------------------------------------------------------------------------
# Substantive question: Did younger and older Americans evaluate the federal
# pandemic response differently?
#
# Tasks:
#   1. Create a binary variable named 'covid_approve' that equals 1 if
#      the respondent approved of government handling (covid in [1, 2]),
#      and 0 if they disapproved (covid in [3, 4]).
#   2. Compute a cross-tabulation of 'covid_approve' by 'age_cohort'
#      normalized across rows (normalize="index").
#   3. Which age cohort expressed the highest approval of federal handling?

...


# ==============================================================================
# The Theory of Composite Scale Construction
# ==============================================================================
# Why do political scientists construct composite scales instead of relying
# on single survey questions?
#
# 1. Random Measurement Error: Every individual survey question contains noise,
#    misunderstandings, and transient respondent feelings. Summing across
#    multiple related items allows random errors to cancel out.
#
# 2. Multidimensional Concepts: Abstract constructs like "political trust",
#    "political knowledge", or "democratic norms" cannot be captured by a
#    single survey item.
#
# 3. Construct Validity & Reliability: Multi-item scales yield continuous,
#    normally distributed quantities with higher predictive power in regression.


# ==============================================================================
# Scale 1: Additive Political Trust Index
# ==============================================================================
# The ANES includes three classic items tapping trust in political institutions:
#   1. trust_gov: "How much of the time can you trust the government in
#      Washington to do what is right?"
#      (1=Always, 2=Most, 3=Half, 4=Some, 5=Never)
#   2. trust_corrupt: "How many government officials are corrupt?"
#      (1=Quite a few, 2=Not many, 3=Hardly any)
#   3. trust_people: "Would you say that most of the time people try to be
#      helpful, or are they mostly looking out for themselves?"
#      (1=Helpful / Trusted, 2=Can't be too careful)

trust_items = ["trust_gov", "trust_corrupt", "trust_people"]

# Step 1: Replace non-substantive negative codes with NaN:
df[trust_items] = df[trust_items].replace([-9, -8, -1], np.nan)

# Step 2: Reverse-code variables so higher numbers always mean MORE trust!
# In trust_gov, 1 = Always trust, 5 = Never trust.
# To flip this direction: new_value = (max + min) - old_value = 6 - old_value.
df["trust_gov_rev"] = 6 - df["trust_gov"]

# In trust_corrupt: 1 = Quite a few corrupt, 3 = Hardly any corrupt.
# Notice: higher values already mean LESS corruption (i.e. MORE trust)!
# No reversal is needed for trust_corrupt.
df["trust_corrupt_rev"] = df["trust_corrupt"]

# In trust_people: 1 = Helpful / Trusted, 2 = Looking out for themselves.
# Flip so 2 = Helpful (trusted), 1 = Looking out:
df["trust_people_rev"] = 3 - df["trust_people"]

# Step 3: Construct the additive scale by summing across rows:
# Using min_count=3 ensures that respondents with missing items get NaN:
rev_trust_vars = ["trust_gov_rev", "trust_corrupt_rev", "trust_people_rev"]
df["trust_index"] = df[rev_trust_vars].sum(axis=1, min_count=3)

# Run the code to inspect the trust scale distribution:
df["trust_index"].describe()

trust_chart = (
    alt.Chart(df.dropna(subset=["trust_index"]))
    .mark_bar(color="steelblue")
    .encode(
        x=alt.X(
            "trust_index",
            type="quantitative",
            bin=alt.Bin(step=1),
            title="Trust Index Score (3=Lowest, 10=Highest)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Political Trust Index",
        width=450,
        height=280,
    )
)
trust_chart


# ==============================================================================
# Scale 2: Objective Political Knowledge Index
# ==============================================================================
# The ANES assesses political knowledge through factual quiz questions:
#   1. Senate term length (correct answer: 6 years)
#   2. Least federal spending area (correct answer: Foreign aid, coded 1)
#   3. House majority party (correct answer: Democrats, coded 1)
#   4. Senate majority party (correct answer: Republicans, coded 2)

# Score each question: 1 if correct, 0 if incorrect or refused:
df["k_senate_term"] = (df["knowledge_senate_term"] == 6).astype(int)
df["k_least_spending"] = (df["knowledge_least_spending"] == 1).astype(int)
df["k_house_majority"] = (df["knowledge_house_majority"] == 1).astype(int)
df["k_senate_majority"] = (df["knowledge_senate_majority"] == 2).astype(int)

knowledge_items = [
    "k_senate_term",
    "k_least_spending",
    "k_house_majority",
    "k_senate_majority",
]

# Sum factual scores into a 0-4 composite scale:
df["political_knowledge"] = df[knowledge_items].sum(axis=1)

# Run the code to inspect knowledge distribution:
df["political_knowledge"].value_counts().sort_index()

knowledge_chart = (
    alt.Chart(df)
    .mark_bar(color="darkslateblue")
    .encode(
        x=alt.X(
            "political_knowledge",
            type="ordinal",
            title="Number of Correct Answers (0 to 4)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Objective Political Knowledge Scale (0 to 4)",
        width=450,
        height=280,
    )
)
knowledge_chart


# ==============================================================================
# Scale 3: Affective Polarization Index
# ==============================================================================
# Affective polarization measures the emotional hostility partisans feel
# toward the opposing political party relative to their own.
#
# Feeling thermometers ask respondents to rate groups on a 0 to 100 scale:
#   0 = Cold / Unfavorable, 50 = Neutral, 100 = Warm / Favorable.
#
# The standard measure of affective polarization is the absolute difference
# between feelings toward Democrats and Republicans:
#   affective_polarization = | feeling_democrat - feeling_republican |

# Keep valid feeling thermometer responses (0 to 100):
valid_ft = (
    df["feeling_democrat"].between(0, 100)
    & df["feeling_republican"].between(0, 100)
)

df["affective_polarization"] = np.nan
df.loc[valid_ft, "affective_polarization"] = (
    df.loc[valid_ft, "feeling_democrat"]
    - df.loc[valid_ft, "feeling_republican"]
).abs()

# Run the code to inspect affective polarization:
df["affective_polarization"].describe()

polarization_chart = (
    alt.Chart(df.dropna(subset=["affective_polarization"]))
    .mark_bar(color="crimson")
    .encode(
        x=alt.X(
            "affective_polarization",
            type="quantitative",
            bin=alt.Bin(maxbins=25),
            title="Absolute Partisan Thermometer Difference (|Dem - Rep|)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Affective Polarization (0 to 100)",
        width=450,
        height=280,
    )
)
polarization_chart


# ==============================================================================
# Scale Validation: Internal Consistency & Correlation
# ==============================================================================
# Before using composite scales in regression models, social scientists
# validate their reliability:
#
# 1. Item Correlation Matrix: Do the items measuring the same concept
#    correlate positively with each other?
# 2. Predictive Validity: Does the scale correlate with related behaviors?

# Check correlation matrix across political trust items:
df[rev_trust_vars].corr().round(2)

# Check correlation matrix across political knowledge items:
df[knowledge_items].corr().round(2)

# Hypothesis test: Does political knowledge correlate with polarization?
# In political science literature, highly knowledgeable citizens tend to be
# more emotionally polarized because they hold stronger partisan attachments:
knowledge_polarization_corr = df["political_knowledge"].corr(
    df["affective_polarization"]
)
knowledge_polarization_corr


# ------------------------------------------------------------------------------
# Hack-Time: Validating Scales Across Political Subgroups
# ------------------------------------------------------------------------------
# Tasks:
#   1. Calculate the mean of 'political_knowledge' across each 'age_cohort'
#      using df.groupby("age_cohort")["political_knowledge"].mean().
#   2. Calculate the mean of 'affective_polarization' across each 'age_cohort'.
#   3. Filter df for strong partisans vs moderates using 'ideology'
#      (e.g. ideology in [1, 7] vs ideology == 4). Compare their average
#      affective polarization scores.

...
