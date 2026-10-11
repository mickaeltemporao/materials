"""
==============================================================================
05: Data Exploration - Rows, Boolean Filtering, and Relationships
==============================================================================

Learning Objectives:
  - Subset observations (filter rows) using relational operators
  - Combine multiple filtering conditions using boolean logic (&, |, ~)
  - Filter missing survey codes using .between() and .isin()
  - Distinguish types of data and levels of measurement in political science
  - Visualize continuous and discrete bivariate relationships with Altair
  - Compute cross-tabulations and build Altair stacked bar charts
  - Apply row filtering to analyze youth political behavior in the ANES
==============================================================================
"""

# ==============================================================================
# Filtering Observations (Rows)
# ==============================================================================
# In empirical research, datasets are rarely clean out of the box.
# Survey files frequently contain irrelevant cases, non-responses, or refusals.
#
# To test hypotheses accurately, you need to filter rows (observations):
#   - Isolating a subgroup of theoretical interest (e.g. young voters).
#   - Removing non-response codes (e.g. -9 "Refused", -8 "Don't know").
#
# In pandas, row filtering is performed using boolean indexing (masks).
# A boolean mask is a Series of True and False values that tells pandas
# which rows to keep (True) and which to discard (False).

import pandas as pd
import altair as alt

# Load the American National Election Studies (ANES 2020) dataset:
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression="gzip", low_memory=False)

# Select key survey variables for political behavior analysis:
my_vars = [
    "V201032",   # Turnout intention: did respondent intend to vote?
    "V201033",   # Vote choice: candidate intended to vote for
    "V201507x",  # Respondent age (continuous / ratio)
    "V201200",   # Ideological self-placement (7-point scale: 1=Lib to 7=Con)
]

df = anes_data[my_vars].copy()
df.columns = ["vote", "vote_int", "age", "ideology"]

# In the REPL, inspect the initial shape and distributions:
df.shape
df.head()

# Notice that ANES uses negative integers for missingness:
#   -9 = Refused, -8 = Don't know, -1 = Inapplicable
df["age"].describe()
df["ideology"].value_counts().sort_index()


# ==============================================================================
# Relational Operators and Boolean Masks
# ==============================================================================
# Relational operators test conditions on every element of a Series:
#   >   Greater than
#   <   Less than
#   >=  Greater than or equal to
#   <=  Less than or equal to
#   ==  Equal to
#   !=  Not equal to
#
# In ANES 2020, valid respondents must be at least 18 years old.
# Let us construct a boolean mask for valid age values:

mask_age = df["age"] >= 18

# In the REPL, evaluate mask_age to verify that it is a boolean Series:
mask_age.head()

# Count how many respondents are valid (True) vs invalid (False):
mask_age.value_counts()

# Apply the mask to inspect the filtered subset:
df[mask_age].describe()

# Save the filtered DataFrame:
df = df[mask_age]
df.shape


# ==============================================================================
# Combining Conditions with Boolean Logic
# ==============================================================================
# In Python, standard logical operators ('and', 'or', 'not') cannot be used
# on pandas Series. Instead, pandas requires bitwise operators:
#   &  (AND) : Both conditions must be True
#   |  (OR)  : At least one condition must be True
#   ~  (NOT) : Inverts True and False
#
# CRITICAL: Always wrap each individual condition in parentheses!
# Bitwise operators have higher precedence than comparison operators.

# Cleaning the ideology variable (valid scale runs from 1 to 7):
# Method 1: Using bitwise & with parentheses:
mask_ideo = (df["ideology"] >= 1) & (df["ideology"] <= 7)

# Method 2: Using the built-in .between() method:
mask_ideo = df["ideology"].between(1, 7)

# Inspect summary statistics of valid ideological placements:
df[mask_ideo].describe()

# Update df to retain only valid observations on age and ideology:
df = df[mask_ideo]
df.shape

# Re-inspect the cleaned distributions using Altair:
clean_age_hist = (
    alt.Chart(df)
    .mark_bar(color="steelblue")
    .encode(
        x=alt.X(
            "age",
            type="quantitative",
            bin=alt.Bin(maxbins=30),
            title="Respondent Age",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Age (Cleaned)",
        width=450,
        height=250,
    )
)
clean_age_hist

clean_ideo_bar = (
    alt.Chart(df)
    .mark_bar(color="darkcyan")
    .encode(
        x=alt.X(
            "ideology",
            type="ordinal",
            title="Ideological Placement (1=Lib to 7=Con)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            title="Number of Respondents",
        ),
    )
    .properties(
        title="Distribution of Ideology (Cleaned)",
        width=450,
        height=250,
    )
)
clean_ideo_bar


# ------------------------------------------------------------------------------
# Intentional Error Challenge: Combining Boolean Conditions
# ------------------------------------------------------------------------------
# 1. Uncomment the line below and send it to the REPL with Shift+Enter.
# 2. Read the error message: what error did Python raise?
# 3. Fix the error and run it again.
# mask_error = df["ideology"] >= 1 and df["ideology"] <= 7


# ------------------------------------------------------------------------------
# Hack-Time: Cleaning Vote Choice and Turnout
# ------------------------------------------------------------------------------
# In ANES 2020:
#   - 'vote' records turnout intentions: 1 = Intend to vote, 2 = Do not intend.
#     Negative values (-9, -8, -1) represent non-responses.
#   - 'vote_int' records presidential vote choice (1=Biden, 2=Trump, etc.).
#
# Tasks:
#   1. In the REPL, inspect df["vote"].value_counts().sort_index().
#   2. Create a boolean mask keeping only valid responses for 'vote' (1 or 2).
#   3. Create a boolean mask keeping only valid candidate selections for
#      'vote_int' (positive values >= 1).
#   4. Use df[...] to check how many respondents remain with valid vote data.

...


# ==============================================================================
# Substantive Application: The Youth Vote in the United States
# ==============================================================================
# A central question in American electoral politics:
# Are young voters significantly more liberal than older cohorts?
#
# In political science research, the youth cohort is commonly defined as
# respondents aged 18 to 30. Let us construct a boolean mask:

mask_youth = df["age"] <= 30

# In the REPL, evaluate the mean ideology for both cohorts:
# (1 = Extremely Liberal, 4 = Moderate, 7 = Extremely Conservative)
df[mask_youth]["ideology"].mean()
df[~mask_youth]["ideology"].mean()


# ------------------------------------------------------------------------------
# Hack-Time: Youth Turnout Intentions
# ------------------------------------------------------------------------------
# Substantive hypothesis: While young voters lean more liberal, they often
# express lower turnout intentions compared to older voters.
#
# Tasks:
#   1. Calculate the percentage of young respondents (age <= 30) who intend
#      to vote (where vote == 1).
#   2. Calculate the percentage of older respondents (age > 30) who intend
#      to vote.
#   3. Does the survey evidence support the hypothesis of a youth turnout gap?

...


# ==============================================================================
# Types of Data and Levels of Measurement
# ==============================================================================
# In quantitative political analysis, data variables are classified into:
#
#   1. Discrete Data: Variables that take distinct, separate values (counts or
#      categories). Examples: number of political parties, seats in parliament.
#
#   2. Continuous Data: Variables that can take an infinite number of values
#      along an unbroken continuum. Examples: respondent age, vote share %.
#
# Stanley S. Stevens (1946) divided data into four measurement levels:
#
#   - Nominal: Differences of kind. Unordered qualitative categories.
#     Mathematical operations are meaningless; only mode/frequencies apply.
#     Example: Party affiliation (Democrat, Republican, Green), Country.
#
#   - Ordinal: Differences of degree. Categories have a meaningful order,
#     but the distance between points is not equal or constant.
#     Example: Education level (High School < Bachelor's < Master's < PhD).
#
#   - Interval: Equal, constant distances between values, but no true zero.
#     Example: 7-point ideology scale, feeling thermometer ratings.
#
#   - Ratio: Equal intervals with an absolute, non-arbitrary zero point.
#     Ratios are meaningful (e.g. 60 years old is twice as old as 30).
#     Example: Respondent age, campaign donations ($), vote counts.
#
# +-------------------+------------+----------+-----------------------------+
# | Measurement Level | Continuous | Discrete | Political Science Example   |
# +-------------------+------------+----------+-----------------------------+
# | Nominal           |            |    X     | Party choice (Dem, Rep)     |
# | Ordinal           |            |    X     | Education level (HS, BA)    |
# | Interval          |     X      |    X     | 7-point ideology scale      |
# | Ratio             |     X      |    X     | Respondent age, vote counts |
# +-------------------+------------+----------+-----------------------------+
#
# Why does this matter?
# Your level of measurement determines which statistical visualization and
# bivariate modeling technique is appropriate:
#   - Continuous & Continuous -> Scatter plots, Hexbin plots
#   - Discrete & Continuous   -> Box plots by group, Grouped distributions
#   - Discrete & Discrete     -> Cross-tabulations, Stacked bar charts


# ==============================================================================
# Continuous & Continuous: Scatter Plots and Overplotting
# ==============================================================================
# What is the relationship between respondent Age and Ideology?
# In theory, both can be treated as continuous / metric variables.

# Direct scatter plot (showing overplotting):
scatter_overplot = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x=alt.X(
            "ideology",
            type="quantitative",
            title="Ideology (1=Lib to 7=Con)",
        ),
        y=alt.Y(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
    )
    .properties(
        title="Age vs. Ideology (Overplotting Problem)",
        width=450,
        height=300,
    )
)
scatter_overplot

# Notice the "overplotting" issue: thousands of survey points overlap,
# hiding the density of observations. We can solve this in three ways:

# Solution 1: Random sampling:
scatter_sample = (
    alt.Chart(df.sample(200, random_state=42))
    .mark_circle(size=50)
    .encode(
        x=alt.X(
            "ideology",
            type="quantitative",
            title="Ideology (1=Lib to 7=Con)",
        ),
        y=alt.Y(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
    )
    .properties(
        title="Age vs. Ideology (Random Sample of 200 Respondents)",
        width=450,
        height=300,
    )
)
scatter_sample

# Solution 2: Alpha transparency (points darken where data clusters):
scatter_opacity = (
    alt.Chart(df)
    .mark_circle(size=30, opacity=0.04)
    .encode(
        x=alt.X(
            "ideology",
            type="quantitative",
            title="Ideology (1=Lib to 7=Con)",
        ),
        y=alt.Y(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
    )
    .properties(
        title="Age vs. Ideology (Alpha Opacity = 0.04)",
        width=450,
        height=300,
    )
)
scatter_opacity

# Solution 3: 2D binned heatmap (density binning):
heatmap_density = (
    alt.Chart(df)
    .mark_rect()
    .encode(
        x=alt.X(
            "ideology",
            type="quantitative",
            bin=alt.Bin(maxbins=7),
            title="Ideology (1=Lib to 7=Con)",
        ),
        y=alt.Y(
            "age",
            type="quantitative",
            bin=alt.Bin(maxbins=15),
            title="Respondent Age",
        ),
        color=alt.Color(
            "count()",
            type="quantitative",
            title="Count",
            scale=alt.Scale(scheme="blues"),
        ),
    )
    .properties(
        title="Age vs. Ideology (2D Density Heatmap)",
        width=450,
        height=300,
    )
)
heatmap_density


# ==============================================================================
# Discrete & Continuous: Box Plots by Group
# ==============================================================================
# When one variable is categorical/discrete and the other is continuous,
# box plots allow you to compare the distribution of the continuous variable
# across each category.
#
# Anatomy of a Box Plot:
#   - Center line: Median (50th percentile)
#   - Box edges: Interquartile Range (IQR, 25th to 75th percentile)
#   - Whiskers: Extend to 1.5 * IQR from box edges
#   - Points: Outliers beyond 1.5 * IQR

# Compare age distributions across ideological placement using Altair:
box_plot = (
    alt.Chart(df)
    .mark_boxplot(size=25)
    .encode(
        x=alt.X(
            "ideology",
            type="ordinal",
            title="Ideology (1=Extr Lib to 7=Extr Con)",
        ),
        y=alt.Y(
            "age",
            type="quantitative",
            title="Respondent Age",
        ),
    )
    .properties(
        title="Age Distribution Across Ideological Self-Placement",
        width=450,
        height=300,
    )
)
box_plot


# ==============================================================================
# Discrete & Discrete: Cross-Tabulations (Contingency Tables)
# ==============================================================================
# When analyzing two discrete/categorical variables, use pd.crosstab()
# to compute frequency counts and percentages across category intersections.

# Focus on major party presidential candidate preferences (1=Biden, 2=Trump):
df_twoparty = df[df["vote_int"].isin([1, 2])].copy()
df_twoparty["candidate"] = df_twoparty["vote_int"].map({
    1: "Biden (Dem)",
    2: "Trump (Rep)",
})

# 1. Raw frequencies (counts):
pd.crosstab(df_twoparty["ideology"], df_twoparty["candidate"])

# Notice that pd.crosstab() is equivalent to groupby().value_counts().unstack():
df_twoparty.groupby("ideology")["candidate"].value_counts().unstack().fillna(0)

# 2. Overall proportions (cell counts divided by grand total):
pd.crosstab(df_twoparty["ideology"], df_twoparty["candidate"], normalize=True)

# 3. Row percentages (normalized by index):
# "Within each ideological group, what percentage voted for Biden vs Trump?"
cand_pct = (
    pd.crosstab(
        df_twoparty["ideology"],
        df_twoparty["candidate"],
        normalize="index",
    )
    * 100
)
cand_pct.round(1)

# Visualizing cross-tabulations: Normalized stacked bar chart in Altair
stacked_bar = (
    alt.Chart(df_twoparty)
    .mark_bar()
    .encode(
        x=alt.X(
            "ideology",
            type="ordinal",
            title="Ideology (1=Extr Lib to 7=Extr Con)",
        ),
        y=alt.Y(
            "count()",
            type="quantitative",
            stack="normalize",
            title="Share within Ideological Group",
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color(
            "candidate",
            type="nominal",
            title="Candidate Choice",
            scale=alt.Scale(
                domain=["Biden (Dem)", "Trump (Rep)"],
                range=["#2b5c8f", "#d95f02"],
            ),
        ),
    )
    .properties(
        title="2020 Presidential Candidate Choice by Ideology",
        width=450,
        height=300,
    )
)
stacked_bar


# ------------------------------------------------------------------------------
# Hack-Time: Cross-Tabulating Turnout by Youth Status
# ------------------------------------------------------------------------------
# Tasks:
#   1. Add a new column to df called 'youth_group' where respondents aged
#      <= 30 are labeled "Youth" and those > 30 are labeled "Older".
#   2. Filter for respondents with valid turnout intention (vote in [1, 2]).
#   3. Compute a row-normalized cross-tabulation showing the percentage of
#      turnout intention (vote == 1) versus non-turnout (vote == 2) for
#      each age group.
#   4. Visualize this comparison with an Altair stacked or grouped bar plot.

...
