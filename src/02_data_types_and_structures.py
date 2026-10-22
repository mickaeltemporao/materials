"""
==============================================================================
02: Data Types and Structures in Python
==============================================================================

Learning Objectives:
  - Understand core primitive data types: int, float, str, bool, None
  - Use type() to inspect object types
  - Apply relational (comparison) operators to evaluate conditions
  - Combine conditions using logical operators: and, or, not
  - Create, index, slice, edit, and extend Python lists
  - Model structured survey respondents using nested lists
==============================================================================
"""

# ==============================================================================
# Data Types
# ==============================================================================
# Python works with several fundamental data types:
#
#   Type      | Short | Description     | Example
#   :---      | :---  | :---            | :---
#   integer   | int   | Whole numbers   | 42
#   float     | float | Decimals        | 0.42
#   string    | str   | Characters/text | "Liberal"
#   None      | None  | Missing/null    | None
#   boolean   | bool  | Logical truth   | True, False
#
# The type() function tells you what type of data an object holds.
# Each data type allows specific operations and restricts others.

a = 42
b = 5
c = a / b

year = 2020
party_1 = "Democratic"
party_2 = "Conservative"

description = """
The American National Election Studies (ANES)
are academically-run national surveys of voters
in the United States, conducted before and after
every presidential election.
"""

seven = "7"
three = "3"


# ------------------------------------------------------------------------------
# Hack-Time: Type Inspection
# ------------------------------------------------------------------------------
# Tasks:
#   1. Use type() to check the type of a, c, party_1, and description.
#   2. What happens if you add seven + three? Run the code to find out!
#   3. Document your observations with a comment below.

...


# ==============================================================================
# Booleans and Missing Information (None)
# ==============================================================================
# Booleans represent logical truth values: True or False (capitalized
# in Python).
# They are essential for filtering survey data and evaluating hypotheses.

vote_2016 = True
vote_2020 = False

# In Python, True behaves numerically as 1, and False behaves as 0:
True + True
True + False

# Python uses None to represent missing, empty, or unobserved values
# (analogous to NA in R or NULL in SQL):
results_2024 = None
type(results_2024)


# ------------------------------------------------------------------------------
# Hack-Time: Missing Data Operations
# ------------------------------------------------------------------------------
# Tasks:
#   1. Check what happens if you try to add a number to results_2024 (e.g. 538).
#   2. Observe the error: why can you not add an integer to NoneType?

...


# ==============================================================================
# Relational (Comparison) Operators
# ==============================================================================
# Relational operators compare two values and return a boolean (True or False):
#
#   Operator | Description      | Example
#   :---:    | :---             | :---
#     ==     | Equal to         | a == b
#     !=     | Not equal to     | a != b
#     <      | Less than        | a < b
#     <=     | Less or equal    | a <= b
#     >      | Greater than     | a > b
#     >=     | Greater or equal | a >= b

a == 42
b > 10
year != 2024


# ------------------------------------------------------------------------------
# Hack-Time: Comparing Objects
# ------------------------------------------------------------------------------
# Tasks:
#   1. Test if year is greater than or equal to 2020.
#   2. Test if party_1 is equal to party_2.
#   3. Test if a * b is strictly less than 250.

...


# ==============================================================================
# Logical Operators
# ==============================================================================
# Logical operators join boolean expressions to evaluate complex conditions:
#
#   Operator | Description                     | Example
#   :---:    | :---                            | :---
#     and    | True only if both sides True    | True and False -> False
#     or     | True if at least one side True  | True or False  -> True
#     not    | Inverts truth value             | not True       -> False

True and True
True and False
False or True
not False

# Consider a fictional survey respondent:
age = 22
sex = "Female"
country = "United States"
employed = True
voted = True
twitter = False

# Testing joint conditions:
country == "United States" and age >= 18
employed and not twitter


# ------------------------------------------------------------------------------
# Hack-Time: Evaluating Survey Eligibility
# ------------------------------------------------------------------------------
# Tasks:
#   1. Under US law, to run for President, a candidate must be born in the US
#      and at least 35 years old. Write a single logical condition testing if
#      this respondent is eligible to run for President.
#   2. Test if the respondent either voted OR uses Twitter.

...


# ==============================================================================
# Lists as Data Structures
# ==============================================================================
# Lists group multiple objects of any type into a single ordered sequence.
# To create a list, place comma-separated values inside square brackets [ ]:

respondent_1 = [age, sex, country, employed, voted, twitter]
respondent_1

type(respondent_1)
len(respondent_1)


# ------------------------------------------------------------------------------
# Hack-Time: Creating Lists
# ------------------------------------------------------------------------------
# Tasks:
#   1. Replace the ... placeholders below with values for a new respondent:
age_2 = ...
sex_2 = ...
country_2 = ...
employed_2 = ...
voted_2 = ...
twitter_2 = ...

respondent_2 = [age_2, sex_2, country_2, employed_2, voted_2, twitter_2]

#   2. Create a list named respondents containing [respondent_1, respondent_2].

...


# ==============================================================================
# List Manipulation: Subsetting & 0-Indexing
# ==============================================================================
# Most programming languages are 0-indexed: the first item is at index 0!
#
#   parties = ["democratic", "republican", "libertarian", "green"]
#   Index:          0              1             2           3

parties = ["democratic", "republican", "libertarian", "green"]
parties[0]
parties[1]

# Real electoral turnout dataset:
data = [
    "turnout", 2012, 54.9, "democratic",
    2016, 55.7, "republican",
    2020, 66.8, "democratic"
]
data[1]


# ------------------------------------------------------------------------------
# Hack-Time: Subsetting Lists
# ------------------------------------------------------------------------------
# Tasks:
#   1. Using the data list, print the turnout for the 2016 election.
#   2. Calculate the average turnout of 2012 (index 2) and 2016 (index 6).

...


# ==============================================================================
# List Manipulation: Slicing ([start:end])
# ==============================================================================
# Slicing extracts a range of elements: list[start:end].
# Note: start is INCLUSIVE, but end is EXCLUSIVE!

# Extracts elements at indices 4 and 5 (year 2016 and turnout 55.7):
data[4:6]


# ------------------------------------------------------------------------------
# Hack-Time: Slicing Lists
# ------------------------------------------------------------------------------
# Tasks:
#   1. Slice the winning party name of the 2020 election from data.
#   2. Slice the first four elements of data.

...


# ==============================================================================
# List Manipulation: Editing Elements
# ==============================================================================
# Lists are mutable: you can reassign any element by referencing its index:

data[7] = "Republican"
data


# ------------------------------------------------------------------------------
# Hack-Time: Editing List Elements
# ------------------------------------------------------------------------------
# Tasks:
#   1. Fix the value at index 7 back to "democratic".
#   2. Verify your change by evaluating data.

...


# ==============================================================================
# List Manipulation: Extending Lists
# ==============================================================================
# You can concatenate lists together using the + operator:

year_2008 = [2008, 58.2, "democratic"]
data = data + year_2008
data


# ------------------------------------------------------------------------------
# Hack-Time: Extending Lists
# ------------------------------------------------------------------------------
# Tasks:
#   1. Create a list year_2004 = [2004, 60.1, "republican"].
#   2. Concatenate year_2004 to data.
#   3. Check the new length of data using len().

...


# ==============================================================================
# Nested Lists: Modeling Datasets
# ==============================================================================
# In political science, datasets are collections of respondents (rows).
# A list of lists is our first step toward tabular data:

respondent_3 = [67, "Male", "New Zealand", False, True, False]
survey_sample = [respondent_1, respondent_2, respondent_3]

# Accessing row 0, column 0 (age of respondent 1):
survey_sample[0][0]


# ------------------------------------------------------------------------------
# Hack-Time: Analyzing Nested Survey Data
# ------------------------------------------------------------------------------
# Tasks:
#   1. Access the country of respondent 3 using index notation.
#   2. Calculate the average age across respondent_1 and respondent_3.

...
