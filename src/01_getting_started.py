"""
==============================================================================
01: Getting Started with Python for Political Science Data Analysis
==============================================================================

Learning Objectives:
  - Run Python code interactively in VS Code using Smart Send (Shift+Enter)
  - Understand how Python executes expressions interactively
  - Document code using comments (#)
  - Use Python as a calculator for arithmetic operations
  - Create and name objects (variable assignments)
  - Combine variables with arithmetic to answer substantive political questions
==============================================================================
"""

# ==============================================================================
# Why Python for Political Science?
# ==============================================================================
# - Free and open-source general-purpose programming language.
# - Enormous scientific ecosystem with over 600,000+ packages on PyPI.
#   (Think of packages like apps on a smartphone: someone already built it!)
# - Welcoming global community with extensive online documentation.
# - Reproducible scientific workflow: no point-and-click black boxes.
# - High-demand analytical skill for political science research, polling,
#   public policy, and data journalism.


# ==============================================================================
# Running Python in VS Code (Interactive Execution)
# ==============================================================================
# If you have used R or RStudio before, this workflow will feel very familiar:
#   - In RStudio: you press Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)
#     to send code from your .R script directly to the R Console.
#   - In VS Code: you press Shift+Enter on any line or highlighted block
#     to run it directly in the interactive terminal below!
#
# Try it now: place your cursor on the line below and press Shift+Enter!
print("Greetings!")

# ==============================================================================
# Printing Without a Printer
# ==============================================================================
# In Python, we use the built-in print() function to explicitly display output.
print("I'm excited to learn Python!")

# When you want to print text characters (known as strings in Python),
# wrap your text inside quotation marks (consistently use double quotes "...").


# ------------------------------------------------------------------------------
# Hack-Time: Printing Practice
# ------------------------------------------------------------------------------
# Tasks:
#   1. Print "Hello!"
#   2. Print your first and last name.
# (Write your code below and run each line with Shift+Enter)

...


# ==============================================================================
# Code Execution, Sequential Flow & Comments
# ==============================================================================
# Python executes code sequentially from top to bottom, like a recipe.
#
# The hash symbol (#) is used to write comments.
# Anything following a # on a line is completely ignored by Python.
# Comments are essential for documenting your assumptions, steps, and
# hypotheses so that collaborators (and your future self!) understand the code.

print(" _____")
# This is a comment that will be ignored when you run this line.
print("|     |")
# Here's another comment. They are always ignored by Python.
print("|     |")
# You can use comments to document steps, formulas, and hypotheses.
print("|_____|")


# ------------------------------------------------------------------------------
# Hack-Time: Documenting Code
# ------------------------------------------------------------------------------
# Run the code below with Shift+Enter, see what it does, and add a comment
# explaining what calculation was performed!

print("Let's do some math!")
print(5 + 7)
...


# ==============================================================================
# Arithmetic Operators (Python as a Calculator)
# ==============================================================================
# In its most basic form, Python can be used as a calculator for math:
#
#   Operator | Description | Example
#   :---:    | :---        | :---
#     +      | Sum         | 5 + 2    -> 7
#     -      | Difference  | 10 - 3   -> 7
#     *      | Product     | 4 * 2    -> 8
#     /      | Quotient    | 15 / 2   -> 7.5
#     **     | Power       | 2 ** 3   -> 8
#     %      | Modulus     | 7 % 3    -> 1 (remainder of division)


# ------------------------------------------------------------------------------
# Hack-Time: Math Operations
# ------------------------------------------------------------------------------
# Tasks:
#   1. Divide 7 by 3.
#   2. Raise 2 to the 5th power.
#   3. Perform another mathematical operation of your choice, and add a comment
#      documenting what your operation does.

...


# ==============================================================================
# The Assignment Operator (=) & Creating Objects
# ==============================================================================
# Objects (variables) store information in memory (numbers, text, survey data).
# To assign a value to an object, we use the equal sign: =
# Syntax: object_name = value

turnout_1 = 55.7

# Notice: when you assign a value, Python saves it in memory silently.
# When running interactively, you do NOT always need print() to inspect it!
# Simply run the variable name itself to display its value:
turnout_1

# Let's create more objects storing real-world electoral data from the
# 2016 and 2020 United States presidential elections:
election = "United States presidential election"

year_1 = 2016
turnout_1 = 55.7
winner_1 = "Republican Party"

year_2 = 2020
turnout_2 = 66.8
winner_2 = "Democratic Party"

# You can print multiple objects together separated by commas:
print(election, year_1)
print(winner_1, turnout_1)

print(election, year_2)
print(winner_2, turnout_2)


# ==============================================================================
# Case Sensitivity & Intentional Debugging
# ==============================================================================
# Python is strictly CASE-SENSITIVE:
# 'winner_2', 'Winner_2', and 'WINNER_2' are treated as three distinct names!


# ------------------------------------------------------------------------------
# Hack-Time: Spot the Error!
# ------------------------------------------------------------------------------
# Task:
#   1. Uncomment the line below (remove the leading #) and run it
#      with Shift+Enter.
#   2. Read the error message carefully: what type of error did Python raise?
#   3. Fix the capitalization error and run it again.

# print(Winner_2)
...


# ==============================================================================
# Combining Variables with Arithmetic
# ==============================================================================
# You can combine stored variables with arithmetic operations to answer
# substantive political science questions.
#
# When you run expressions interactively, Python displays their values:

sum_val = 3 + 14
sum_val

difference_val = 10 - 3
difference_val

product_val = 42 * 2.41
product_val

# Now let's analyze voter turnout trends using our election variables:

# Question 1: By how many percentage points did turnout rise in 2020 vs 2016?
diff_turnout = turnout_2 - turnout_1
diff_turnout

# Question 2: What is the relative percentage change in turnout?
pct_change = (turnout_2 - turnout_1) / turnout_1 * 100
pct_change

# Question 3: What was the average turnout across both elections?
avg_turnout = (turnout_1 + turnout_2) / 2
avg_turnout


# ------------------------------------------------------------------------------
# Hack-Time: 2012 US Presidential Election
# ------------------------------------------------------------------------------
# In the 2012 US Presidential Election:
#   - Year: 2012
#   - Voter turnout: 54.9%
#   - Winning party: "Democratic Party"
#
# Task 1: Replace the ... (Ellipsis) placeholders below with 2012 election data:
year_3 = ...
turnout_3 = ...
winner_3 = ...

# Task 2: By how many percentage points did turnout change in 2016 vs 2012?
# (Write your code below)
...

# Task 3: What is the average turnout across all 3 elections (2012, 2016, 2020)?
# (Write your code below)
...


# ==============================================================================
# Naming Guidelines in Python
# ==============================================================================
# When writing Python code for political data analysis, follow these rules:
#   1. Python is case-sensitive (vote_democrat != Vote_Democrat).
#   2. Variable names MUST start with a letter (or underscore).
#      They cannot start with a number.
#   3. Variable names can only contain alphanumeric characters and underscores
#      (a-z, A-Z, 0-9, _). No spaces, dashes, or special symbols (-, @, $, %).
#   4. Variable names should be descriptive and meaningful
#      (e.g., 'turnout_2020' instead of 'x').
#   5. Use snake_case: all lowercase words separated by underscores
#      (e.g., 'ideology_score', 'party_id').
