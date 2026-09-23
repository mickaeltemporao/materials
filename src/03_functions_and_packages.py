"""
==============================================================================
03: Functions and Packages for Political Science Data Analysis
==============================================================================

Learning Objectives:
  - Understand functions and how they eliminate repetitive code
  - Use the built-in help() function and inspect function parameters
  - Distinguish between general functions and object-specific methods
  - Chain methods together to clean and process text
  - Define custom functions using def, docstrings, parameters, and return
  - Import standard and third-party modules (math, random, pandas)
  - Load real-world survey data from the ANES 2020 using pandas
==============================================================================
"""

# ==============================================================================
# Functions in Python
# ==============================================================================
# A function is a named block of code that performs a specific task.
# You pass data into a function (arguments/parameters), and it can return
# a calculated result.
#
# Why do functions matter in empirical research?
#   - Organize code into logical, transparent analytical steps.
#   - Avoid repetition (DRY: Don't Repeat Yourself).
#   - Make empirical analyses modular, testable, and reproducible!
#
# Calling syntax:
#   function_name(arguments)
#   output = function_name(arguments)


# ------------------------------------------------------------------------------
# Hack-Time: Functions Used So Far
# ------------------------------------------------------------------------------
# Tasks:
#   1. Write and evaluate three built-in functions we used in lessons 01 and 02.
#   2. Use type() to check the type of the print function itself.




# ==============================================================================
# The help() Function & Parameter Inspection
# ==============================================================================
# Python includes an interactive documentation system built into the REPL.
# Pass any function, module, or object to help() to read its documentation:

help(len)

# Functions can take optional keyword parameters that modify their behavior.
# For example, print() accepts a sep parameter (separator between items):
print("Democrat", "Republican", "Independent", sep=" | ")


# ------------------------------------------------------------------------------
# Hack-Time: Exploring Documentation
# ------------------------------------------------------------------------------
# Tasks:
#   1. Use help(print) to see all optional parameters accepted by print().
#   2. Print three words separated by three dashes ("---") using sep.




# ==============================================================================
# Methods: Object-Specific Functions
# ==============================================================================
# A method is a function that belongs to a specific type of object.
# While general functions are called as function(object), methods are called
# directly on the object using dot notation: object.method()

sentence = "The American National Election Studies survey voters."

# String methods:
sentence.upper()
sentence.lower()
sentence.count("election")

# Lists have their own distinct methods:
turnout_sample = [55.7, 66.8, 54.9, 58.2]
turnout_sample.append(60.1)
turnout_sample


# ------------------------------------------------------------------------------
# Hack-Time: String and List Methods
# ------------------------------------------------------------------------------
# Tasks:
#   1. Create a list party = ["Dem", "Rep", "Dem", "Dem", "Rep", "Rep", "Dem"].
#   2. Use a list method to count how many times "Dem" appears in the list.




# ==============================================================================
# Method Chaining
# ==============================================================================
# When a method returns an object of the same or another compatible type,
# you can chain a second method directly onto the result:

text = "   POLITICAL SCIENCE   "
text.strip().lower()


# ------------------------------------------------------------------------------
# Hack-Time: Method Chaining
# ------------------------------------------------------------------------------
# Tasks:
#   1. Given candidate = "  kamala harris  ", chain strip() and title()
#      to clean leading whitespace and capitalize the words properly.




# ==============================================================================
# Defining Custom Functions
# ==============================================================================
# When you need to repeat a set of calculations on different variables,
# define your own function with the def keyword.
#
# Standard structure:
#   def function_name(parameter_1, parameter_2):
#       """Docstring explaining what the function calculates."""
#       result = parameter_1 + parameter_2
#       return result

def greet(name):
    """Return a personalized greeting."""
    return f"Welcome to data analysis, {name}!"

greet("Alex")


# ------------------------------------------------------------------------------
# Hack-Time: Writing Basic Functions
# ------------------------------------------------------------------------------
# Tasks:
#   1. Define a function calculate_turnout_diff(t1, t2) that subtracts
#      t1 from t2 and returns the difference.
#   2. Test your function using the 2016 (55.7) and 2020 (66.8) turnouts.




# ==============================================================================
# Functions with Statistical Calculations
# ==============================================================================
# Let us write functions to calculate summary statistics from raw lists:

def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers."""
    total = sum(numbers)
    count = len(numbers)
    return total / count

turnout_history = [66.8, 55.7, 54.9, 58.2, 56.7, 51.2, 49.0]
calculate_mean(turnout_history)


# ------------------------------------------------------------------------------
# Hack-Time: Statistical Functions
# ------------------------------------------------------------------------------
# Tasks:
#   1. Define a function summarize_turnout(numbers) that returns both
#      the mean and the number of observations (as a tuple or list).
#   2. Call summarize_turnout(turnout_history) and inspect the output.




# ==============================================================================
# Modules and Packages
# ==============================================================================
# A module is a Python file containing functions and definitions.
# A package is a collection of modules.
#
# Python comes with a standard library of pre-installed modules.
# To use a module, bring it into memory with import:

import math

math.sqrt(64)
math.pi


# ------------------------------------------------------------------------------
# Hack-Time: Using the Math Module
# ------------------------------------------------------------------------------
# Tasks:
#   1. Check the documentation of math.floor using help(math.floor).
#   2. Apply math.floor to 66.8.




# ==============================================================================
# Aliases and Partial Imports
# ==============================================================================
# You can give an imported package a shorter alias using the as keyword:
#   import pandas as pd
#
# Or import only specific functions from a module:
#   from random import randint

from random import randint

randint(1, 100)


# ------------------------------------------------------------------------------
# Hack-Time: Importing with Aliases
# ------------------------------------------------------------------------------
# Tasks:
#   1. Import the datetime module using the alias dt.
#   2. Inspect dt using type().




# ==============================================================================
# Introduction to pandas
# ==============================================================================
# pandas is the fundamental data science package for working with tabular data
# (data frames). In political science, almost all survey datasets (ANES, ESS,
# Afrobarometer) and country-level indicators (V-Dem, World Bank) are loaded
# and manipulated using pandas.

import pandas as pd

# Load real survey data from the American National Election Studies (ANES 2020):
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
df = pd.read_csv(data_url, compression="gzip")

# In the REPL, simply send df.head() to inspect the first 5 rows:
df.head()


# ------------------------------------------------------------------------------
# Hack-Time: Inspecting ANES 2020 Variables
# ------------------------------------------------------------------------------
# In the ANES 2020 dataset:
#   - Variable V201033 records 2020 presidential vote intention:
#       1 = Joe Biden (Democrat)
#       2 = Donald Trump (Republican)
#
# Tasks:
#   1. Send df["V201033"].value_counts() to the REPL to inspect frequencies.
#   2. Add normalize=True inside value_counts() to see proportions instead!
