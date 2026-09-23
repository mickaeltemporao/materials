# Guidelines for Political Science Data Analysis Teaching Codebase (src/*.py)

This document establishes the architecture, pedagogical philosophy, and
formatting conventions for converting and maintaining interactive Python
teaching scripts in `src/`.

---

## 1. Target Environment & Execution Philosophy
- **VS Code Smart Send / REPL Workflow:** All scripts in `src/` are designed
  for students running code interactively using VS Code Smart Send
  (`Shift+Enter`) to an active Python REPL (matching the RStudio `.R` script
  to Console workflow).
- **Interactive Evaluation vs. `print()`:** In the interactive REPL, sending
  a variable name or expression directly evaluates and displays it
  (e.g., `turnout_1`, `df.head()`, `diff_turnout`). Do **not** wrap every
  evaluation in `print()`—reserve `print()` for explicit message output.
- **Top-to-Bottom Executability:** The script must run cleanly from top to
  bottom (e.g., `python src/script.py`) with 0 unhandled syntax or runtime
  errors.

---

## 2. Formatting & Style Rules (Strict PEP 8 Line Length & Spacing)
- **Line Length Limit (Max 80 Characters):** Every line of code, comment,
  docstring, and divider line must be strictly **<= 80 characters**.
  Never exceed column 80!
- **Section Dividers (Exactly 80 Characters):**
  - Major topic headers use double-line dividers (`# ` + 78 `=`):
    ```python
    # ====================================================================
    # Descriptive Topic Title
    # ====================================================================
    ```
  - Hack-Times use single-line dividers (`# ` + 78 `-`):
    ```python
    # --------------------------------------------------------------------
    # Hack-Time: Exercise Title
    # --------------------------------------------------------------------
    ```
- **Uniform Student Exercise Spacing:** Always leave **exactly 2 blank lines**
  below each Hack-Time task/exercise for student code.
- **Module Header Docstrings:** Keep headers clean and focused on Title and
  Learning Objectives. Do not include redundant `Course:` or `File:` lines:
  ```python
  """
  ====================================================================
  01: Lesson Name
  ====================================================================

  Learning Objectives:
    - Objective 1
    - Objective 2
  ====================================================================
  """
  ```

---

## 3. Pedagogical Rules for Hack-Times (Strict: No Spoilers!)
- **No Answers in Comments:** Provide clear, structured instructions and
  background context, but **NEVER** provide the solution code or commented-out
  answers. Leave blank space for students to write code.
- **Self-Documentation:** Only ask students to document their own code or
  operations, not lines that already have teacher explanations.
- **The Ellipsis (`...`) Convention:** When providing templates for students
  to complete, use `...` (Ellipsis) as the placeholder (e.g., `year_3 = ...`).
  It is valid Python syntax, avoids parser crashes, and raises informative
  errors if evaluated prematurely.

---

## 4. Intentional Errors & Debugging Challenges
- **Commented Lines:** Intentional errors (e.g., case-sensitivity typos or
  unimported packages) must be commented out by default so the script can
  execute top-to-bottom without unhandled exceptions.
- **Instructions:** Tell the student:
  ```python
  # 1. Uncomment the line below and send it to the REPL with Shift+Enter.
  # 2. Read the error message: what error did Python raise?
  # 3. Fix the error and run it again.
  # print(Winner_2)
  ```
- **No Spoilers:** Do not reveal the exact error type or the corrected code
  in the prompt comments. Let the student inspect the traceback.

---

## 5. Curriculum Sequence in `src/`
1. `src/01_getting_started.py`: Basics, math, variables, VS Code REPL intro.
2. `src/02_data_types_and_structures.py`: Types, booleans, lists, slicing.
3. `src/03_functions_and_packages.py`: Functions, methods, packages, pandas.
4. `src/04_data_exploration_columns.py`: Series & DataFrames, summary stats.
5. `src/05_data_exploration_rows.py`: Filtering, masks, scales, cross-tabs.
6. `src/06_data_management_and_scales.py`: Recoding, binning, scale index.
7. `src/07_intro_to_modeling.py`: What is a model? Mean as baseline, OLS.
8. `src/08_linear_regression.py`: Multiple regression, controls, Typst table.
9. `src/09_lab_modeling.py`: Hands-on modeling lab script for term projects.
10. `src/10_visualization_insights.py`: Altair declarative graphics and comms.
