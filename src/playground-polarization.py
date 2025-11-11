# Load Pandas
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col

# Step 1: Load, select & rename Variables
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data = pd.read_csv(data_url, compression='gzip')

# How does political knowledge impact levels of 
# affective polarization

# H1: low levels lead to more affective polarization

vars = {
    "V201033": "vote_intention",
    "V201507x": "age",
    "V201600": "sex",
    "V201511x": "education",
    "V201617x": "income",
    "V201228": "party_id",
    "V201231x": "party_id_str",
    "V201232": "party_id_imp",
    "V201200": "ideology",
    "V201156": "feeling_democrat", 
    "V201157": "feeling_republican",
    "V201641": "political_knowledge_intro",
    "V201642": "political_knowledge_catch1",
    "V201643": "political_knowledge_catch_feedback",
    "V201644": "political_knowledge_senate_term",
    "V201645": "political_knowledge_least_spending",
    "V201646": "political_knowledge_house_majority",
    "V201647": "political_knowledge_senate_majority",
    "V202406": "political_interest",
    "V202407": "follow_politics_media",
    "V202408": "understand_issues"
}

# Selecte & Rename variables to make them more descriptive
df = anes_data[vars.keys()]
df = df.rename(columns=vars)

# Step 2: Clean & Create Relevant Variables
# We will create a political knowledge scale by summing the correct answers to the political knowledge questions.

# Define a function to clean and recode political knowledge responses
def clean_knowledge_variable(series, correct_values):
    # Replace invalid codes with NaN
    series_cleaned = series.replace([-9, -5, -4, -1], np.nan)
    # Recode correct answers as 1, others as 0
    series_cleaned = series_cleaned.apply(lambda x: 1 if x in correct_values else 0)
    return series_cleaned
#

df['political_knowledge_senate_term'] = clean_knowledge_variable(df['political_knowledge_senate_term'], [6])
df['political_knowledge_least_spending'] = clean_knowledge_variable(df['political_knowledge_least_spending'], [1])
df['political_knowledge_house_majority'] = clean_knowledge_variable(df['political_knowledge_house_majority'], [1])
df['political_knowledge_senate_majority'] = clean_knowledge_variable(df['political_knowledge_senate_majority'], [2])

# Apply an Awareness filter
df = df[df["political_knowledge_catch1"].between(1000,2000)]

# Creating a political knowledge scale
political_knowledge_vars = [
    "political_knowledge_senate_term",
    "political_knowledge_least_spending",
    "political_knowledge_house_majority",
    "political_knowledge_senate_majority"
]

df['political_knowledge_scale'] = df[political_knowledge_vars].sum(axis=1)

df['political_knowledge_scale'] = df[political_knowledge_vars].sum(axis=1)


df['pk_dummy'] = df['political_knowledge_scale'] == 4 
df['pk_dummy'] = df['pk_dummy'].astype(int)
df['pk_dummy'].value_counts()


# TODO: Quick Data Cleaning | WARNING We drop irrelevant observations!

# age
df['age'].describe()
mask = df['age'] >= 18
df = df[mask]

# sex
df['sex'].value_counts()
mask = df['sex'].between(1,2)
df = df[mask]
df['sex'] = df['sex'].apply(lambda x: 1 if x == 1 else 0)
# Or with map
# df['sex'] = df['sex'].map({1: 1, 2: 0})

# education
df['education'].value_counts()
mask = df['education'] > 0
df = df[mask]

# income
df['income'].value_counts()
mask = df['income'] > 0
df = df[mask]

# ideology
df['ideology'].value_counts()
mask = df['ideology'].between(1,7)
df = df[mask]

# party_id & other related
df['party_id'].value_counts()
mask = df['party_id'].between(1,3)
df = df[mask]

df['party_id_str'].value_counts()
mask = df['party_id'].between(1,7)
df = df[mask]

df['party_id_imp'].value_counts()
mask = df['party_id_imp'].between(1,5)
df = df[mask]

# vote_intention 
df = df[df['vote_intention'].between(1,2)]  # We are keeping intentions for major parties

# You could also skip the mask step 
# political_interest
df['political_interest'].value_counts()
df = df[df['political_interest'] > 0]

# follow_politics_media 
df['follow_politics_media'].value_counts()
df = df[df['follow_politics_media'] > 0]

# understand_issues
df['understand_issues'].value_counts()
df = df[df['understand_issues'] > 0]


### Step 3: Build Quantities of Interests
# Build an Affective Polarization Variable
# Calculate the affective polarization based on feeling thermometer scores.
mask = (df['feeling_democrat'] >= 0) & (df['feeling_republican'] >= 0 )
df = df[mask]

df['affective_polarization'] = np.abs(df['feeling_democrat'] - df['feeling_republican'])

# Take a look at our DV
df['affective_polarization'].plot(kind='kde', title='Density Plot')

# Or using Seaborn!
sns.kdeplot(
   data=df, x="affective_polarization", hue="sex",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)

# Take a look at the IV
df['political_knowledge_scale'].value_counts().sort_index().plot(
    kind='bar',
    title="Distribution of Political Knowledge Scale"
)

# Step 4: Modeling - Build a Regression Model
# Use the cleaned data stored in `df` to build a regression model. 
# We will include control variables such as age, sex, education, income, and ideology.

# First we define the model formula
# It takes the following form DV ~ IVs

formula = "affective_polarization ~ political_knowledge_scale + age + sex + education + ideology"

# Fit the regression model
model = sm.ols(formula=formula, data=df).fit()

# Print the summary of the regression model
model.summary()

# Export model to Markdown/Latex
print(summary_col([model]).as_latex())

# Step 5: Visualize Results
# Create a visualization to summarize the results of the regression model.

# Create a DataFrame with coefficients and confidence intervals
coef_df = pd.DataFrame({
    'coef': model.params,
    'lower_ci': model.conf_int()[0],
    'upper_ci': model.conf_int()[1],
    'pval': model.pvalues
}).drop('Intercept')


# Make the figure
plt.figure(figsize=(8, 10))
# Plot each coefficient with its confidence interval
plt.errorbar(coef_df['coef'], coef_df.index, xerr=(coef_df['coef'] - coef_df['lower_ci'], coef_df['upper_ci'] - coef_df['coef']), fmt='o', color='b', elinewidth=2, capsize=4)
plt.axvline(x=0, color='grey', linestyle='--')  # Add a vertical line at zero for reference
plt.title('Regression Coefficients with Confidence Intervals')
plt.xlabel('Coefficient')
plt.ylabel('Variables')
plt.yticks(ticks=range(len(coef_df)), labels=coef_df.index)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()


# Create a visualization of the predicted values of affective polarization 
# across the range of political knowledge scores.

predicted_values_full = model.predict(x)
plt.figure(figsize=(10, 6))
plt.scatter(df['political_knowledge_scale'], y, label='Actual Values', color='blue', alpha=0.5)
plt.scatter(df['political_knowledge_scale'], predicted_values_full, label='Predicted Values', color='red', alpha=0.5)
# Add labels and title
plt.title('Actual vs. Predicted Affective Polarization')
plt.xlabel('Political Knowledge Scale')
plt.ylabel('Affective Polarization')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

