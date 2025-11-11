# Load Pandas
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load, select & rename Variables
data_url = "https://raw.githubusercontent.com/datamisc/ts-2020/main/data.csv"
anes_data  = pd.read_csv(data_url, compression='gzip')

# Prepare data
clean_data = anes_data[['V201507x', 'V201200']].rename(columns={'V201507x': 'age', 'V201200': 'ideology'})
clean_data = clean_data[(clean_data['ideology'].between(1, 7)) & (clean_data['age'] >= 18)]

# Relationship between variables can be complex to see

# Add random lines
models = pd.DataFrame({
    'a1': np.random.uniform(-5, 5, 50),
    'a2': np.random.uniform(-0.25, 0.25, 50)
})

# Is the relationship observed due to random chance?

# To avoid biases, quantify the strength of relationships using linear regression from `statsmodels`.

# What is a model?

# Create data
my_data = pd.DataFrame({
    'time_to_iep': [16.93, 19.49, 18.21, 19.09, 17.67, 18.48, 16.37, 17.57, 19.18, 18.74, 17.15, 17.76, 17.2, 19.78, 18.34,
                    17.93, 18.09, 17.14, 19.41, 17.99, 16.54, 18.42, 16.65, 19.83, 18.32, 18.13, 16.72, 18.05, 18.5, 19.45,
                    17.22, 17.32, 19.48, 18.93, 18.69, 18.78, 18.58, 18.8, 18.28, 20.06, 18.12, 18.64, 18.16, 17.44, 18.96,
                    17.55, 19.09, 17.95, 21.01, 18.19]
})

# Visualize
plt.figure(figsize=(8, 6))
sns.histplot(my_data['time_to_iep'], alpha=.5)
plt.axvline(my_data['time_to_iep'].mean(), color='red', linestyle='dashed', linewidth=1.5)
plt.text(my_data['time_to_iep'].mean(), 4.5, f"{my_data['time_to_iep'].mean():.2f}", color='red', ha='right')

my_data['time_to_iep'].plot.hist(alpha=0.5, bins=10, edgecolor='black')
# Add a vertical line at the mean
mean_value = my_data['time_to_iep'].mean()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=5)
# Annotate the mean value on the plot
plt.text(mean_value, 4.5, f" {mean_value:.2f}", color='red', ha='left', fontsize=16)
plt.xlabel('Time to IEP')
plt.ylabel('Frequency')
plt.title('Histogram of Time to IEP')


# Hack Time!

# If you're interested in how the dataset used in this session was generated: https://github.com/mickaeltemporao/CMT4A-CMSS-TEMPLATE/blob/main/src/make_data.R

# Session 6: Data Modeling

# Import and load data
data_url = "https://github.com/mickaeltemporao/CMT4A-CMSS-TEMPLATE/raw/main/data/clean_2016.rds"
tb = pd.read_pickle(data_url)

# Always look at the data first!
print(tb.head())

# Last time we ended by checking the relationship between ideology and age
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tb, x='age', y='ideology')
plt.show()

# Let's add the line of best fit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tb, x='age', y='ideology')
sns.regplot(data=tb, x='age', y='ideology', scatter=False)
plt.show()

# Linear Models in Python
import statsmodels.api as sm

# Fit a model of ideology as a function of age
X = tb['age']
y = tb['ideology']
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
print(model.summary())

# Is there an effect of party identification on ideology
X = tb['party_id']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Is there an effect of education on ideology?
X = tb['education']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Is there an effect about gender on ideology?
X = tb['gender']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Comprehensive model
X = tb[['age', 'party_id', 'education', 'gender']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
