import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# Load the Excel file
df = pd.read_csv('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/MBA 551 - Predictive Analytics/hdata modified 1.csv')

# Display the first few rows of the dataframe
print(df.head())

# Check for missing values and data types
print(df.info())
missing_values = df.isnull().sum()

# Select relevant columns that might relate to internet usage, health outcomes, satisfaction, and demographics
relevant_columns = [
    'InternetConnection', 'ConfidentInternetHealth', 'Electronic2_HealthInfo', 'Electronic2_MessageDoc',
    'ReceiveTelehealthCare', 'Telehealth_GoodCare',
    'CancerConcernedQuality', 'LifeHasMeaning', 'ClearSenseDir', 'DeepFulfillment',
    'AgeGrpA', 'Education', 'RACE_CAT2', 'HHInc'
]
selected_data = df[relevant_columns]

# Print information about selected columns
print("Missing Values:\n", missing_values[relevant_columns])
print("Selected Data Descriptive Stats:\n", selected_data.describe(include='all'))

# Visualization setup
plt.figure(figsize=(16, 15))

# Subplot 1: Internet Confidence vs. Health Info usage
plt.subplot(2, 2, 1)
sns.countplot(x='ConfidentInternetHealth', hue='Electronic2_HealthInfo', data=selected_data)
plt.title('Confidence in Internet Health vs. Health Info Usage')
plt.xticks(rotation=45)

# Subplot 2: Internet Connection vs. Telehealth Care Satisfaction
plt.subplot(2, 2, 2)
sns.countplot(x='InternetConnection', hue='Telehealth_GoodCare', data=selected_data)
plt.title('Internet Connection vs. Telehealth Care Satisfaction')
plt.xticks(rotation=45)

# Correlation plot for selected variables
plt.subplot(2, 1, 2)
# Factorizing categorical columns for correlation
corr_data = selected_data.copy()
for col in corr_data.select_dtypes(include=['object']).columns:
    corr_data[col], _ = pd.factorize(corr_data[col])
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Encode categorical variables
encoder = LabelEncoder()
selected_data_encoded = selected_data.copy()
for col in selected_data.columns:
    if selected_data_encoded[col].dtype == 'object':
        selected_data_encoded[col] = encoder.fit_transform(selected_data[col])

# Check for any remaining missing values
print("Missing Values After Encoding:\n", selected_data_encoded.isnull().sum())

# Define the model variables
X_columns = ['InternetConnection', 'AgeGrpA', 'Education']  # Update list as per your model's needs
X = pd.get_dummies(selected_data_encoded[X_columns], drop_first=True)
y = pd.Categorical(selected_data_encoded['Telehealth_GoodCare'], ordered=True).codes

# Build the ordinal logistic regression model
model1 = OrderedModel(y, X, distr='logit')
results1 = model1.fit(method='bfgs')
print(results1.summary())

# Dichotomize 'LifeHasMeaning' and run logistic regression
y_life_binary = (selected_data_encoded['LifeHasMeaning'] >= selected_data_encoded['LifeHasMeaning'].max() - 1).astype(int)
X_simplified = X[['InternetConnection_Somewhat_satisfied', 'InternetConnection_Not_satisfied']]  # Adjust column names if necessary

# Run logistic regression
logistic_model = sm.Logit(y_life_binary, sm.add_constant(X_simplified))
results2 = logistic_model.fit()
print(results2.summary())


