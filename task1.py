import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic_df.head())

# Get a summary of the dataset
print(titanic_df.info())

# Get basic statistics of the dataset
print(titanic_df.describe())

# Check for missing values
print(titanic_df.isnull().sum())

# Plot histograms for numerical features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
titanic_df[numerical_features].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Plot bar plots for categorical features
categorical_features = ['Sex', 'Pclass', 'Embarked', 'Survived']

for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=titanic_df)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Compute the correlation matrix
correlation_matrix = titanic_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of Age vs Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_df)
plt.title('Age vs Fare colored by Survival')
plt.show()

# Box plots to identify outliers
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=titanic_df[feature])
    plt.title(f'Box plot of {feature}')
    plt.show()
