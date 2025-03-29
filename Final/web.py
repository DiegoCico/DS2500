import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the median age dataset
df_age = pd.read_csv('./Median age.csv')
df_age.columns = df_age.columns.str.strip().str.lower()

# Load the GDP per capita dataset (make sure the file is available)
df_gdp = pd.read_csv('./gdp.csv')
df_gdp.columns = df_gdp.columns.str.strip().str.lower()

# Determine which column to use for country names:
# We assume df_age has 'name' (or 'slug') and df_gdp has 'country'
if 'name' in df_age.columns:
    country_key_age = 'name'
elif 'slug' in df_age.columns:
    country_key_age = 'slug'
else:
    raise KeyError("No country name column found in the median age dataset (expected 'name' or 'slug').")

if 'country' in df_gdp.columns:
    country_key_gdp = 'country'
else:
    raise KeyError("No country name column found in the GDP dataset (expected 'country').")

# Merge the two datasets on the country name
merged_df = pd.merge(df_age, df_gdp, left_on=country_key_age, right_on=country_key_gdp, how='inner')

# Drop rows where essential columns are missing
merged_df = merged_df.dropna(subset=['years', 'gdp_per_capita'])

# ----------------------------
# Graph: Scatter Plot of GDP per Capita vs. Median Age with Regression Line
X = merged_df['years'].values.reshape(-1, 1)  # Median Age
y = merged_df['gdp_per_capita'].values          # GDP per Capita

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(merged_df['years'], merged_df['gdp_per_capita'], alpha=0.7, label="Data points", edgecolor='black')
plt.plot(merged_df['years'], y_pred, color='red', linewidth=2, label="Regression line")
plt.xlabel("Median Age (Years)")
plt.ylabel("GDP per Capita")
plt.title("Relationship between Median Age and GDP per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('gdp_vs_median_age.png')
plt.show()
