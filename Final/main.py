import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df_age = pd.read_csv('./Median age.csv')
df_age.columns = df_age.columns.str.strip().str.lower()

df_gdp = pd.read_csv('./gdp.csv')
df_gdp.columns = df_gdp.columns.str.strip().str.lower()

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

if 'gdp per capita' in df_gdp.columns:
    gdp_col = 'gdp per capita'
elif 'gdp_per_capita' in df_gdp.columns:
    gdp_col = 'gdp_per_capita'
else:
    raise KeyError("No GDP per Capita column found in the GDP dataset.")

merged_df = pd.merge(df_age, df_gdp, left_on=country_key_age, right_on=country_key_gdp, how='inner')
merged_df = merged_df.dropna(subset=['years', gdp_col])

merged_df[gdp_col] = merged_df[gdp_col].replace({'\$': '', ',': ''}, regex=True).astype(float)
merged_df['years'] = merged_df['years'].astype(float)

X = merged_df['years'].values.reshape(-1, 1)
y = merged_df[gdp_col].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(merged_df['years'], merged_df[gdp_col], alpha=0.7, label="Data points", edgecolor='black')
plt.plot(merged_df['years'], y_pred, color='red', linewidth=2, label="Regression line")
plt.xlabel("Median Age (Years)")
plt.ylabel("GDP per Capita")
plt.title("Relationship between Median Age and GDP per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('gdp_vs_median_age.png')
plt.show()
