import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set(style='whitegrid')

literacy_df = pd.read_csv('cross-country-literacy-rates.csv')
median_age_df = pd.read_csv('Median age.csv')

literacy_df.rename(columns={'Entity': 'Country', 'Literacy rate': 'LiteracyRate'}, inplace=True)
median_age_df.rename(columns={'name': 'Country', ' years': 'MedianAge'}, inplace=True)

merged_df = pd.merge(literacy_df, median_age_df, on='Country', how='inner')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='LiteracyRate', y='MedianAge', hue='region', palette='deep', s=100)
X = merged_df[['LiteracyRate']]
y = merged_df['MedianAge']
model = LinearRegression()
model.fit(X, y)
x_vals = np.linspace(merged_df['LiteracyRate'].min(), merged_df['LiteracyRate'].max(), 100)
y_vals = model.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_vals, color='red', label='Overall Regression')
plt.xlabel('Literacy Rate (%)')
plt.ylabel('Median Age (years)')
plt.title('Literacy Rate vs. Median Age by Region')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('scatterplot_literacy_vs_median_age.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 8))
merged_sorted = merged_df.sort_values(by='LiteracyRate', ascending=False)
sns.barplot(x='LiteracyRate', y='Country', hue='region', data=merged_sorted, dodge=False, palette="viridis")
plt.xlabel('Literacy Rate (%)')
plt.ylabel('Country')
plt.title('Literacy Rates by Country and Region (Sorted)')
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('barplot_literacy_by_country_region.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='LiteracyRate', data=merged_df, palette='pastel')
plt.xlabel('Region')
plt.ylabel('Literacy Rate (%)')
plt.title('Distribution of Literacy Rates by Region')
plt.tight_layout()
plt.savefig('boxplot_literacy_by_region.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_df, x='LiteracyRate', bins=20, kde=True, hue='region', multiple='stack', palette='deep')
plt.xlabel('Literacy Rate (%)')
plt.title('Histogram of Literacy Rates by Region')
plt.tight_layout()
plt.savefig('histogram_literacy_by_region.png', dpi=300)
plt.show()
