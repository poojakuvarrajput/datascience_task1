import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv", skiprows=4)

# Select needed columns
df = data[['Country Name', '2022']].dropna()

# Convert to numeric
df['2022'] = pd.to_numeric(df['2022'], errors='coerce')
df = df.dropna()

# Get top 10 countries
top10 = df.sort_values(by='2022', ascending=False).head(10)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(top10['Country Name'], top10['2022'])

plt.xticks(rotation=45)
plt.title("Top 10 Most Populated Countries (2022)")
plt.xlabel("Country")
plt.ylabel("Population")

plt.tight_layout()
plt.show()