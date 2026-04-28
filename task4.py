import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "US_Accidents_March23.csv",
    nrows=50000,
    usecols=['Start_Time', 'Weather_Condition', 'Sunrise_Sunset']
)

df['Start_Time'] = pd.to_datetime(df['Start_Time'])

df['Day'] = df['Start_Time'].dt.day_name()
df['Hour'] = df['Start_Time'].dt.hour

sns.countplot(x='Day', data=df)
plt.xticks(rotation=45)
plt.show()

sns.histplot(df['Hour'], bins=24)
plt.show()

sns.countplot(y='Weather_Condition', data=df)
plt.show()

sns.countplot(x='Sunrise_Sunset', data=df)
plt.show()