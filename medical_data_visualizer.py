# medical_data_visualizer.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['BMI'] = df['weight'] / (df['height'] / 100)**2
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Clean the data
df = df[(df['ap_lo'] <= df['ap_hi'])
        & (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'
  df_cat = pd.melt(df,
                   id_vars=['cardio'],
                   value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                     'overweight'
                   ])

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = df_cat.groupby(['cardio', 'variable',
                           'value']).size().reset_index(name='total')

  # Draw the catplot with 'sns.catplot()'
  fig = sns.catplot(x='variable',
                    y='total',
                    hue='value',
                    col='cardio',
                    kind='bar',
                    data=df_cat)

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Calculate the correlation matrix
  corr = df.corr()

  # Drop the 'BMI' column to exclude it from the heatmap
  corr.drop(columns=['BMI'], inplace=True)

  # Generate a mask for the upper triangle
  mask = np.triu(corr)

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(10, 8))

  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',  # Adjusted format to display two decimal places
    mask=mask,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax)

  # Set the labels for the plot
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

  # Do not modify the next two lines
  fig.tight_layout()
  fig.savefig('heatmap.png')
  return fig
