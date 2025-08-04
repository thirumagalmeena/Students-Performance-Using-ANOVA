import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

df = pd.read_csv("data/StudentsPerformance.csv")
df.columns = df.columns.str.lower().str.replace(" ","_")

def oneway_anova(df,group_cols,score_col):
    groups = df[group_cols].unique()
    samples = [df[df[group_cols] == g][score_col] for g in groups]
    f_stat, p_val = f_oneway(*samples)
    print(f"\n=== One-Way ANOVA: {group_cols} -> {score_col} ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Reject H0 - At least one group mean is significantly different.")
    else:
        print("Result: Fail to reject H0 - No significant difference between group means.")

for score in ['math_score', 'reading_score', 'writing_score']:
    oneway_anova(df, group_cols='gender', score_col=score)

# Plot
sns.set(style="whitegrid")

def plot_box(group_col, score_col):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=group_col, y=score_col, palette='Set2')
    plt.title(f"{score_col.replace('_', ' ').title()} by {group_col.title()}")
    plt.xlabel(group_col.title())
    plt.ylabel(score_col.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()

for score in ['math_score', 'reading_score', 'writing_score']:
    plot_box(group_col='gender', score_col=score)