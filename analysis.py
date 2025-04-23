# DSA 210 Intro to Data Science – Automated EDA + Tests + Tuning
# Research Question: GDP per Capita vs Women’s Tertiary Education

import os
import pandas as pd
import matplotlib.pyplot as plt

# Prepare output folder
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)

# 1) Load each OECD CSV file
def load_csv(path):
    return pd.read_csv(path, header=1)

paths = {
    'education': 'education.csv',
    'gdp': 'gdp.csv',
    'labour': 'labour.csv',
    'fertility': 'fertility.csv'
}

# Dictionary to store our dataframes
data = {key: load_csv(path) for key, path in paths.items()}

# 2) Filter & prepare each dataset
def filter_indicator(df, code):
    df = df[(df['INDICATOR'] == code) & (df['FREQUENCY'] == 'A')]
    latest_year = df['TIME'].max()
    df = df[df['TIME'] == latest_year]
    return df[['LOCATION','Value']].rename(columns={'LOCATION':'Country'})

edu = filter_indicator(data['education'], 'EDUADULT').rename(columns={'Value':'Women_Edu_Pct'})
gdp = filter_indicator(data['gdp'],       'GDP').rename(columns={'Value':'GDP_Per_Capita_USD'})
lab = filter_indicator(data['labour'],    'LFPR').rename(columns={'Value':'Women_Emp_Pct'})
fert= filter_indicator(data['fertility'], 'FERTILITY').rename(columns={'Value':'Fertility_Rate'})

# 3) Merge datasets
merged_data = (edu
               .merge(gdp, on='Country')
               .merge(lab, on='Country')
               .merge(fert, on='Country')
               .dropna()
               .reset_index(drop=True))

# Ensure numeric types
for col in ['Women_Edu_Pct','GDP_Per_Capita_USD','Women_Emp_Pct','Fertility_Rate']:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
merged_data.dropna(inplace=True)

# 4) Save merged data
df_csv = os.path.join(output_dir, "processed_data.csv")
merged_data.to_csv(df_csv, index=False)
print(f"Merged data saved to: {df_csv}")

# 5) Top 5 and Bottom 5 summaries
metrics = [
    ('Women_Edu_Pct', "Women's Education (%)"),
    ('GDP_Per_Capita_USD', 'GDP Per Capita (USD)'),
    ('Women_Emp_Pct', "Women's Employment (%)"),
    ('Fertility_Rate', 'Fertility Rate')
]
print("\n===== Top & Bottom 5 Countries for Each Metric =====")
for col, label in metrics:
    print(f"\n--- {label} ---")
    top5 = merged_data[['Country', col]].sort_values(col, ascending=False).head(5)
    bottom5 = merged_data[['Country', col]].sort_values(col, ascending=True).head(5)
    print("Top 5:")
    print(top5.to_string(index=False))
    print("Bottom 5:")
    print(bottom5.to_string(index=False))

# 6) Create histograms
def create_histogram(data, column, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(data[column], bins=8, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Countries")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

create_histogram(merged_data, 'Women_Edu_Pct',
                "Distribution of Women's Tertiary Education",
                "Percentage", 'education_histogram.png')

create_histogram(merged_data, 'GDP_Per_Capita_USD',
                "Distribution of GDP Per Capita",
                "GDP Per Capita (USD)", 'gdp_histogram.png')

# 7) Create scatter plot of GDP vs Education
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['GDP_Per_Capita_USD'], merged_data['Women_Edu_Pct'], color='blue')
for _, row in merged_data.iterrows():
    plt.text(row['GDP_Per_Capita_USD'], row['Women_Edu_Pct'], row['Country'], fontsize=8)
plt.title('GDP Per Capita vs Women\'s Education')
plt.xlabel('GDP Per Capita (USD)')
plt.ylabel('Women\'s Education (%)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'gdp_vs_education.png'))
plt.close()

# 8) Create bar charts for top countries

def create_top_bar(data, column, title, ylabel, color, filename):
    top_data = data.sort_values(column, ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(top_data['Country'], top_data[column], color=color)
    plt.title(title)
    plt.xlabel('Country')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

create_top_bar(merged_data, 'Women_Edu_Pct',
              'Top 10 Countries by Women\'s Tertiary Education',
              "Women's Education (%)", 'purple', 'top_education_countries.png')

create_top_bar(merged_data, 'GDP_Per_Capita_USD',
              'Top 10 Countries by GDP Per Capita',
              'GDP Per Capita (USD)', 'green', 'top_gdp_countries.png')

# 9) Create relationship overview plots
plt.figure(figsize=(12, 10))

# GDP vs Education
plt.subplot(2, 2, 1)
plt.scatter(merged_data['GDP_Per_Capita_USD'], merged_data['Women_Edu_Pct'], color='blue')
plt.title('GDP vs Education')
plt.xlabel('GDP Per Capita')
plt.ylabel('Women\'s Education (%)')
plt.grid(True)

# Employment vs Education
plt.subplot(2, 2, 2)
plt.scatter(merged_data['Women_Emp_Pct'], merged_data['Women_Edu_Pct'], color='red')
plt.title('Employment vs Education')
plt.xlabel('Women\'s Employment (%)')
plt.ylabel('Women\'s Education (%)')
plt.grid(True)

# Fertility vs Education
plt.subplot(2, 2, 3)
plt.scatter(merged_data['Fertility_Rate'], merged_data['Women_Edu_Pct'], color='orange')
plt.title('Fertility vs Education')
plt.xlabel('Fertility Rate')
plt.ylabel('Women\'s Education (%)')
plt.grid(True)

# GDP vs Employment
plt.subplot(2, 2, 4)
plt.scatter(merged_data['GDP_Per_Capita_USD'], merged_data['Women_Emp_Pct'], color='green')
plt.title('GDP vs Employment')
plt.xlabel('GDP Per Capita')
plt.ylabel('Women\'s Employment (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'relationship_plots.png'))
plt.close()
print("Plots saved in the output_files directory.")