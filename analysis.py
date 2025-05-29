# analysis.py
# DSA 210 Intro to Data Science – Automated EDA + Tests + Tuning
# Research Question: GDP per Capita vs Women's Tertiary Education

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 0) Prepare output folder
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)

# 1) Load & inspect each OECD CSV
paths = {
    'education': 'education.csv',
    'gdp': 'gdp.csv',
    'labour': 'labour.csv',
    'fertility': 'fertility.csv'
}

data = {}
for key, path in paths.items():
    df = pd.read_csv(path, header=1)
    print(f"Loaded {key}: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head(2), "\n")
    data[key] = df

# 2) Filter & rename indicators
def filter_indicator(df, indicator_col, code, value_name):
    df = df[(df['INDICATOR']==code) & (df['FREQUENCY']=='A')]
    year_min, year_max = df['TIME'].min(), df['TIME'].max()
    print(f" - {code}: years {int(year_min)} to {int(year_max)}")
    latest = df['TIME'].max()
    df = df[df['TIME']==latest]
    return df[['LOCATION','Value']].rename(columns={'LOCATION':'Country','Value':value_name})

edu = filter_indicator(data['education'], 'INDICATOR', 'EDUADULT', 'Women_Edu_Pct')
gdp = filter_indicator(data['gdp'],       'INDICATOR', 'GDP',      'GDP_Per_Capita_USD')
lab = filter_indicator(data['labour'],    'INDICATOR', 'LFPR',     'Women_Emp_Pct')
fert= filter_indicator(data['fertility'], 'INDICATOR', 'FERTILITY','Fertility_Rate')
print()

# 3) Merge & clean
df = edu.merge(gdp, on='Country') \
        .merge(lab, on='Country') \
        .merge(fert, on='Country') \
        .dropna().reset_index(drop=True)

print(f"After merging: {df.shape[0]} countries × {df.shape[1]} indicators")
print(df.head(3), "\n")

# Ensure numeric types
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# 4) Save merged data
merged_csv = os.path.join(output_dir, "processed_data.csv")
df.to_csv(merged_csv, index=False)
print(f"Merged data saved to: {merged_csv}\n")

# 5) Plotting functions
def create_histogram(data, column, title, xlabel, fname):
    plt.figure(figsize=(8,5))
    plt.hist(data[column], bins=8, edgecolor='black', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Countries")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def create_top_bar(data, column, title, ylabel, fname, color):
    top = data.nlargest(5, column)
    bottom = data.nsmallest(5, column)
    # Top 5
    print(f"Top 5 by {column}:")
    for i, row in top.iterrows():
        print(f" - {row['Country']}: {row[column]:.2f}")
    print(f"Bottom 5 by {column}:")
    for i, row in bottom.iterrows():
        print(f" - {row['Country']}: {row[column]:.2f}")
    plt.figure(figsize=(10,6))
    plt.bar(top['Country'], top[column], color=color)
    plt.title(title)
    plt.xlabel('Country')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

# 6) Generate original plots
create_histogram(df, 'Women_Edu_Pct',
    "Distribution of Women's Tertiary Education",
    "Percentage", 'education_histogram.png')

create_histogram(df, 'GDP_Per_Capita_USD',
    "Distribution of GDP Per Capita",
    "USD", 'gdp_histogram.png')

plt.figure(figsize=(10,6))
plt.scatter(df['GDP_Per_Capita_USD'], df['Women_Edu_Pct'], color='blue')
for _, r in df.iterrows():
    plt.text(r['GDP_Per_Capita_USD'], r['Women_Edu_Pct'], r['Country'], fontsize=7)
plt.title("GDP Per Capita vs Women's Education")
plt.xlabel("GDP Per Capita (USD)")
plt.ylabel("Women's Education (%)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'gdp_vs_education.png'))
plt.close()

create_top_bar(df, 'Women_Edu_Pct',
    "Top & Bottom 5 Countries by Women's Education",
    "% Education", 'top_education_countries.png', color='purple')

create_top_bar(df, 'GDP_Per_Capita_USD',
    "Top & Bottom 5 Countries by GDP Per Capita",
    "USD", 'top_gdp_countries.png', color='green')

# 2×2 relationship grid
plt.figure(figsize=(12,10))
pairs = [
    ('GDP_Per_Capita_USD','Women_Edu_Pct','GDP vs Education'),
    ('Women_Emp_Pct','Women_Edu_Pct','Employment vs Education'),
    ('Fertility_Rate','Women_Edu_Pct','Fertility vs Education'),
    ('GDP_Per_Capita_USD','Women_Emp_Pct','GDP vs Employment')
]
for i, (x,y,ttl) in enumerate(pairs,1):
    plt.subplot(2,2,i)
    plt.scatter(df[x], df[y])
    plt.title(ttl)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'relationship_plots.png'))
plt.close()

# 7) EDA & Hypothesis Tests
print("\n=== Correlation Matrix ===")
corr = df.iloc[:,1:].corr()
print(corr, "\n")

# Test 1: High vs Low GDP groups
median_gdp = df['GDP_Per_Capita_USD'].median()
high = df[df['GDP_Per_Capita_USD'] >= median_gdp]['Women_Edu_Pct']
low  = df[df['GDP_Per_Capita_USD'] <  median_gdp]['Women_Edu_Pct']
t1, p1 = stats.ttest_ind(high, low, equal_var=False)
print("Comparison of Women's Education between high- and low-GDP countries:")
print(f" - High-GDP (n={len(high)}): mean={high.mean():.2f}")
print(f" - Low-GDP  (n={len(low)}):  mean={low.mean():.2f}")
print(f" - t-statistic={t1:.3f}, p-value={p1:.4f}")
print(" - Difference is statistically " + ("significant (p<0.05)" if p1<0.05 else "not significant (p>=0.05)"))

# Test 2: Pearson correlation significance
r, p2 = stats.pearsonr(df['GDP_Per_Capita_USD'], df['Women_Edu_Pct'])
print("\nPearson correlation between GDP and Women's Education:")
print(f" - r = {r:.3f}, p-value = {p2:.4f}")
print(" - Correlation is " + ("statistically significant (p<0.05)" if p2<0.05 else "not significant (p>=0.05)"))

# 8) Random Forest regression with hyperparameter tuning
X = df[['GDP_Per_Capita_USD','Women_Emp_Pct','Fertility_Rate']]
y = df['Women_Edu_Pct']

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('rf',    RandomForestRegressor(random_state=42))
])
param_grid = {
    'rf__n_estimators': [50,100],
    'rf__max_depth':    [None,5],
    'rf__min_samples_split': [2,5]
}
cv = KFold(n_splits=4, shuffle=True, random_state=1)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', n_jobs=-1)
grid.fit(X, y)

print("\n=== Random Forest CV Results ===")
print(f"Best params: {grid.best_params_}")
print(f"Best CV R² : {grid.best_score_:.3f}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
best = grid.best_estimator_
y_pred = best.predict(X_te)
print(f"Test set R² = {r2_score(y_te,y_pred):.3f}")
print(f"Test set MSE= {mean_squared_error(y_te,y_pred):.2f}")

print(f"\nAll outputs (plots + processed_data.csv) are in '{output_dir}'")

# ML PART
print("MACHINE LEARNING ANALYSIS")

print("\n Now predicting Women's Education Percentage using different ML models")

# For visualization purposes, a function to plot actual vs predicted values
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'Actual vs Predicted Values ({model_name})')
    plt.xlabel('Actual Women\'s Education (%)')
    plt.ylabel('Predicted Women\'s Education (%)')
    plt.grid(True, alpha=0.3)
    
    # Add country labels to points
    for i, country in enumerate(df.iloc[y_true.index]['Country']):
        plt.annotate(country, (y_true.iloc[i], y_pred[i]), fontsize=8)
    
    plt.savefig(os.path.join(output_dir, f'ml_{model_name.lower()}_predictions.png'))
    plt.close()

# Prepare data
X = df[['GDP_Per_Capita_USD', 'Women_Emp_Pct', 'Fertility_Rate']]
y = df['Women_Edu_Pct']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nSplit the data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")

# 1. Linear Regression - simplest model
print("\n1. Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Calculate metrics
lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
print(f"   - R² Score: {lr_r2:.3f} (higher is better, 1.0 is perfect)")
print(f"   - Mean Squared Error: {lr_mse:.2f} (lower is better)")

# Print coefficients for interpretation
print("\n   Linear Regression Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"   - {feature}: {coef:.4f}")
    
# Explain what these coefficients mean
print("\n   Interpretation of coefficients:")
print("   - These values show how much Women's Education changes when each feature increases by 1 unit")
print("   - Positive coefficients mean they increase Women's Education")
print("   - Negative coefficients mean they decrease Women's Education")

# Create visualization
plot_predictions(y_test, lr_pred, "Linear Regression")

# 2. Decision Tree - a simple tree-based model
print("\n2. Training Decision Tree model...")
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)  # Limiting depth for simplicity
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Calculate metrics
dt_r2 = r2_score(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
print(f"   - R² Score: {dt_r2:.3f}")
print(f"   - Mean Squared Error: {dt_mse:.2f}")

# Feature importance
print("\n   Decision Tree Feature Importance:")
for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f"   - {feature}: {importance:.4f}")

# Create visualization
plot_predictions(y_test, dt_pred, "Decision Tree")

# 3. Support Vector Regression - a more advanced model
print("\n3. Training Support Vector Regression model...")
# Using simpler parameters for an intro course
svr_model = SVR(kernel='linear')  
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

# Calculate metrics
svr_r2 = r2_score(y_test, svr_pred)
svr_mse = mean_squared_error(y_test, svr_pred)
print(f"   - R² Score: {svr_r2:.3f}")
print(f"   - Mean Squared Error: {svr_mse:.2f}")

# Create visualization
plot_predictions(y_test, svr_pred, "SVR")

# 4. Comparing all models
print("\n4. Comparison of all models:")
models = {
    "Linear Regression": (lr_r2, lr_mse),
    "Decision Tree": (dt_r2, dt_mse),
    "SVR": (svr_r2, svr_mse),
    "Random Forest": (r2_score(y_te, y_pred), mean_squared_error(y_te, y_pred))
}

# Sort models by R² score (higher is better)
sorted_models = sorted(models.items(), key=lambda x: x[1][0], reverse=True)

print("\n   Model Rankings (by R² Score):")
for i, (model_name, (r2, mse)) in enumerate(sorted_models, 1):
    print(f"   {i}. {model_name}: R²={r2:.3f}, MSE={mse:.2f}")

# Create a bar chart to compare model performance
plt.figure(figsize=(10, 6))
model_names = [model[0] for model in sorted_models]
r2_scores = [model[1][0] for model in sorted_models]

plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Comparison - R² Score (higher is better)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)  # R² score is typically between 0 and 1
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ml_model_comparison.png'))
plt.close()

# 5. What-if Analysis: Predicting for hypothetical countries
print("\n5. What-if Analysis: Predicting Women's Education for hypothetical countries")

# Define hypothetical countries with different characteristics
hypothetical_countries = [
    {'name': 'High GDP, High Employment, Low Fertility', 
     'GDP_Per_Capita_USD': 70000, 'Women_Emp_Pct': 80, 'Fertility_Rate': 1.5},
    {'name': 'High GDP, Low Employment, High Fertility', 
     'GDP_Per_Capita_USD': 70000, 'Women_Emp_Pct': 40, 'Fertility_Rate': 4.0},
    {'name': 'Low GDP, High Employment, Low Fertility', 
     'GDP_Per_Capita_USD': 10000, 'Women_Emp_Pct': 80, 'Fertility_Rate': 1.5},
    {'name': 'Low GDP, Low Employment, High Fertility', 
     'GDP_Per_Capita_USD': 10000, 'Women_Emp_Pct': 40, 'Fertility_Rate': 4.0}
]

# Create DataFrame for prediction
hypo_df = pd.DataFrame(hypothetical_countries)
hypo_features = hypo_df[['GDP_Per_Capita_USD', 'Women_Emp_Pct', 'Fertility_Rate']]

# Using our best model to make predictions
best_model_name = sorted_models[0][0]
if best_model_name == "Linear Regression":
    best_model = lr_model
elif best_model_name == "Decision Tree":
    best_model = dt_model
elif best_model_name == "SVR":
    best_model = svr_model
else:
    best_model = best  # Random Forest from earlier

hypo_predictions = best_model.predict(hypo_features)

# Display predictions
print(f"\n   Using our best model ({best_model_name}) to predict Women's Education:")
for i, country in enumerate(hypothetical_countries):
    print(f"   - {country['name']}:")
    print(f"     GDP: ${country['GDP_Per_Capita_USD']:,}, Women's Employment: {country['Women_Emp_Pct']}%, Fertility Rate: {country['Fertility_Rate']}")
    print(f"     Predicted Women's Education: {hypo_predictions[i]:.2f}%")

# Create bar chart for hypothetical countries
plt.figure(figsize=(12, 6))
plt.bar(range(len(hypothetical_countries)), hypo_predictions, color='purple')
plt.xticks(range(len(hypothetical_countries)), [c['name'] for c in hypothetical_countries], rotation=45, ha='right')
plt.title("Predicted Women's Education for Hypothetical Countries")
plt.ylabel("Predicted Women's Education (%)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ml_hypothetical_predictions.png'))
plt.close()

# 6. Investigating the relationship between GDP and Education in more detail
print("\n6. Investigating relationship between GDP and Women's Education")

# Create a range of GDP values
gdp_range = np.linspace(df['GDP_Per_Capita_USD'].min(), df['GDP_Per_Capita_USD'].max(), 100)

# Create predictions with other variables at their mean
mean_emp = X['Women_Emp_Pct'].mean()
mean_fertility = X['Fertility_Rate'].mean()

# Create DataFrame for prediction
gdp_test = pd.DataFrame({
    'GDP_Per_Capita_USD': gdp_range,
    'Women_Emp_Pct': [mean_emp] * 100,
    'Fertility_Rate': [mean_fertility] * 100
})

# Make predictions
gdp_predictions = best_model.predict(gdp_test)

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['GDP_Per_Capita_USD'], df['Women_Edu_Pct'], alpha=0.6, label='Actual Data')
plt.plot(gdp_range, gdp_predictions, 'r-', linewidth=2, label=f'Predicted by {best_model_name}')
plt.title("GDP per Capita vs Women's Tertiary Education")
plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Women's Education (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(output_dir, 'ml_gdp_education_relationship.png'))
plt.close()

# Calculate how much Women's Education increases per $10,000 GDP increase
gdp_diff = 10000
index_low = 0
index_high = next(i for i, gdp in enumerate(gdp_range) if gdp >= gdp_range[0] + gdp_diff)
edu_diff = gdp_predictions[index_high] - gdp_predictions[index_low]

print(f"\n   Effect of GDP on Women's Education:")
print(f"   - For a ${gdp_diff:,} increase in GDP per capita, Women's Education changes by approximately {edu_diff:.2f}%")
print(f"   - This is based on predictions from our {best_model_name} model")

print("\nMachine Learning analysis completed!")
print(f"Additional visualizations have been saved to the '{output_dir}' folder")