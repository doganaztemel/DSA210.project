# DSA210.project
# Research Proposal: The Relationship Between GDP per Capita and Women's Tertiary Education Attainment in OECD Countries

## 1. Research Question

- **Is there a significant relationship between GDP per capita and women’s tertiary education attainment rates in OECD countries?**

## 2. Motivation

Understanding the interplay between economic prosperity and educational attainment is crucial for policymakers aiming to promote sustainable development and gender equality. In many OECD countries, women have surpassed men in tertiary education attainment, with **52% of 25-34-year-old women holding tertiary qualifications compared to 39% of men**. Investigating how GDP per capita influences these educational trends can provide insights into the effectiveness of economic policies and investments in education.

## 3. Data Sources

This study utilizes datasets from **OECD.org**, covering education and economic indicators for OECD countries.

- **OECD Education Statistics:**  
  - *Source:* [OECD Education Data](https://data.oecd.org/eduatt/adult-education-level.htm#indicator-chart)  
  - *Description:* Provides data on educational attainment by gender across OECD countries.  
  - *Key Variables:*  
    - Percentage of women aged 25-64 with tertiary education.  
    - Percentage of men aged 25-64 with tertiary education.  

- **OECD Economic Indicators:**  
  - *Source:* [OECD GDP Data](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm)  
  - *Description:* Offers data on economic performance indicators, including GDP per capita.  
  - *Key Variables:*  
    - GDP per capita in USD.  

- **Dataset Files:**  
  - The raw datasets used in this study can be accessed here: [Google Drive Dataset](https://drive.google.com/drive/folders/1SA7Y0LUGDIU6Qoxsva87JDQY1Tp45lXP?usp=drive_link)  

## 4. Methodology

### 4.1 Data Collection and Preparation

- Gather data from OECD databases for the most recent year available.  
- Ensure data consistency by aligning definitions and measurement units.  
- Address missing values through appropriate imputation methods.  

### 4.2 Exploratory Data Analysis (EDA)

- Compute summary statistics (mean, median, standard deviation) for GDP per capita and women’s tertiary education attainment rates.  
- Visualize data distributions using histograms and boxplots.  
- Examine the relationship between GDP per capita and women’s tertiary education attainment using scatter plots.  

### 4.3 Statistical Analysis

#### Correlation Analysis

- Calculate the **Pearson correlation coefficient** to assess the strength and direction of the relationship between GDP per capita and women’s tertiary education attainment.  

#### Regression Analysis

- Perform a **simple linear regression** with women’s tertiary education attainment as the dependent variable and GDP per capita as the independent variable.  
- Evaluate the regression model’s assumptions and fit.  

## 5. Tools Used

This study will use **Python** for data analysis and visualization. The following libraries will be utilized:

- **Pandas:** For data manipulation and analysis.  
- **NumPy:** For numerical computations.  
- **Matplotlib & Seaborn:** For visualizing distributions and relationships.  
- **Scipy & Statsmodels:** For statistical hypothesis testing and regression modeling.  
