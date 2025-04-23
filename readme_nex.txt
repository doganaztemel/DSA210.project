# GDP and Women's Tertiary Education in OECD Countries

## Project Overview
This project explores the relationship between GDP per capita and women's tertiary education attainment rates across OECD countries. The analysis focuses on basic data exploration and visualization, suitable for an introductory data science course.

## Data Sources
The analysis uses four datasets from the OECD database:
- **education.csv**: Contains women's tertiary education percentages
- **gdp.csv**: Contains GDP per capita data
- **labour.csv**: Contains women's employment statistics
- **fertility.csv**: Contains fertility rate information

## Project Structure
The analysis follows these steps:
1. **Data Loading**: Reading CSV files from the OECD database
2. **Data Preparation**: Filtering for relevant indicators and the most recent year
3. **Data Merging**: Combining datasets based on country
4. **Data Visualization**: Creating histograms, scatter plots, and bar charts

## Visualizations Created
The script generates several visualizations in the `output_files` folder:
1. **Histograms** showing the distribution of:
   - Women's tertiary education percentages
   - GDP per capita values

2. **Scatter Plot** showing the relationship between:
   - GDP per capita and women's education levels

3. **Bar Charts** showing:
   - Top 10 countries by women's tertiary education
   - Top 10 countries by GDP per capita

4. **Relationship Overview**: Multi-panel scatter plots showing relationships between:
   - GDP vs Women's Education
   - Women's Employment vs Education
   - Fertility Rate vs Education
   - GDP vs Women's Employment

## How to Run the Code
1. Place the CSV files in the same directory as the script
2. Install required packages: `pip install pandas matplotlib`
3. Run the script: `python analysis.py`
4. View the results in the `output_files` folder

## Key Findings
This analysis helps answer questions about:
- How women's tertiary education levels are distributed across OECD countries
- Which countries have the highest and lowest values for both variables
- Whether there appears to be a relationship between GDP and women's education levels
- How other factors like employment and fertility relate to women's education

## Next Steps
For future analysis, we could:
- Explore data over multiple years to identify trends
- Include additional socioeconomic variables
- Examine regional patterns and group countries by geographic location

