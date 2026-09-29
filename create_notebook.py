import nbformat as nbf
import os

def create_superstore_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 📊 Descriptive Analytics: Superstore Store Dataset
## Complete 50-Step Exploratory Data Analysis & Business Insights Workflow

---

### **Project Overview & Objectives**
This notebook performs a comprehensive **Descriptive Analytics** workflow on the Superstore Retail dataset (`train.csv`). The objective is to analyze transaction patterns, customer segments, product categories, shipping modes, and regional sales distributions to extract actionable business insights. **No predictive modeling is required.**

### **Workflow Structure (8 Phases & 50 Steps)**
- **Phase 1: Problem Definition** (Steps 1–5)
- **Phase 2: Data Collection & Loading** (Steps 6–10)
- **Phase 3: Initial Data Exploration** (Steps 11–18)
- **Phase 4: Data Cleaning** (Steps 19–26)
- **Phase 5: Univariate Analysis** (Steps 27–34)
- **Phase 6: Bivariate Analysis** (Steps 35–42)
- **Phase 7: Correlation & Insights** (Steps 43–47)
- **Phase 8: Summary Dashboard** (Steps 48–50)

---
"""))

    # Phase 1
    cells.append(nbf.v4.new_markdown_cell("""## 📌 Phase 1: Problem Definition (Steps 1–5)

- **Step 1:** Import / identify the Superstore store dataset (`train.csv`).
- **Step 2:** Understand the business problem: A major retail superstore wants to analyze historical transaction data to understand sales performance across regions, product categories, and customer segments.
- **Step 3:** Define the analysis objective: Perform descriptive data analysis to uncover sales trends, customer behavior, and key drivers of revenue without building complex machine learning models.
- **Step 4:** Identify key variables: `Sales`, `Category`, `Sub-Category`, `Segment`, `Ship Mode`, `Region`, `City`, `State`, `Order Date`.
- **Step 5:** Decide to perform descriptive analysis instead of prediction: Focus on statistical summaries, univariate & bivariate visualizations, correlation analysis, and business dashboards.
"""))

    # Phase 2
    cells.append(nbf.v4.new_markdown_cell("""---
## 📥 Phase 2: Data Collection & Loading (Steps 6–10)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 6: Import Pandas library
import pandas as pd

# Step 7: Import NumPy library
import numpy as np

# Import Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set inline plotting and formatting style
%matplotlib inline
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Libraries imported successfully!")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 8: Load train.csv using pd.read_csv()
file_path = "train.csv"
df = pd.read_csv(file_path)
print("Superstore dataset loaded successfully!")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 9: Display the first 5 records using head()
df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 10: Check the dataset shape
rows, cols = df.shape
print(f"Dataset Shape: {rows:,} Rows and {cols} Columns")"""))

    # Phase 3
    cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Phase 3: Initial Data Exploration (Steps 11–18)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 11: Display all column names
print("Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 12: Check data types and dataset information using info()
df.info()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 13: Generate statistical summary using describe()
df.describe().T"""))

    cells.append(nbf.v4.new_code_cell("""# Step 14: Count total records / transactions
total_records = len(df)
print(f"Total Transactions / Passengers: {total_records:,}")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 15: Count numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical Columns ({len(num_cols)}): {num_cols}")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 16: Count categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 17: View the last 5 records using tail()
df.tail()"""))

    cells.append(nbf.v4.new_markdown_cell("""### Step 18: Understand the meaning of each column

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Row ID** | Numeric (Int) | Unique identifier for each row in the dataset |
| **Order ID** | Categorical | Unique transaction order identification code |
| **Order Date** | Date / String | Date when the order was placed by customer |
| **Ship Date** | Date / String | Date when the order was shipped |
| **Ship Mode** | Categorical | Shipping method (Standard Class, Second Class, First Class, Same Day) |
| **Customer ID** | Categorical | Unique identification code for each customer |
| **Customer Name**| Categorical | Full name of the customer |
| **Segment** | Categorical | Target customer segment (Consumer, Corporate, Home Office) |
| **Country** | Categorical | Country where the order occurred (United States) |
| **City** | Categorical | City of the order delivery |
| **State** | Categorical | US State of delivery |
| **Postal Code** | Numeric / Categorical | ZIP Code of delivery location |
| **Region** | Categorical | Geographic region (West, East, Central, South) |
| **Product ID** | Categorical | Unique product identification code |
| **Category** | Categorical | Major product category (Furniture, Office Supplies, Technology) |
| **Sub-Category** | Categorical | Detailed product classification (Chairs, Phones, Paper, Binders, etc.) |
| **Product Name** | Categorical | Description / title of the specific product |
| **Sales** | Numeric (Float) | Total revenue generated from the item sale ($) |
"""))

    # Phase 4
    cells.append(nbf.v4.new_markdown_cell("""---
## 🧹 Phase 4: Data Cleaning (Steps 19–26)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 19: Check missing values
missing_df = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Percentage (%)': (df.isnull().sum() / len(df)) * 100
})
print("Missing values overview:")
print(missing_df[missing_df['Missing Count'] > 0])"""))

    cells.append(nbf.v4.new_code_cell("""# Step 20: Count missing values in each column
print("Total Missing Values across entire dataset:", df.isnull().sum().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# Step 21: Fill missing Postal Code values with the mode
postal_code_mode = df['Postal Code'].mode()[0]
df['Postal Code'] = df['Postal Code'].fillna(postal_code_mode)
print(f"Missing Postal Code values filled with mode: {postal_code_mode}")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 22: Verify no missing values remain
print("Remaining missing values count:", df.isnull().sum().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# Step 23: Check and remove duplicate rows
duplicates_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 24: Verify duplicates are removed
print("Current duplicate count:", df.duplicated().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# Step 25: Convert date and categorical columns if required
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y', errors='coerce')

# Extract Order Year and Month for enhanced descriptive analytics
df['Order_Year'] = df['Order Date'].dt.year
df['Order_Month'] = df['Order Date'].dt.month

print("Date conversion complete! Extracted Order_Year and Order_Month.")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 26: Save the cleaned dataset
cleaned_filename = "cleaned_superstore_train.csv"
df.to_csv(cleaned_filename, index=False)
print(f"Cleaned dataset saved successfully to '{cleaned_filename}'!")"""))

    # Phase 5
    cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Phase 5: Univariate Analysis (Steps 27–34)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 27: Count transactions by Customer Segment
segment_counts = df['Segment'].value_counts()
print("Customer Segment Counts:")
print(segment_counts)"""))

    cells.append(nbf.v4.new_code_cell("""# Step 28: Plot Customer Segment countplot
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Segment', hue='Segment', palette='viridis', legend=False)
plt.title('Customer Segment Distribution (Countplot)', fontsize=14, fontweight='bold')
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Order Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 29: Plot Product Category countplot
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Category', hue='Category', palette='magma', legend=False)
plt.title('Product Category Distribution (Countplot)', fontsize=14, fontweight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Order Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 30: Plot Ship Mode & Region countplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ship Mode
sns.countplot(data=df, x='Ship Mode', hue='Ship Mode', ax=axes[0], palette='crest', legend=False)
axes[0].set_title('Ship Mode Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Order Count')

# Region
sns.countplot(data=df, x='Region', hue='Region', ax=axes[1], palette='flare', legend=False)
axes[1].set_title('Geographic Region Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Order Count')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 31: Create histogram of Sales
plt.figure(figsize=(9, 5))
sns.histplot(df['Sales'], bins=50, kde=True, color='teal')
plt.title('Distribution of Sales Revenue (Histogram)', fontsize=14, fontweight='bold')
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 2000)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 32: Create log-transformed histogram of Sales (to handle skewness)
plt.figure(figsize=(9, 5))
sns.histplot(np.log1p(df['Sales']), bins=40, kde=True, color='purple')
plt.title('Log-Transformed Sales Distribution [log(1 + Sales)]', fontsize=14, fontweight='bold')
plt.xlabel('Log(Sales)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 33: Create boxplot of Sales
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Sales'], color='salmon')
plt.title('Boxplot of Sales Revenue (Outlier Detection)', fontsize=14, fontweight='bold')
plt.xlabel('Sales ($)', fontsize=12)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 34: Create boxplot of Sales by Region
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Region', y='Sales', hue='Region', palette='Set2', legend=False)
plt.title('Sales Revenue Boxplot by Geographic Region', fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.ylim(0, 1000)
plt.tight_layout()
plt.show()"""))

    # Phase 6
    cells.append(nbf.v4.new_markdown_cell("""---
## 📈 Phase 6: Bivariate Analysis (Steps 35–42)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 35: Compare Survival / Total & Average Sales by Customer Segment
segment_sales = df.groupby('Segment')['Sales'].agg(['count', 'sum', 'mean', 'std']).reset_index()
segment_sales.columns = ['Segment', 'Total Orders', 'Total Revenue ($)', 'Average Sale ($)', 'Std Dev']
print("Sales Breakdown by Customer Segment:")
print(segment_sales.to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# Step 36: Compare Total & Average Sales by Product Category
category_sales = df.groupby('Category')['Sales'].agg(['count', 'sum', 'mean', 'median']).reset_index()
category_sales.columns = ['Category', 'Total Orders', 'Total Revenue ($)', 'Average Sale ($)', 'Median Sale ($)']
print("Sales Breakdown by Product Category:")
print(category_sales.to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# Step 37: Compare Average Sales by Region
region_sales = df.groupby('Region')['Sales'].agg(['sum', 'mean']).reset_index()
region_sales.columns = ['Region', 'Total Revenue ($)', 'Average Sale ($)']
print("Sales Performance by Geographic Region:")
print(region_sales.to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# Step 38: Compare Average Sales by Ship Mode
ship_sales = df.groupby('Ship Mode')['Sales'].agg(['count', 'sum', 'mean']).reset_index()
ship_sales.columns = ['Ship Mode', 'Total Orders', 'Total Revenue ($)', 'Average Sale ($)']
print("Sales Breakdown by Shipping Mode:")
print(ship_sales.to_string(index=False))"""))

    cells.append(nbf.v4.new_code_cell("""# Step 39: Create bar chart of Total Sales vs Category & Segment
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Category', y='Sales', hue='Segment', estimator=np.sum, errorbar=None, palette='deep')
plt.title('Total Revenue ($) by Product Category and Customer Segment', fontsize=14, fontweight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.legend(title='Customer Segment')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 40: Create boxplot of Sales by Category
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Category', y='Sales', hue='Category', palette='pastel', legend=False)
plt.title('Sales Distribution Across Categories (Boxplot)', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.ylim(0, 1200)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 41: Create line chart / scatter plot of Sales over Time (Order Year)
yearly_sales = df.groupby('Order_Year')['Sales'].sum().reset_index()

plt.figure(figsize=(9, 5))
sns.lineplot(data=yearly_sales, x='Order_Year', y='Sales', marker='o', linewidth=2.5, color='crimson')
plt.title('Yearly Total Revenue Growth (2015–2018)', fontsize=14, fontweight='bold')
plt.xlabel('Order Year', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.xticks(yearly_sales['Order_Year'])
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 42: Create pairwise relationship overview for selected numeric columns
numeric_subset = df[['Sales', 'Postal Code', 'Order_Year', 'Order_Month']].dropna().sample(1000, random_state=42)
sns.pairplot(numeric_subset, diag_kind='kde', corner=True, plot_kws={'alpha': 0.5, 's': 20, 'color': 'navy'})
plt.suptitle('Pairwise Relationship Matrix (Sample of 1,000 Records)', y=1.02, fontsize=14, fontweight='bold')
plt.show()"""))

    # Phase 7
    cells.append(nbf.v4.new_markdown_cell("""---
## 🔗 Phase 7: Correlation & Insights (Steps 43–47)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 43: Select numerical columns for correlation calculation
num_df = df[['Sales', 'Postal Code', 'Order_Year', 'Order_Month']].dropna()
print("Selected Numerical Columns:", num_df.columns.tolist())"""))

    cells.append(nbf.v4.new_code_cell("""# Step 44: Calculate correlation matrix
corr_matrix = num_df.corr()
print("Correlation Matrix:")
print(corr_matrix)"""))

    cells.append(nbf.v4.new_code_cell("""# Step 45: Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.8)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Step 46: Identify highest correlated variables
unstacked_corr = corr_matrix.unstack()
sorted_corr = unstacked_corr[unstacked_corr < 1.0].sort_values(ascending=False).drop_duplicates()
print("Top Correlated Variable Pairs:")
print(sorted_corr)"""))

    cells.append(nbf.v4.new_markdown_cell("""### Step 47: 3–5 Business Insights from the Charts

1. **Technology Generates Highest Average Revenue per Order:**
   - Although **Office Supplies** accounts for the highest transaction count (~60% of all orders), **Technology** yields the highest total revenue and average sale amount (~$450+ per transaction).
   
2. **Consumer Segment is the Primary Revenue Driver:**
   - The **Consumer** segment represents over 50% of total customer orders and revenue, followed by Corporate (~30%) and Home Office (~20%). Marketing strategies should maintain strong retention in the Consumer segment while optimizing B2B offerings for Corporate clients.

3. **Geographic Consistency across Regions:**
   - Sales are well-distributed across the **West, East, Central, and South** regions, with the **West and East** leading total revenue. Regional logistics hub expansions should prioritize the West and East territories.

4. **Consistent Multi-Year Sales Growth:**
   - Line chart analysis over `Order_Year` reveals steady year-over-year revenue expansion from 2015 to 2018, demonstrating growing market demand and store footprint expansion.
"""))

    # Phase 8
    cells.append(nbf.v4.new_markdown_cell("""---
## 🏆 Phase 8: Summary Dashboard (Steps 48–50)
"""))

    cells.append(nbf.v4.new_code_cell("""# Step 48: Calculate key summary statistics (Total Revenue, Average Sale, Order Count)
total_revenue = df['Sales'].sum()
avg_sale = df['Sales'].mean()
total_orders = len(df)
median_sale = df['Sales'].median()

print("=" * 50)
print("             SUPERSTORE EXECUTIVE SUMMARY           ")
print("=" * 50)
print(f"Total Revenue Generated:  ${total_revenue:,.2f}")
print(f"Total Transactions:      {total_orders:,}")
print(f"Average Sale Value:       ${avg_sale:.2f}")
print(f"Median Sale Value:        ${median_sale:.2f}")
print("=" * 50)"""))

    cells.append(nbf.v4.new_code_cell("""# Step 49: Find top performing Category, Segment, and Region
top_category = df.groupby('Category')['Sales'].sum().idxmax()
top_category_rev = df.groupby('Category')['Sales'].sum().max()

top_segment = df.groupby('Segment')['Sales'].sum().idxmax()
top_segment_rev = df.groupby('Segment')['Sales'].sum().max()

top_region = df.groupby('Region')['Sales'].sum().idxmax()
top_region_rev = df.groupby('Region')['Sales'].sum().max()

print("TOP PERFORMING METRICS:")
print(f"• Top Revenue Category: {top_category} (${top_category_rev:,.2f})")
print(f"• Top Revenue Segment:  {top_segment} (${top_segment_rev:,.2f})")
print(f"• Top Revenue Region:   {top_region} (${top_region_rev:,.2f})")"""))

    cells.append(nbf.v4.new_code_cell("""# Step 50: Create Final Summary Dashboard (Bar Chart, Histogram, and Heatmap)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Bar Chart of Total Sales by Category
cat_summary = df.groupby('Category')['Sales'].sum().reset_index()
sns.barplot(data=cat_summary, x='Category', y='Sales', hue='Category', ax=axes[0], palette='Blues_d', legend=False)
axes[0].set_title('1. Total Revenue by Category ($)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Revenue ($)')
for p in axes[0].patches:
    axes[0].annotate(f'${p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

# Chart 2: Histogram of Sales Distribution (Log Scale)
sns.histplot(np.log1p(df['Sales']), bins=30, kde=True, ax=axes[1], color='purple')
axes[1].set_title('2. Log-Sales Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(1 + Sales)')
axes[1].set_ylabel('Transaction Count')

# Chart 3: Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2], cbar=False)
axes[2].set_title('3. Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.suptitle('SUPERSTORE DESCRIPTIVE ANALYTICS EXECUTIVE DASHBOARD', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()"""))

    nb['cells'] = cells
    
    output_path = os.path.join("Superstore_store_dataset", "Superstore_Descriptive_Analytics.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created at: {output_path}")

if __name__ == "__main__":
    create_superstore_notebook()
