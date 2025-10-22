import os
import pandas as pd
from pathlib import Path
import re
import difflib
import numpy as np

def load_yearly_dfs(start=2015, end=2025, data_dir=None):
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
    dfs = {}
    for year in range(start, end + 1):
        path = data_dir / f"amazon_india_{year}.csv"
        if path.exists():
            dfs[year] = pd.read_csv(path)
            print(f"Loaded data for year {year}")
        else:
            print(f"Warning: file not found: {path}")
    return dfs

dfs = load_yearly_dfs()

#print sample data for 2015
#print(dfs[2015])

#........Data Cleaning Questions.........

#Q1 - Date cleaning for order date column

def print_order_dates(dfs, n=5):
    col_variants = ('order_date', 'order date', 'Order Date', 'OrderDate', 'orderDate', 'Order_Date')
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                print(f"--- {year} ({col}) ---")
                df[col] = pd.to_datetime(df[col], errors = 'coerce').dt.strftime('%Y-%m-%d')
                print(df[col].head())
                break
        else:
            print(f"--- {year} --- order_date column not found")

# print sorted order dates for each year
#print_order_dates(dfs, n=5)

#Q2 - Date cleaning for original_price_inr column

def print_original_prices(dfs, n=5):
    col_variants = ('original_price_inr', 'Original Price INR', 'originalPriceINR', 'Original_Price_INR')
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                print(f"--- {year} ({col}) ---")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(df[col].head())
                break
        else:
            print(f"--- {year} --- original_price_inr column not found")

# print sorted prices for each year
#print_original_prices(dfs, n=5)

#Q3 - Standardize all customer ratings to numeric scale 1.0-5.0 
# and replace missing values with 0

def cleaned_ratings(dfs):
    
    col_variants = (
        'customer_rating','customer rating','Customer Rating','customerRating',
        'rating','Rating','Customer_Rating','avg_rating','rating_value'
    )
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                series = df[col].dropna()
                df[col] = pd.to_numeric(series, errors='coerce')
                df[col] = df[col].fillna(0)
                print(df[col].head())
                break
        else:
            print(f"--- {year} --- customer rating column not found")

# print cleaned ratings for each year
#cleaned_ratings(dfs)

#Q4 - Standardize all city names and handle geographical variations
#get all unique names from all dataframes and accordingly create a mapping
#then apply the mapping to standardize city names

# apply mapping to all dataframes - generalized function

col_variants = ()

def apply_map(dfs, mapping, col_variants):
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                s = df[col].fillna('').str.lower().str.strip()
                s = s.str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
                s = s.str.replace(r'\s+', ' ', regex=True)     # normalize whitespace
                df[col] = df[col].replace(mapping)
                break


def unique_city_names(dfs):
    col_variants = ('city','City','customer_city','Customer City','customerCity')
    cities = {}
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                df[col] = df[col].fillna('').str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)     # normalize whitespace
                cities.update({name: True for name in df[col].unique() if name})
                break
        else:
            print(f"--- {year} --- city column not found")
    return sorted(cities)

all_cities = unique_city_names(dfs)
#print(all_cities)

city_map = {
    'bombay': 'mumbai', 'mumba': 'mumbai', 'mum': 'mumbai',
    'bangalore': 'bengaluru', 'banglore': 'bengaluru', 'bengalore': 'bengaluru',
    'madras': 'chennai', 'chenai': 'chennai',
    'calcutta': 'kolkata', 'delhi ncr': 'new delhi', 'newdelhi': 'new delhi'
}

city_variants = ('city','City','customer_city','Customer City','customerCity')
apply_map(dfs, city_map, city_variants)
#print(sorted(dfs[2015]['customer_city'].unique()))

#Q5 - Convert all boolean columns to consistent True/False format.

def standardize_boolean_columns(dfs):
    true_values = {'true', 'yes', '1', 't', 'y'}
    false_values = {'false', 'no', '0', 'f', 'n'}
    for year in sorted(dfs):
        df = dfs[year]
        for col in df.select_dtypes(include=['object']).columns:
            s = df[col].str.lower().str.strip()
            if s.isin(true_values | false_values).all():
                df[col] = s.map(lambda x: True if x in true_values else False)
                print(f"Standardized boolean column: {col} in year {year}")

#standardize_boolean_columns(dfs)

#Q6 - Standardize category names across the dataset and ensure consistent naming conventions.
#get all unique category from all dataframes and accordingly create a mapping
#then apply the mapping to standardize category names

def unique_category_names(dfs):
    col_variants = ('category','Category','product_category','Product Category','productCategory')
    categories = {}
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                df[col] = df[col].fillna('').str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)     # normalize whitespace
                categories.update({name: True for name in df[col].unique() if name})
                break
        else:
            print(f"--- {year} --- category column not found")
    return sorted(categories)

all_categories = unique_category_names(dfs)
#print(all_categories)

category_map = {'electronics': 'Electronics', 'electronic': 'Electronics',
                'electronics accessories': 'Electronics', 'electronicss': 'Electronics'}

#col name variants for category in all dataframes
category_variants = ('category','Category','product_category','Product Category','productCategory')

apply_map(dfs, category_map, category_variants)
#print(sorted(dfs[2015]['category'].unique()))

#Q7 - delivery_days column - ensure all values are numeric and represent actual days

import re
import numpy as np

def clean_delivery_days(dfs,
                        col_variants=('delivery_days', 'Delivery Days', 'deliveryDays', 'Delivery_Days'),
                        max_days=30,
                        same_day_value=0,
                        fillna_with=None):

    # Parse and clean delivery-days columns in-place:
    #   - 'Same Day' -> same_day_value
    #   - 'overnight' -> 1
    #   - ranges like '1-2 days' -> rounded mean (int)
    #   - extracts numbers where present
    #   - negative / > max_days -> set to NaN
    #   - optionally fill NaN with fillna_with (e.g. median)

    def _parse_cell(x):
        if x is None:
            return np.nan
        s = str(x).strip().lower()
        if s == '' or s in ('nan', 'none'):
            return np.nan
        
        match True:
            case _ if re.search(r'\bsame\b|\bsame day\b|\bsame-day\b', s):
                return same_day_value
            case _ if 'overnight' in s:
                return 1
            case _:
                nums = re.findall(r'\d+', s)
                if not nums:
                    return np.nan
                nums = list(map(int, nums))
                val = nums[0] if len(nums) == 1 else int(round(sum(nums) / len(nums)))
                if val < 0 or val > max_days:
                    return np.nan
                return val

    cleaned = {}
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                parsed = df[col].apply(_parse_cell)
                if fillna_with is None:
                    fill_val = 0
                if fill_val is not None:
                    parsed = parsed.fillna(fill_val)
                # store as nullable integer dtype
                df[col] = parsed.astype('Int64')
                cleaned[year] = df[col]
                break
        else:
            print(f"--- {year} --- delivery_days column not found")
    return cleaned

clean_delivery_days(dfs)

#Q8 - Duplicate transation IDs - identify and handle duplicate transactions

#Identify duplicate transactions where the same customer, product, date, 
# and amount appear multiple times. 

for year in sorted(dfs):
    df = dfs[year]
    cols = ['customer_id', 'product_id', 'order_date', 'quantity','final_amount_inr']
    #exact_duplicate: all columns identical -> drop duplicates (keep first)
    # ensure quantity is numeric
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    mask = df.duplicated(subset=cols, keep = 'first')
    ## keep bulk order: order_id present and identical across group
    order_col = 'order_id'
    if order_col in df.columns:
        quantity_threshold = 15
        # ensure quantity exists and is numeric
        df['quantity'] = pd.to_numeric(df.get('quantity', 0), errors='coerce').fillna(0)
        # handle possible NaN order_id by including them with dropna=False (pandas >=1.1)
        try:
            order_quant = df.groupby('order_id', dropna=False)['quantity'].sum().reset_index()
        except TypeError:
            order_quant = df.groupby('order_id')['quantity'].sum().reset_index()
        bulk_orders = order_quant['order_id']['quantity'] > quantity_threshold
        bulk_order_ids = set(bulk_orders['order_id'].unique())
        # exclude bulk orders from duplicates
        mask = mask & (~df['order_id'].isin(bulk_order_ids))
        num_duplicates = mask.sum()
        print(num_duplicates)
        if num_duplicates > 0:
            df.drop(index=df[mask].index, inplace=True)
        

#Q9 - Correcting for outlier prices - decimals

import seaborn as sns
import matplotlib.pyplot as plt

#before correction
#sns.boxplot(dfs[2021]['final_amount_inr'])
#plt.show()

def correct_price_outliers(dfs, price_col='final_amount_inr', factor_threshold=10):
    for year in sorted(dfs):
        df = dfs[year]
        if price_col in df.columns:
            median_price = df[price_col].median()
            def _correct_price(x):
                if x >= factor_threshold * median_price:
                    return x / 100
                return x
            df[price_col] = df[price_col].apply(_correct_price)
correct_price_outliers(dfs)

#after correction
#sns.boxplot(dfs[2021]['final_amount_inr'])
#plt.show()

#Q10 - Standardize payment methods
#Standardize payment method categories and create a 
# clean categorical hierarchy.

def unique_payment_names(dfs):
    col_variants = ('payment_method','Payment Method','paymentMethod','Payment_Method')
    payments = {}
    for year in sorted(dfs):
        df = dfs[year]
        for col in col_variants:
            if col in df.columns:
                df[col] = df[col].fillna('').str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)     # normalize whitespace
                payments.update({name: True for name in df[col].unique() if name})
                break
        else:
            print(f"--- {year} --- payment column not found")
    return payments

all_payments = unique_payment_names(dfs)
print(all_payments)
#As payment category is clean, no mapping is required.


#.....Exploratory Data Analysis Questions......

#Q1 - Yearly Sales Trend Analysis
#Create a comprehensive revenue trend analysis showing yearly revenue growth from 
# 2015-2025. Include percentage growth rates, trend lines, 
# and highlight key growth periods with annotations.

revenue = {}
for year in sorted(dfs):
    df = dfs[year]
    revenue[year] = df['final_amount_inr'].sum()

print(revenue)

# Convert revenue dict to DataFrame
revenue_df = pd.DataFrame(list(revenue.items()), columns=['Year', 'Total Revenue'])

# Sort by Year (if needed)
revenue_df = revenue_df.sort_values('Year')

# Calculate percentage growth rates
revenue_df['Pct Growth'] = revenue_df['Total Revenue'].pct_change() * 100

# Plot revenue and trend line
plt.figure(figsize=(10,6))
sns.lineplot(data=revenue_df, x='Year', y='Total Revenue', marker='o', label='Revenue')

# Add trend line (linear regression)
sns.regplot(data=revenue_df, x='Year', y='Total Revenue', scatter=False, color='red', label='Trend Line')

# Annotate key growth periods
max_growth_year = revenue_df.loc[revenue_df['Pct Growth'].idxmax(), 'Year']
max_growth_val = revenue_df.loc[revenue_df['Pct Growth'].idxmax(), 'Total Revenue']
plt.annotate(f'Max Growth: {max_growth_year}', 
             xy=(max_growth_year, max_growth_val), 
             xytext=(max_growth_year, max_growth_val*1.05),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=10, color='green')

min_growth_year = revenue_df.loc[revenue_df['Pct Growth'].idxmin(), 'Year']
min_growth_val = revenue_df.loc[revenue_df['Pct Growth'].idxmin(), 'Total Revenue']
plt.annotate(f'Min Growth: {min_growth_year}', 
             xy=(min_growth_year, min_growth_val), 
             xytext=(min_growth_year, min_growth_val*0.95),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red')

plt.title('Yearly Revenue Trend (2015-2025) with Growth Rates & Trend Line')
plt.xlabel('Year')
plt.ylabel('Total Revenue (INR)')
plt.legend()
plt.tight_layout()
plt.show()

#Q2 - Sales analysis
#Analyze seasonal patterns in sales data. 
#Create monthly sales heatmaps and identify peak selling months.
#Compare seasonal trends across different years and categories.

# Combine all data into a single DataFrame with Year and Month columns
combined_data = []
for year in sorted(dfs):
    df = dfs[year]
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['Year'] = year
        df['Month'] = df['order_date'].dt.month
        combined_data.append(df)            
all_data = pd.concat(combined_data, ignore_index=True)
# Group by Year and Month to get total sales
monthly_sales = all_data.groupby(['Year', 'Month'])['final_amount_inr'].sum()
# set monthly sales in million INR for better visualization
monthly_sales = monthly_sales / 1_000_000 # convert to million INR
# Pivot for heatmap
sales_pivot = monthly_sales.unstack(level=0).fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(sales_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Monthly Sales in million INR (2015-2025)')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()              
plt.show()

#Q3 - Customer Segmentation Analysis

#Build a customer segmentation analysis using RFM 
# (Recency, Frequency, Monetary) methodology. 
# Create scatter plots and segment customers into 
# meaningful groups with actionable insights.

# Calculate RFM metrics
import datetime as dt
latest_date = dt.datetime(2025, 12, 31)
rfm_data = []
for year in sorted(dfs):
    df = dfs[year]
    if 'customer_id' in df.columns and 'order_date' in df.columns and 'final_amount_inr' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        rfm = df.groupby('customer_id').agg({
            'order_date': lambda x: (latest_date - x.max()).days,
            'customer_id': 'count',
            'final_amount_inr': 'sum'
        }).rename(columns={
            'order_date': 'Recency',
            'customer_id': 'Frequency',
            'final_amount_inr': 'Monetary'
        })
        rfm_data.append(rfm)
rfm_df = pd.concat(rfm_data, ignore_index=True) 

##Groups customers into segments based on their RFM scores, 
# such as "Champions" (high R, F, M), 
# "Potential Loyalists" (high R, good spending, average F), 
# or "At Risk" customers (declining R, historically high F/M). 
# These segments can guide targeted marketing strategies.

# Heatmap of RFM
plt.figure(figsize=(10, 6))

sns.heatmap(rfm_df.corr(), annot=True, cmap="coolwarm")
plt.title('RFM Correlation Heatmap')
plt.tight_layout()
plt.show()  

#Q4 - Payment Method Evolution Analysis

#evolution of payment methods from 2015-2025. Show the rise of UPI, decline of COD, and create stacked area charts to demonstrate market share changes over time.
payment_trends = []
for year in sorted(dfs):
    df = dfs[year]
    if 'payment_method' in df.columns and 'final_amount_inr' in df.columns:
        payment_summary = df.groupby('payment_method')['final_amount_inr'].sum().reset_index()
        payment_summary['Year'] = year
        payment_trends.append(payment_summary)
payment_trends_df = pd.concat(payment_trends, ignore_index=True)
payment_pivot = payment_trends_df.pivot(index='Year', columns='payment_method', values='final_amount_inr').fillna(0)
payment_pivot_pct = payment_pivot.div(payment_pivot.sum(axis=1), axis=0) * 100
payment_pivot_pct.plot(kind='area', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title('Payment Method Market Share Evolution (2015-2025)')
plt.xlabel('Year')
plt.ylabel('Market Share (%)')
plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Q5