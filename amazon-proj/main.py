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
        mask = df.duplicated(subset=cols, keep = 'first')
        ## keep bulk order: order_id present and identical across group
        pivot_df = df[mask].copy()
        pivot_df = pd.DataFrame(pivot_df) 
        pivot_df = pivot_df.pivot_table(columns=['customer_id','product_id'], values='quantity', aggfunc='sum', fill_value = 0)
        print(pivot_df)



