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
                df[col] = pd.to_datetime(df[col], errors = 'coerce', format='%d-%m-%Y')
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
#print(all_payments)
#As payment category is clean, no mapping is required.

#.....Exploratory Data Analysis Questions......

def EDA_plots(dfs):

    #Q1 - Yearly Sales Trend Analysis
    #Create a comprehensive revenue trend analysis showing yearly revenue growth from 
    # 2015-2025. Include percentage growth rates, trend lines, 
    # and highlight key growth periods with annotations.

    revenue = {}
    for year in sorted(dfs):
        df = dfs[year]
        revenue[year] = df['final_amount_inr'].sum()

    #print(revenue)

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
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce', format='%d-%m-%Y')
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

    #Q5 - Perform category-wise performance analysis. 
    # Create treemaps, bar charts, and pie charts showing revenue contribution, growth rates, 
    # and market share for each product category.

    #As electronics is the only category present in the data,
    # we will create line chart of revenue from 2015-2025.


    # # Plot revenue and trend line
    plt.figure(figsize=(10,6))
    sns.lineplot(data=revenue_df, x='Year', y='Total Revenue', marker='o', label='Revenue')

    # # Annotate key growth periods
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

    plt.title('Yearly Revenue for Electronics (2015-2025) with Growth Rates')
    plt.xlabel('Year')
    plt.ylabel('Total Revenue (INR)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Q6 - Analyze Prime membership impact on customer behavior. 
    # Compare average order values, order frequency, and 
    # category preferences between Prime and non-Prime customers 
    # using multiple visualization types.

    #convert is_prime_member to boolean as it's not done earlier

    def is_prime_boolean(dfs):
        true_values = {'true', 'yes', '1', 't', 'y'}
        false_values = {'false', 'no', '0', 'f', 'n'}
        col_variants = ('is_prime_member', 'Is Prime Member', 'prime_member', 'Prime Member', 'primeMember')
        for year in sorted(dfs):
            df = dfs[year]
            for col in col_variants:
                if col in df.columns:
                    s = df[col].str.lower().str.strip()
                    if s.isin(true_values | false_values).all():
                        df[col] = s.map(lambda x: True if x in true_values else False)
                        #print(f"Standardized boolean column: {col} in year {year}")

    is_prime_boolean(dfs)

    prime_analysis = []
    for year in sorted(dfs):
        df = dfs[year]
        if 'is_prime_member' in df.columns and 'final_amount_inr' in df.columns and 'customer_id' in df.columns:
            prime_summary = df.groupby('is_prime_member').agg({
                'final_amount_inr': 'mean',
                'customer_id': 'count'
            }).rename(columns={
                'final_amount_inr': 'Avg Order Value',
                'customer_id': 'Order Frequency'
            }).reset_index()
            prime_summary['Year'] = year
            prime_analysis.append(prime_summary)
    prime_analysis_df = pd.concat(prime_analysis, ignore_index=True)
    prime_analysis_df['Prime Status'] = prime_analysis_df['is_prime_member'].map({
        False: 'Non-Prime',
        True: 'Prime'
    })


    #side by side plots for avg order value as line and order frequency as pie chart

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    my_palette = {
        'Non-Prime': "blue",
        'Prime': "red"
    }

    # --- Plot line chart on the first subplot (axes[0]) ---
    sns.lineplot(data=prime_analysis_df, x='Year', y='Avg Order Value', hue='Prime Status', ax = axes[0], marker='o', 
                palette = my_palette)
    axes[0].set_title('Average Order Value: Prime vs Non-Prime Customers (2015-2025)')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Average Order Value (INR)')
    axes[0].legend(title='Prime Members analysis')

    # --- Plot bar chart on the second subplot (axes[1]) ---
    sns.barplot(data=prime_analysis_df, x='Year', y='Order Frequency', hue='Prime Status', ax=axes[1],
                palette = my_palette)
    axes[1].set_title('Total Order Frequency: Prime vs Non-Prime')
    axes[1].set_xlabel('Prime Member Status')
    axes[1].set_ylabel('Order Frequency')
    axes[1].legend(title='Prime Members analysis')
                                                        
    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout()
    plt.show()

    #Q7 - Create geographic analysis of sales performance across Indian cities and states. 
    # Build choropleth maps and bar charts showing revenue density and growth patterns by tier

    revenue_geo_analysis = []
    for year in sorted(dfs):
        df = dfs[year]
        if 'customer_city' in df.columns and 'final_amount_inr' in df.columns:
            geo_summary = df.groupby(['customer_city','customer_state','customer_tier'])['final_amount_inr'].sum().reset_index()
            geo_summary['Year'] = year
            revenue_geo_analysis.append(geo_summary)
    revenue_geo_df = pd.concat(revenue_geo_analysis, ignore_index=True)
    #aggregate revenue by city across years
    revenue_geo_df.rename(columns={'final_amount_inr': 'Revenue'}, inplace=True)

    revenue_state = revenue_geo_df.groupby('customer_state')['Revenue'].sum().reset_index()

    #print(revenue_geo_df.head)
    import plotly.express as pt
    from geopy.geocoders import Nominatim

    # Create a geocoder instance with a custom user_agent
    geolocator = Nominatim(user_agent="my_map_plot_app")

    # Wrap the geocoder with a RateLimiter to enforce a minimum delay of 1 second
    geocode_rate_limited = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Function to get coordinates with error handling
    def get_coordinates(state):
        try:
            location = geocode_rate_limited(f"{state}, India", timeout=10) # Set a longer timeout
            if location:
                return location.latitude, location.longitude
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Error geocoding {state}: {e}. Retrying with delay...")
            # RateLimiter has built-in retry logic, but this adds more resilience
            return None, None
        return None, None

    # Apply the function to the DataFrame
    revenue_geo_df[['latitude', 'longitude']] = revenue_geo_df['customer_state'].apply(
        lambda x: pd.Series(get_coordinates(x))
    )

    # Filter out any states that could not be geocoded
    revenue_geo_df.dropna(subset=['latitude', 'longitude'], inplace=True)


    # Scatter geo plot for revenue by state
    fig = pt.scatter_geo(revenue_geo_df,
                        lat ="latitude",
                        lon ="longitude",
                        color="customer_state",
                        size="Revenue",
                        hover_name="customer_state",
                        projection="natural earth",
                        title="Revenue (in millions) across Indian Cities",
                        size_max=40,
                        template="plotly_dark"
                        )

    # Manually focus the map on India
    fig.update_geos(
        scope='asia',
        center={'lon': 78.9629, 'lat': 20.5937},
        projection_scale=2 # this value determines the zoom
    )
    fig.show()

    #bar chart for growth patterns by tier
    tier_summary = revenue_geo_df.groupby('customer_tier')['Revenue'].sum().reset_index()
    sns.barplot(data=tier_summary, x='customer_tier', y='Revenue')
    plt.title('Revenue by Customer Tier')
    plt.xlabel('Customer Tier')
    plt.ylabel('Total Revenue in millions (INR)')
    plt.tight_layout()
    plt.show()



    #Q8 - Study festival sales impact using before/during/after analysis. 
    # Visualize revenue spikes during Diwali, Prime Day, and other 
    # festivals with detailed time series analysis.

    #Dfs has columns 'is_festival_period' and 'festival_name' 

    #sort out festivals as per time, plot time series of revenue
    #with legends for each year

    def plot_festival_impact(dfs):
        festival_data = []
        for year in sorted(dfs):
            df = dfs[year]
            fest_df = df[df['is_festival_sale'] == 'True']
            fest_summary = fest_df.groupby('festival_name')['final_amount_inr'].sum().reset_index()
            fest_summary['Year'] = year
            festival_data.append(fest_summary)
                
        festival_df = pd.concat(festival_data, ignore_index=True)
        #print(festival_df)
        #Time series plot for festival impact
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=festival_df, x='Year', y='final_amount_inr', hue='festival_name', marker='o')
        plt.title('Festival Sales Impact (2015-2025)')
        plt.xlabel('Year')
        plt.ylabel('Total Revenue during Festival Sales (INR)')
        plt.legend(title='Festival Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    plot_festival_impact(dfs)

    #Q9 - Analyze customer age group behavior and preferences. 
    # Create demographic analysis with category preferences, 
    # spending patterns, and shopping frequency across different 
    # age segments.

    def customer_age_analysis(dfs):
        age_data = []
        for year in sorted(dfs):
            df = dfs[year]
            if 'customer_age_group' in df.columns and 'final_amount_inr' in df.columns:
                age_summary = df.pivot_table(index=['customer_age_group', 'category', 'customer_spending_tier'],
                    values=['quantity', 'final_amount_inr'],
                    aggfunc={'quantity': 'sum', 'final_amount_inr': 'mean'}).reset_index()
                age_data.append(age_summary)
                age_data_df = pd.concat(age_data, ignore_index=True)
                age_data_df = age_data_df.groupby('customer_age_group').agg({
                    'quantity': 'sum',
                    'final_amount_inr': 'mean'
                }).rename(columns={
                    'quantity': 'Total Quantity Purchased',
                    'final_amount_inr': 'Avg Spending'
                }).reset_index()
        # Plotting
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
        sns.barplot(data=age_data_df, x='customer_age_group', y='Total Quantity Purchased', ax=axes[0])
        axes[0].set_title('Total Quantity Purchased by Age Group')
        axes[0].set_xlabel('Customer Age Group')
        axes[0].set_ylabel('Total Quantity Purchased')
        sns.lineplot(data=age_data_df, x='customer_age_group', y='Avg Spending', ax=axes[1])
        axes[1].set_title('Average Spending by Age Group')
        axes[1].set_xlabel('Customer Age Group')
        axes[1].set_ylabel('Average Spending (INR)')
        plt.tight_layout()
        plt.show()
    
    customer_age_analysis(dfs)

#EDA_plots(dfs)

#Question 10
#Build price vs demand analysis using scatter plots and correlation matrices. 
#Analyze how pricing strategies affect sales volumes across different categories
#and customer segments.

