# Created by: Ms.Aye Theingi Thwin

""" 
Data Profiling 
"""
# Create the retail datasets and perform EDA

import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
stores = pd.read_csv("stores.csv")
products = pd.read_csv("products.csv")
sales = pd.read_csv("sales.csv")
inventory = pd.read_csv("inventory.csv")

# Convert the date columns to datetime format
sales['sale_date'] = pd.to_datetime(sales['sale_date'])

# Calculate revenue for each sale
sales["revenue"] = sales["quantity"] * sales["unit_price"]

# Resonate the record numbers
print("Stores dataset:", stores.shape)
print("Products dataset:", products.shape)
print("Sales dataset:", sales.shape)
print("Inventory dataset:", inventory.shape)

print("\nSales columns:")
print(sales.columns)



""" 
Check data quality and missing values 
"""

print("\nMissing values in Sales dataset:")
print(sales.isnull().sum())

# Check for duplicates in Sales dataset
print("\nDuplicate records in Sales dataset:", sales.duplicated().sum())



""" 
Descriptive statistics for Sales dataset 
"""

print("\nDescriptive statistics for Sales dataset:")
print(sales.describe())

# Overall business KPIs - Power BI Executive Dashboard
total_units = sales["quantity"].sum()
total_revenue = sales["revenue"].sum()
average_transaction_units = sales["quantity"].mean()
average_revenue = sales["revenue"].mean()



"""
Overall Business KPIs - Power BI Executive Dashboard
"""

print("\nOverall Business KPIs:")
print(f"Total Units Sold: {total_units}")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Units per Transaction: {average_transaction_units:.2f}")
print(f"Average Revenue per Transaction: ${average_revenue:,.2f}")



"""
Monthly Sales Trend Analysis
"""

# Group sales by month and calculate total quantity and revenue
monthly_sales = sales.groupby(sales['sale_date'].dt.to_period('M')).agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index()

# Display monthly sales trend
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Visualize monthly revenue
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['sale_date'].astype(str), monthly_sales['revenue'], marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



"""
Identify Top Performing Products and Stores generating the most revenue
"""

# Aggregate sales data to find top performing products
product_sales = (sales.groupby('product_id')
    .agg(
        units_sold = ('quantity', 'sum'),
        revenue = ('revenue', 'sum')
).reset_index().sort_values(by='revenue', ascending=False))

# Join product information to get product names
product_sales = product_sales.merge(
    products[
        [
            'product_id', 
             'product_name',
             'category',
             'cost'
        ]
    ], 
    on='product_id', how='left'
)

# Calculate profit
product_sales['profit'] = product_sales['revenue'] - product_sales['cost'] * product_sales['units_sold']

top_products = (
    product_sales.sort_values(by='revenue', ascending=False)
    .head(10)
)

print(top_products)


"""
Visualize Top Performing Products
"""

plt.figure(figsize=(12, 6))

plt.bar(
    top_products['product_name'],
    top_products['revenue']
)
plt.title('Top 10 Products by Revenue')
plt.xlabel('Product')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



"""
Category Performance
"""

# Aggregate sales data to find category performance
category_sales = (
    product_sales
    .groupby("category")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        profit=("profit", "sum")
    )
    .reset_index()
    .sort_values(
        "revenue",
        ascending=False
    )
)

print(category_sales)

# Visualize Category Performance
plt.figure(figsize=(10, 6))

plt.bar(
    category_sales["category"],
    category_sales["revenue"]
)

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


"""
Regional Analysis
"""

regional_sales = (
    sales.merge(
        stores[
            [
                "store_id", 
                "region",
                "store_size"
            ]
        ],
        on="store_id",
        how="left"
    )
    .groupby("region")
    .agg(
        revenue=("revenue", "sum"),
        units_sold=("quantity", "sum")
    )
    .reset_index()
    .sort_values(
        "revenue",
        ascending=False
    )
)

print(regional_sales)



# Store Performance

store_sales = [
    sales.merge(
        stores,
        on = "store_id",
        how = "left"
    )
    .groupby(
        [
            "store_id",
            "store_name",
            "region",
            "store_size"
        ]
    )
    .agg(
        revenue = ("revenue", "sum"),
        units_sold = ("quantity", "sum")
    )
    .reset_index()
]



# Top Stores

"""
# Convert list to a DataFrame
df = pd.DataFrame(store_sales)

# Work on original code perfectly
print(df["revenue"].sort_values(ascending=False).head(10).index)
"""

# Sort the list of dictionaries by the revenue key in descending order
sorted_sales = sorted(store_sales, key=lambda x: x["revenue"], reverse=True)

# Print the top 10 items (or just their indexes/identities)
for item in sorted_sales[:10]:
    print(item)




"""
Inventory risk analysis
"""

inventory["stock_status"] = np.where(
    inventory["stock_on_hand"] < inventory["reorder_point"],
    "High Risk",
    "Healthy"
)

# Calculate the percentage of stores at risk
print(
    inventory["stock_status"].value_counts(normalize=True) * 100
)
