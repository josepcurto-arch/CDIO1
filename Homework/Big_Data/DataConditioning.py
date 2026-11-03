"""
Load data:
1. Load the dataset "shoreline_distances_2017_2024.csv" (one of the files in the zip available in Atenea)
2. Inspect the structure (head, tail, info, plot).
3. Convert the date column to datetime format.
4. Create a pivot table with date as index, transect_id as columns, and distance_m as values.
Clean data:
5. Compute percentage of missing values and plot the distribution by transect.
6. Select a subset of transects (a spatial region) with relatively low missing data (e.g., ~100 transects).
7. Filter problematic transects (columns) based on a threshold (e.g., 10–30% of NaN).
8. Filter outliers per transect (e.g., using IQR per transect)
9. Filter low coverage dates (rows) (e.g., with more than 30–50% NaNs in valid transects)
Time series construction:
10. Compute the aggregated (mean) shoreline distance series across transects (in the selected region).
11. Explore resampling at different frequencies (e.g., weekly, biweekly, monthly) and choose the most appropriate one.
12. Fill missing values using linear interpolation.
13. Save final time series to CSV
"""


import pandas as pd

coastlineData = pd.read_csv("Homework/Big_Data/shoreline_distances_2017_2024.csv")

Capcalera = coastlineData.keys()


coastlineData.date = pd.to_datetime(coastlineData.date,yearfirst=True)
print(type(coastlineData.date))

Pivot = coastlineData[["date","transect_id","distance_m"]]
print(Pivot)
