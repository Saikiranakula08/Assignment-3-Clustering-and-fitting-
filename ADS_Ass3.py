#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:55:03 2024

@author: saikiranakula
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

def describe_data(dataframe, name):
    """
    Display a statistical summary of the dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to describe.
    - name (str): A label for the DataFrame in the summary.

    Returns:
    - None: Displays the statistical summary.
    """
    print(f"Summary for {name}:\n{dataframe.describe()}\n{'='*40}\n")

def transpose_data(dataframe):
    """
    Transpose the given dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to transpose.

    Returns:
    - DataFrame: The transposed DataFrame.
    """
    return dataframe.transpose()

def spectral_clustering_scatter_plot(data_agri, data_arable, start_year, end_year):
    """
    Generate a scatter plot for the interplay between Agricultural Land and Arable Land using Spectral Clustering.

    Parameters:
    - data_agri (DataFrame): DataFrame containing agricultural land data.
    - data_arable (DataFrame): DataFrame containing arable land data.
    - start_year (int): Starting year for the analysis.
    - end_year (int): Ending year for the analysis.

    Returns:
    - None: Displays the scatter plot with Spectral Clustering results.
    """

    # Extract relevant columns
    columns_to_select_agri = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]
    columns_to_select_arable = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]

    # Copy dataframes to avoid modifying original data
    df_agri = data_agri[columns_to_select_agri].copy()
    df_arable = data_arable[columns_to_select_arable].copy()

    # Transpose the dataframes for better visualization in describe function
    df_agri_transposed = transpose_data(df_agri)
    df_arable_transposed = transpose_data(df_arable)

    # Describe the dataframes
    describe_data(df_agri_transposed, 'Agricultural Land')
    describe_data(df_arable_transposed, 'Arable Land')

    # Merge the two dataframes based on 'Country Name'
    merged_df = pd.merge(df_agri, df_arable, on='Country Name', suffixes=('_agri', '_arable'))

    # Drop rows with missing values
    merged_df.dropna(inplace=True)

    # Prepare data for clustering
    X = merged_df.iloc[:, 1:]  # Select all columns except 'Country Name'

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply spectral clustering
    spectral = SpectralClustering(n_clusters=2, random_state=42)
    merged_df['Cluster'] = spectral.fit_predict(X_scaled)

    # Plot the clustered data
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with grid on the back side
    plt.scatter(merged_df.iloc[:, 1], merged_df.iloc[:, 2], c=merged_df['Cluster'], cmap='coolwarm', edgecolors='k', s=50, zorder=10)
    
    # Grid on the back side
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3, zorder=5)
    
    plt.xlabel(f'Agricultural Land (% of total land area) - {start_year} to {end_year}')
    plt.ylabel(f'Arable Land (% of total land area) - {start_year} to {end_year}')
    plt.title(f'Agricultural Land vs Arable Land from {start_year} to {end_year}')
    plt.show()

# Load your datasets
data_agri = pd.read_excel('/Users/saikiranakula/Downloads/API_AG.LND.AGRI.ZS_DS2_en_excel_v2_6299125.xls', skiprows=3)
data_arable = pd.read_excel('/Users/saikiranakula/Downloads/API_AG.LND.ARBL.ZS_DS2_en_excel_v2_6299397.xls', skiprows=3)

# Specify the range of years for columns
start_year = 1990
end_year = 2020

# Create the scatter plot with Spectral clustering for agricultural land vs. arable land
spectral_clustering_scatter_plot(data_agri, data_arable, start_year, end_year)


def polynomial_curve(x, a, b, c, d):
    """
    Polynomial function for curve fitting.

    Parameters:
    - x (array): Input values.
    - a, b, c, d (float): Coefficients for the polynomial.

    Returns:
    - array: Output values from the polynomial function.
    """
    return a * x**3 + b * x**2 + c * x + d

def describe_data(dataframe, name):
    """
    Display a statistical summary of the dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to describe.
    - name (str): A label for the DataFrame in the summary.

    Returns:
    - None: Displays the statistical summary.
    """
    print(f"Summary for {name}:\n{dataframe.describe()}\n{'='*40}\n")

def transpose_data(dataframe):
    """
    Transpose the given dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to transpose.

    Returns:
    - DataFrame: The transposed DataFrame.
    """
    return dataframe.transpose()

# Read the dataset
file_name = '/Users/saikiranakula/Downloads/API_AG.LND.AGRI.ZS_DS2_en_excel_v2_6299125.xls'
my_data_set = pd.read_excel(file_name, skiprows=3)

# Select Portugal's agricultural land data from the dataset
portugal_agricultural_land = my_data_set[my_data_set['Country Name'] == 'Portugal'][['Country Name'] + [str(year) for year in range(1990, 2022)]]

# Prepare the data for curve fitting
years = np.array(range(1990, 2022))  # Use data up to 2022 for fitting
agricultural_land_values = portugal_agricultural_land.values[:, 1:].flatten()

# Use curve_fit to fit the data
params, covariance = curve_fit(polynomial_curve, years, agricultural_land_values)

# Make predictions for the years 1990 to 2030
years_extended = np.array(range(1990, 2025))  # Extend the range to 2030
predicted_agricultural_land_1990_2030 = polynomial_curve(years_extended, *params)

# Plot the results
plt.figure(figsize=(12, 10))
plt.plot(years, agricultural_land_values, label='Actual Data')
plt.plot(years_extended, predicted_agricultural_land_1990_2030, color='red', label='Curve Fitting Model')
plt.scatter([2025], predicted_agricultural_land_1990_2030[-1], color='green', marker='X', label='Predicted for 2030')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Agricultural Land (% of total land area)', fontsize=15, color='mediumblue')
plt.title('Agricultural Land Prediction for Portugal', fontsize=17, color='mediumblue')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5, color='black')
plt.xticks(range(1990, 2025, 5), fontsize='15')
plt.savefig('portugal_agricultural_land_prediction_curve_fit_2025.png', dpi=300)
plt.show()

# Display the predicted agricultural land for 2030
print(f'Predicted Agricultural Land for Portugal in 2025: {predicted_agricultural_land_1990_2030[-1]:,.2f}% of total land area')

# Transpose and describe the data
transposed_data = transpose_data(portugal_agricultural_land)
describe_data(transposed_data, 'Portugal Agricultural Land')

# Read the dataset
file_name = '/Users/saikiranakula/Downloads/API_AG.LND.AGRI.ZS_DS2_en_excel_v2_6299125.xls'
my_data_set = pd.read_excel(file_name, skiprows=3)

# Select Portugal's agricultural land data from the dataset and transpose it
portugal_agricultural_land = my_data_set[my_data_set['Country Name'] == 'Portugal'][['Country Name'] + [str(year) for year in range(1990, 2022)]]
transposed_data = portugal_agricultural_land.T

# Display summary statistics of the dataset
data_description = transposed_data.describe()

# Prepare the data for curve fitting
years = np.array(range(1990, 2022))  # Use data up to 2022 for fitting
agricultural_land_values = transposed_data.values[1:].flatten()

# Define a polynomial function for curve fitting
def polynomial_curve(x, a, b, c, d):
    """
    Polynomial curve function for curve fitting.

    Parameters:
    - x: Input variable (in this case, years)
    - a, b, c, d: Coefficients of the polynomial curve

    Returns:
    - Predicted values based on the polynomial curve
    """
    return a * x**3 + b * x**2 + c * x + d

# Use curve_fit to fit the data
params, covariance = curve_fit(polynomial_curve, years, agricultural_land_values)

# Make predictions for the years 1990 to 2030
years_extended = np.array(range(1990, 2025))  # Extend the range to 2030
predicted_agricultural_land_1990_2030 = polynomial_curve(years_extended, *params)

# Plot the results without the red curve fitting line
plt.figure(figsize=(12, 10))
plt.plot(years, agricultural_land_values, label='Actual Data')
plt.scatter([2025], predicted_agricultural_land_1990_2030[-1], color='green', marker='X', label='Predicted for 2030')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Agricultural Land (% of total land area)', fontsize=15, color='mediumblue')
plt.title('Agricultural Land Prediction for Portugal', fontsize=17, color='mediumblue')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5, color='black')
plt.xticks(range(1990, 2025, 5), fontsize='15')
plt.savefig('portugal_agricultural_land_prediction_2025.png', dpi=300)
plt.show()

# Display the summary statistics of the dataset
print("Summary Statistics of Agricultural Land Data:")
print(data_description)

# Display the predicted agricultural land for 2030
print(f'Predicted Agricultural Land for Portugal in 2025: {predicted_agricultural_land_1990_2030[-1]:,.2f}% of total land area')

# Read the dataset
file_name = '/Users/saikiranakula/Downloads/API_AG.LND.AGRI.ZS_DS2_en_excel_v2_6299125.xls'
my_data_set = pd.read_excel(file_name, skiprows=3)

# Select United Kingdom's agricultural land data from the dataset
uk_agricultural_land = my_data_set[my_data_set['Country Name'] == 'United Kingdom'][['Country Name'] + [str(year) for year in range(1990, 2022)]]

# Prepare the data for curve fitting
years = np.array(range(1990, 2022))  # Use data up to 2022 for fitting
agricultural_land_values_uk = uk_agricultural_land.values[:, 1:].flatten()

# Define a polynomial function for curve fitting
def polynomial_curve(x, a, b, c, d):
    """
    Polynomial curve function for curve fitting.

    Parameters:
    - x: Input variable (in this case, years)
    - a, b, c, d: Coefficients of the polynomial curve

    Returns:
    - Predicted values based on the polynomial curve
    """
    return a * x**3 + b * x**2 + c * x + d

# Use curve_fit to fit the data for the United Kingdom
params_uk, covariance_uk = curve_fit(polynomial_curve, years, agricultural_land_values_uk)

# Make predictions for the years 1990 to 2030 for the United Kingdom
years_extended = np.array(range(1990, 2025))  # Extend the range to 2030
predicted_agricultural_land_uk_1990_2030 = polynomial_curve(years_extended, *params_uk)

# Plot the results for the United Kingdom
plt.figure(figsize=(12, 10))
plt.plot(years, agricultural_land_values_uk, label='Actual Data (UK)')
plt.plot(years_extended, predicted_agricultural_land_uk_1990_2030, color='red', label='Curve Fitting Model (UK)')
plt.scatter([2025], predicted_agricultural_land_uk_1990_2030[-1], color='green', marker='X', label='Predicted for 2030 (UK)')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Agricultural Land (% of total land area)', fontsize=15, color='mediumblue')
plt.title('Agricultural Land Prediction for the United Kingdom', fontsize=17, color='mediumblue')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5, color='black')
plt.xticks(range(1990, 2025, 5), fontsize='15')
plt.savefig('uk_agricultural_land_prediction_curve_fit_2025.png', dpi=300)
plt.show()

# Display the predicted agricultural land for the United Kingdom in 2025
print(f'Predicted Agricultural Land for the United Kingdom in 2025: {predicted_agricultural_land_uk_1990_2030[-1]:,.2f}% of total land area')


def describe_data(dataframe, name):
    """
    Display a statistical summary of the dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to describe.
    - name (str): A label for the DataFrame in the summary.

    Returns:
    - None: Displays the statistical summary.
    """
    print(f"Summary for {name}:\n{dataframe.describe()}\n{'='*40}\n")

def transpose_data(dataframe):
    """
    Transpose the given dataframe.

    Parameters:
    - dataframe (DataFrame): The DataFrame to transpose.

    Returns:
    - DataFrame: The transposed DataFrame.
    """
    return dataframe.transpose()

# Read the dataset
file_name = '/Users/saikiranakula/Downloads/API_AG.LND.AGRI.ZS_DS2_en_excel_v2_6299125.xls'
my_data_set = pd.read_excel(file_name, skiprows=3)

# Select United Kingdom's agricultural land data from the dataset
uk_agricultural_land = my_data_set[my_data_set['Country Name'] == 'United Kingdom'][['Country Name'] + [str(year) for year in range(1990, 2022)]]

# Describe the data for the United Kingdom
describe_data(transpose_data(uk_agricultural_land), 'United Kingdom Agricultural Land')

# Prepare the data for curve fitting
years = np.array(range(1990, 2022))  # Use data up to 2022 for fitting
agricultural_land_values_uk = uk_agricultural_land.values[:, 1:].flatten()

# Define a polynomial function for curve fitting
def polynomial_curve(x, a, b, c, d):
    """
    Polynomial function for curve fitting.

    Parameters:
    - x (array): Input values.
    - a, b, c, d (float): Coefficients for the polynomial.

    Returns:
    - array: Output values from the polynomial function.
    """
    return a * x**3 + b * x**2 + c * x + d

# Use curve_fit to fit the data for the United Kingdom
params_uk, covariance_uk = curve_fit(polynomial_curve, years, agricultural_land_values_uk)

# Make predictions for the years 1990 to 2030 for the United Kingdom
years_extended = np.array(range(1990, 2025))  # Extend the range to 2030
predicted_agricultural_land_uk_1990_2030 = polynomial_curve(years_extended, *params_uk)

# Plot the results for the United Kingdom without the red curve fitting line
plt.figure(figsize=(12, 10))
plt.plot(years, agricultural_land_values_uk, label='Actual Data (UK)')
plt.scatter([2025], predicted_agricultural_land_uk_1990_2030[-1], color='green', marker='X', label='Predicted for 2030 (UK)')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Agricultural Land (% of total land area)', fontsize=15, color='mediumblue')
plt.title('Agricultural Land Prediction for the United Kingdom', fontsize=17, color='mediumblue')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5, color='black')
plt.xticks(range(1990, 2025, 5), fontsize='15')
plt.savefig('uk_agricultural_land_prediction_2025.png', dpi=300)
plt.show()

# Display the predicted agricultural land for the United Kingdom in 2025
print(f'Predicted Agricultural Land for the United Kingdom in 2025: {predicted_agricultural_land_uk_1990_2030[-1]:,.2f}% of total land area')
