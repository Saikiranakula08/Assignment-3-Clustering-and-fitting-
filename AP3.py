#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:14:46 2024

@author: saikiranakula
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

def spectral_clustering_scatter_plot(data_agri, data_arable, start_year, end_year):
    # Extract relevant columns
    columns_to_select_agri = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]
    columns_to_select_arable = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]

    df_agri = data_agri[columns_to_select_agri].copy()
    df_arable = data_arable[columns_to_select_arable].copy()

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
