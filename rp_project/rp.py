# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:58:29 2024

@author: rafa
"""

import matplotlib.pyplot as plt
import numpy as np

# Dummy data for the purpose of demonstration (replace with actual data)
years = np.array([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
consumption = np.array([3659, 4199.88, 4702, 4976, 5420, 5782, 5802, 6120, 6591,7150, 7285])

plt.figure(figsize=(15, 3))  # Adjusted to match the aspect ratio of the original plot
plt.bar(years, consumption, color='purple')
plt.title('Electricity Consumption')
plt.xlabel('Year')
plt.ylabel('Consumption (TWh)')
plt.xticks(years)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the recreated plot image

plt.show()  # Show the plot
#%%
import matplotlib.pyplot as plt
production_amounts = [3325.07, 662.05, 43.07, 74.29, 50.0]
power_sources = ['Thermal', 'Hydro', 'Wind', 'Nuclear', 'Solar']

# Calculating the percentage for the pie chart labels
total_production = sum(production_amounts)
percentages = [f"{(amount / total_production) * 100:.1f}%" for amount in production_amounts]

# Creating the pie chart
plt.figure(figsize=(10, 6))
plt.pie(production_amounts, labels=percentages, startangle=140, colors=['#5cacee', '#8db6cd', '#8470ff', '#7d9ec0', '#b0c4de'])
plt.title("China's Electricity Production by Power Source in 2010")
plt.legend(power_sources, loc='upper right')

#%%
import matplotlib.pyplot as plt

# Given data for 2010 and 2020
production_amounts_2010 = [3325.07, 662.05, 43.07, 74.29, 50.0]
production_amounts_2020 = [5330.25, 1355.21, 414.60, 366.25, 142.10]
power_sources = ['Thermal', 'Hydro', 'Wind', 'Nuclear', 'Solar']
colors = ['#5cacee', '#8db6cd', '#8470ff', '#7d9ec0', '#b0c4de']

# Calculate percentages for 2010 and 2020
percentages_2010 = [f"{(amount / sum(production_amounts_2010)) * 100:.1f}%" for amount in production_amounts_2010]
percentages_2020 = [f"{(amount / sum(production_amounts_2020)) * 100:.1f}%" for amount in production_amounts_2020]

# Create subplots for 2010 and 2020
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Pie chart for 2010
ax1.pie(production_amounts_2010, labels=percentages_2010, startangle=140, colors=colors)
ax1.set_title("China's Electricity Production by Power Source in 2010 ")
ax1.legend(power_sources, loc='upper right')

# Pie chart for 2020
ax2.pie(production_amounts_2020, labels=percentages_2020, startangle=140, colors=colors)
ax2.set_title("China's Electricity Production by Power Source in 2020 ")
ax2.legend(power_sources, loc='upper right')

plt.tight_layout()
plt.show()



#%%


