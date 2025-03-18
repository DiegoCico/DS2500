import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import matplotlib.patches as mpatches

EARTH_RADIUS = 6371

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the haversine distance between two geographic points.

    Parameters:
      lat1 (float): Latitude of the first point in degrees.
      lon1 (float): Longitude of the first point in degrees.
      lat2 (float): Latitude of the second point in degrees.
      lon2 (float): Longitude of the second point in degrees.

    Returns:
      float: Distance between the two points in kilometers.
    """
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS * c

def read_data(stations_file, temp_file):
    """
    Read station and temperature data from CSV files.

    Parameters:
      stations_file (str): Filename for station data.
      temp_file (str): Filename for temperature data.

    Returns:
      tuple: (stations_df, temp_df)
    """
    stations_df = pd.read_csv(
        stations_file,
        header=None,
        names=['id', 'wid', 'lat', 'lon']
    )
    temp_df = pd.read_csv(
        temp_file,
        header=None,
        names=['id', 'wid', 'mon', 'day', 'temp']
    )
    return stations_df, temp_df

def clean_stations_data(stations_df):
    """
    Clean the station data by dropping missing values and invalid coordinates.

    Parameters:
      stations_df (DataFrame): DataFrame containing station data.

    Returns:
      DataFrame: Cleaned station data.
    """
    cleaned_df = stations_df.dropna()
    cleaned_df = cleaned_df[
        (cleaned_df['lat'] != 0) & (cleaned_df['lon'] != 0)
        ]
    return cleaned_df

def filter_stations_near(stations_df, center_lat, center_lon, max_distance=100):
    """
    Filter stations that are within a specified distance from a center point.

    Parameters:
      stations_df (DataFrame): Cleaned station data.
      center_lat (float): Latitude of the center point.
      center_lon (float): Longitude of the center point.
      max_distance (float): Maximum distance in kilometers.

    Returns:
      DataFrame: Stations within the specified distance with an added 'd' column.
    """
    distances = []
    for _, row in stations_df.iterrows():
        distance = haversine_distance(
            center_lat, center_lon, row['lat'], row['lon']
        )
        distances.append(distance)
    stations_df = stations_df.copy()
    stations_df['d'] = distances
    near_df = stations_df[stations_df['d'] <= max_distance]
    return near_df

def get_min_temp_at_station(temp_df, station_id, month):
    """
    Get the minimum temperature for a given station and month.

    Parameters:
      temp_df (DataFrame): Temperature data.
      station_id (int): Station identifier.
      month (int): Month (as an integer).

    Returns:
      float: Minimum temperature.
    """
    station_temp = temp_df[(temp_df['id'] == station_id) & (temp_df['mon'] == month)]
    return station_temp['temp'].min()

def get_mean_temp_on_day(temp_df, month, day):
    """
    Get the mean temperature for a specific month and day.

    Parameters:
      temp_df (DataFrame): Temperature data.
      month (int): Month (as an integer).
      day (int): Day of the month.

    Returns:
      float: Mean temperature.
    """
    day_temp = temp_df[(temp_df['mon'] == month) & (temp_df['day'] == day)]
    return day_temp['temp'].mean()

def plot_mean_temperature(temp_df, stations_near_df, month):
    """
    Plot mean daily temperature for stations near a center for a given month.

    Parameters:
      temp_df (DataFrame): Temperature data.
      stations_near_df (DataFrame): Data for stations near the center.
      month (int): Month for which to plot the data.
    """
    temp_near_df = pd.merge(temp_df, stations_near_df[['id']], on='id')
    temp_near_month_df = temp_near_df[temp_near_df['mon'] == month]
    mean_daily_temp = temp_near_month_df.groupby('day')['temp'].mean()

    plt.figure()
    plt.plot(mean_daily_temp.index, mean_daily_temp.values, marker='X')
    plt.title('Mean Temperature')
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.grid(False)
    plt.savefig('plot1.png')
    plt.show()

def temp_to_color(temp):
    """
    Convert a temperature value to an RGB color.

    Parameters:
      temp (float): Temperature value.

    Returns:
      list: RGB color as a list of three integers.
    """
    if temp > 90:
        return [255, 0, 0]
    elif temp > 80:
        return [255, 128, 0]
    elif temp > 70:
        return [255, 255, 0]
    elif temp > 60:
        return [128, 255, 128]
    elif temp > 50:
        return [0, 255, 0]
    elif temp > 40:
        return [0, 128, 255]
    else:
        return [128, 0, 128]

def plot_us_temperature_map(temp_df, stations_df, month, day):
    """
    Plot a US temperature map for a given day with a legend indicating the temperature ranges.

    Parameters:
      temp_df (DataFrame): Temperature data.
      stations_df (DataFrame): Station data with latitude and longitude.
      month (int): Month for which to plot the map.
      day (int): Day of the month.
    """
    min_lat, max_lat = 25.0, 50.0
    min_lon, max_lon = -125.0, -65.0
    rows, cols = 100, 150
    us_map = np.zeros((rows, cols, 3))

    temp_day_df = temp_df[(temp_df['mon'] == month) & (temp_df['day'] == day)]
    temp_day_df = pd.merge(temp_day_df, stations_df[['id', 'lat', 'lon']], on='id')

    for _, row in temp_day_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        temperature = row['temp']

        if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
            continue
        row_idx = int((max_lat - lat) / (max_lat - min_lat) * (rows - 1))
        col_idx = int((lon - min_lon) / (max_lon - min_lon) * (cols - 1))
        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            us_map[row_idx, col_idx] = temp_to_color(temperature)

    plt.figure(figsize=(12, 8))
    plt.imshow(us_map)
    plt.title('All the temperatures in the US on January 28th, 1986')
    plt.subplots_adjust(right=0.75)

    legend_patches = [
        mpatches.Patch(color=(1, 0, 0), label='> 90°F'),
        mpatches.Patch(color=(1, 128/255, 0), label='81°F - 90°F'),
        mpatches.Patch(color=(1, 1, 0), label='71°F - 80°F'),
        mpatches.Patch(color=(128/255, 1, 128/255), label='61°F - 70°F'),
        mpatches.Patch(color=(0, 1, 0), label='51°F - 60°F'),
        mpatches.Patch(color=(0, 128/255, 1), label='41°F - 50°F'),
        mpatches.Patch(color=(128/255, 0, 128/255), label='≤ 40°F')
    ]

    # Place the legend outside the plot area
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Temperature Ranges')
    plt.savefig('plot2.png', bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to execute the analysis and visualizations.
    """
    # File names for the CSV data
    stations_file = 'stations.csv'
    temp_file = 'temp.csv'

    # Read data from CSV files
    stations_df, temp_df = read_data(stations_file, temp_file)

    # Print shape before cleaning
    print(f"Shape before cleaning: {stations_df.shape}")

    # Clean station data
    stations_clean_df = clean_stations_data(stations_df)
    print("Stations data after cleaning:")
    print(stations_clean_df)
    print(f"Shape after cleaning: {stations_clean_df.shape}")

    # Find stations near Cape Canaveral
    cape_lat = 28.396837
    cape_lon = -80.605659
    stations_near_df = filter_stations_near(stations_clean_df, cape_lat, cape_lon)
    print(f"Stations within 100 km: {len(stations_near_df)}")

    # Compute temperature statistics for January
    station_id = 722040
    jan = 1
    min_temp = get_min_temp_at_station(temp_df, station_id, jan)
    print(f"Min temperature at station {station_id}: {min_temp}")

    mean_temp_jan28 = get_mean_temp_on_day(temp_df, jan, 28)
    print(f"Mean temperature on Jan 28: {mean_temp_jan28}")

    # Plot mean daily temperature for stations near Cape Canaveral in January
    plot_mean_temperature(temp_df, stations_near_df, jan)

    # Plot US temperature map for January 28 with the color legend outside the graph
    plot_us_temperature_map(temp_df, stations_clean_df, jan, 28)

if __name__ == '__main__':
    main()
