import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

# Haversine distance function
def haversine(a, b, c, d):
    R = 6371
    a_rad = radians(a)
    b_rad = radians(b)
    c_rad = radians(c)
    d_rad = radians(d)
    da = c_rad - a_rad
    db = d_rad - b_rad
    x = sin(da/2)**2 + cos(a_rad) * cos(c_rad) * sin(db/2)**2
    y = 2 * atan2(sqrt(x), sqrt(1-x))
    return R * y

# Load CSV files without headers; added columns for simplicity
st = pd.read_csv('stations.csv', header=None, names=['id', 'wid', 'lat', 'lon'])
tp = pd.read_csv('temp.csv', header=None, names=['id', 'wid', 'mon', 'day', 'temp'])

# Quick prints
print("Shape before cleaning:", st.shape)
st_clean = st.dropna()
st_clean = st_clean[(st_clean['lat'] != 0) & (st_clean['lon'] != 0)]
print("Rows after cleaning:", st_clean)
print("Shape after cleaning:", st_clean.shape)

# Find stations near Cape Canaveral
cape_lat, cape_lon = 28.396837, -80.605659
dists = []
for i, r in st_clean.iterrows():
    dists.append(haversine(cape_lat, cape_lon, r['lat'], r['lon']))
st_clean['d'] = dists
st_near = st_clean[st_clean['d'] <= 100]
print("Stations within 100km:", len(st_near))

# Min temp at station 722040 in January
tp722 = tp[(tp['id'] == 722040) & (tp['mon'] == 1)]
print("Min temp at 722040:", tp722['temp'].min())

# Mean temp on Jan 28
t28 = tp[(tp['mon'] == 1) & (tp['day'] == 28)]
print("Mean temp on Jan 28:", t28['temp'].mean())

# Mean temp for each day for stations near Cape Canaveral
tp_near = pd.merge(tp, st_near[['id']])
tp_near_jan = tp_near[tp_near['mon'] == 1]
mean_daily = tp_near_jan.groupby('day')['temp'].mean()
plt.figure()
plt.plot(mean_daily.index, mean_daily.values, marker='X')
plt.title('Mean Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.grid(True)
plt.savefig('plot1.png')
plt.show()

# Plot 2: Map of US temps on Jan 28
min_lat, max_lat = 25.0, 50.0
min_lon, max_lon = -125.0, -65.0
rows, cols = 100, 150
us_map = np.zeros((rows, cols, 3))

def temp_to_color(t):
    if t > 90:
        return [255, 0, 0]
    elif t > 80:
        return [255, 128, 0]
    elif t > 70:
        return [255, 255, 0]
    elif t > 60:
        return [128, 255, 128]
    elif t > 50:
        return [0, 255, 0]
    elif t > 40:
        return [0, 128, 255]
    else:
        return [128, 0, 128]

t28 = tp[(tp['mon'] == 1) & (tp['day'] == 28)]
t28 = pd.merge(t28, st_clean[['id', 'lat', 'lon']])
for i, r in t28.iterrows():
    a = r['lat']
    b = r['lon']
    t = r['temp']
    if a < min_lat or a > max_lat or b < min_lon or b > max_lon:
        continue
    r_idx = int((max_lat - a) / (max_lat - min_lat) * (rows - 1))
    c_idx = int((b - min_lon) / (max_lon - min_lon) * (cols - 1))
    if r_idx < 0 or r_idx >= rows or c_idx < 0 or c_idx >= cols:
        continue
    us_map[r_idx, c_idx] = temp_to_color(t)

plt.figure()
plt.imshow(us_map)
plt.title('US Temp Jan 28')
plt.tight_layout()
plt.savefig('plot2.png')
plt.show()
