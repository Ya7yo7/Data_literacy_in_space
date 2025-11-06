import pandas as pd

# Define Mars landing sites (planetocentric latitude, east longitude)
sites = [
    ("Spirit",       -14.5692, 175.4729),
    ("Opportunity",   -1.9483, 354.4742),
    ("Phoenix",       68.22,   234.25),
    ("Curiosity",     -4.5895, 137.4417),
    ("InSight",        4.502,  135.623),
    ("Perseverance",  18.44,    77.45)
]

# Create bounding boxes ±0.5° in both directions
def make_box(lat, lon):
    lat_min = max(-90.0, lat - 0.5)
    lat_max = min(90.0, lat + 0.5)
    lon_west = (lon - 0.5) % 360.0
    lon_east = (lon + 0.5) % 360.0
    return lat_min, lat_max, lon_west, lon_east

records = []
for name, lat, lon in sites:
    lat_min, lat_max, lon_west, lon_east = make_box(lat, lon)
    records.append({
        "Mission": name,
        "Latitude_center": lat,
        "Longitude_center": lon,
        "Min_latitude": lat_min,
        "Max_latitude": lat_max,
        "Westernmost_longitude": lon_west,
        "Easternmost_longitude": lon_east
    })

# Store and display as DataFrame
df = pd.DataFrame(records)
print(df)

# Optional: save to CSV for easy copy into ODE
df.to_csv("mars_landing_boxes_1deg.csv", index=False)
