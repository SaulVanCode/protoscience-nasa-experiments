#!/usr/bin/env python3
"""
Download all NASA datasets used in ProtoScience experiments.
Run this once: python scripts/download_data.py
"""
import urllib.request
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def download(url, filename, desc):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"  [skip] {filename} already exists")
        return
    print(f"  Downloading {desc}...")
    urllib.request.urlretrieve(url, path)
    print(f"  -> {filename}")


print("=" * 60)
print("Downloading NASA datasets for ProtoScience experiments")
print("=" * 60)

# E062: Exoplanets
print("\n1. NASA Exoplanet Archive (TAP API)")
download(
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+pl_name,pl_orbper,pl_orbsmax,pl_rade,pl_bmasse,pl_eqt,st_teff,st_rad,st_mass,sy_dist+FROM+ps+WHERE+pl_orbper+IS+NOT+NULL+AND+pl_orbsmax+IS+NOT+NULL+AND+default_flag=1&format=csv",
    "exoplanets.csv",
    "3,500+ confirmed exoplanets"
)

# E063: Fireballs
print("\n2. NASA CNEOS Fireball Database (API)")
download(
    "https://ssd-api.jpl.nasa.gov/fireball.api",
    "fireballs.json",
    "1,000+ fireball events"
)

# E065: Sunspots
print("\n3. SILSO Monthly Sunspot Numbers")
download(
    "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
    "sunspots_monthly.csv",
    "Monthly sunspots 1749-present"
)

# E066: Gravitational Waves
print("\n4. GWTC Gravitational Wave Event Catalog")
download(
    "https://gwosc.org/eventapi/csv/GWTC/",
    "gw_events.csv",
    "219 LIGO/Virgo merger events"
)

# E067: Asteroids
print("\n5. JPL Small-Body Database (10,000 asteroids)")
download(
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=full_name,e,a,i,om,w,per,n,ma,tp,epoch,H,diameter,albedo,class&sb-kind=a&limit=10000",
    "asteroids_sbdb.json",
    "10,000 asteroid orbital elements"
)

# E068: Mars Weather
print("\n6. Mars Curiosity REMS Weather Data")
download(
    "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json",
    "mars_weather.json",
    "4,500+ sols of Mars weather"
)

print("\n" + "=" * 60)
print("All datasets downloaded to data/")
print("=" * 60)
