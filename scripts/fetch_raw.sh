#!/usr/bin/env bash
# Public raw data for all films (NOAA SPC tornadoes, NOAA NHC HURDAT2, USGS ComCat earthquakes). Run from the repository root.
set -eu
mkdir -p data/raw/quakes
UA="Mozilla/5.0"
curl -sfL --retry 3 -A "$UA" -o data/raw/tornadoes.csv https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv
curl -sfL --retry 3 -A "$UA" -o data/raw/hurdat2.txt https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt
curl -sfL --retry 3 -A "$UA" -o data/raw/quakes/ca_m4.csv "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=1900-01-01&endtime=2025-12-31&minmagnitude=4&minlatitude=31.5&maxlatitude=42.6&minlongitude=-125.5&maxlongitude=-113.5&orderby=time-asc&limit=20000"
test "$(wc -l < data/raw/quakes/ca_m4.csv)" -gt 5000
