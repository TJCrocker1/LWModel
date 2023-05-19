import PreProcessing.sunpos as sp
import pandas as pd

#wd = pd.read_csv('Data/WDdat_bysite_2022-09-16.csv')
wd = pd.read_csv('../Data/weather_data.csv')
wd["date_time"] = pd.to_datetime(wd['date_time'])

# add the year, month, day, hour and minute variables
year = [int(i) for i in wd['date_time'].dt.strftime('%Y')]
month = [int(i) for i in wd['date_time'].dt.strftime('%-m')]
day = [int(i) for i in wd['date_time'].dt.strftime('%-d')]
hour = [int(i) for i in wd['date_time'].dt.strftime('%-H')]
minute = [int(i) for i in wd['date_time'].dt.strftime('%-M')]

hour = [hour[i] + minute[i]/60 for i in range(0, len(hour))]

# add sun position to weather data:
wd[["azimuth"]] = 0
wd[["elevation"]] = 0

for i in range(0, wd[wd.columns[0]].count()):
    print(i)
    sun_pos = sp.sun_position(year = year[i], month = month[i], day = day[i],
                          hour = hour[i], minute= 0, sec = 0,
                          lat= wd["lat"][i], longitude= wd["lon"][i])
    wd.loc[i, "azimuth"] = sun_pos[0]
    wd.loc[i, "elevation"] = sun_pos[1]

wd.to_csv("../Data/weather_data.csv")

