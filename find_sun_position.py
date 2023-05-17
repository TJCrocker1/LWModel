import sunangle.sunpos as sp
import pandas as pd

wd = pd.read_csv('Data/weather_data.csv')

# add sun position to weather data:
wd[["azimuth"]] = 0
wd[["elevation"]] = 0

for i in range(0, wd[wd.columns[0]].count()):
    print(i)
    sun_pos = sp.sun_position(year = wd["year"][i], month = wd["month"][i], day = wd["day"][i],
                          hour= wd["hour"][i], minute= wd["minute"][i],
                          lat= wd["lat"][i], longitude= wd["lon"][i])
    wd.loc[i, "azimuth"] = sun_pos[0]
    wd.loc[i, "elevation"] = sun_pos[1]

wd.to_csv("Data/weather_data_out.csv")

