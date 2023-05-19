# join lat lon to weather data process NA:
`%>%`  <- magrittr::`%>%`
load("../Data/stat_loc.RData")
wd <- readr::read_csv("../Data/WDdat_bysite_2022-09-16.csv") %>%
  dplyr::select(site_id, station_id, date_time, rain, wind_speed, wind_dir, "temp"= est_temp, "rh" = est_rh, leaf_wetness) %>%
  dplyr::mutate(
    rain = ifelse(rain < 0, NA, rain),
    rh = ifelse(rh < 0, NA, rh),
    wind_speed = ifelse(wind_speed < 0, NA, wind_speed),
    wind_dir = ifelse(wind_dir < 0, NA, wind_dir),
    leaf_wetness = ifelse(leaf_wetness < 0, NA, leaf_wetness)
  ) %>% 
  dplyr::left_join(station_locations %>% dplyr::select(station_id, "lat" = station_lat, "lon" = station_lon),
                   by = "station_id")
readr::write_csv(wd, file = "../Data/weather_data.csv")
