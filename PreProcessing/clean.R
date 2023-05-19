
readr::read_csv("Data/weather_data.csv")
# --------------------------------------------------------------------------------------------------------------------------------------
# Find re-scale values of weather data:
# --------------------------------------------------------------------------------------------------------------------------------------

rain_mean <- mean(wd$rain, na.rm = T); rain_sd <- sd(wd$rain, na.rm = T); rain_missing <- 0
windspeed_mean <- mean(wd$wind_speed, na.rm = T); windspeed_sd <- sd(wd$wind_speed, na.rm = T); windspeed_missing <- 0
temp_mean <- mean(wd$temp, na.rm = T); temp_sd <- sd(wd$temp, na.rm = T); temp_missing <- 0
rh_mean <- mean(wd$rh, na.rm = T); rh_sd <- sd(wd$rh, na.rm = T); rh_missing <- 0

winddir_missing <- -1
leafwetness_missing <- sum(wd$leaf_wetness, na.rm = T) / nrow(wd)


# --------------------------------------------------------------------------------------------------------------------------------------
# rescale weather data:
# --------------------------------------------------------------------------------------------------------------------------------------

wd <- wd %>%
  dplyr::mutate(
    rain = ifelse(is.na(rain), rain_missing, (rain - rain_mean) / rain_sd),
    wind_speed = ifelse(is.na(wind_speed), windspeed_missing, (wind_speed - windspeed_mean)/windspeed_sd),
    temp = ifelse(is.na(temp), temp_missing, (temp - temp_mean)/temp_sd),
    rh = ifelse(is.na(rh), rh_missing, (rh - rh_mean)/rh_sd),
    wind_dir = ifelse(is.na(wind_dir), -1, wind_dir/360 * 2 * pi),
    leaf_wetness = ifelse(is.na(leaf_wetness), leafwetness_missing, as.numeric(leaf_wetness>0)),
    year = lubridate::year(date_time),
    month = lubridate::month(date_time),
    day = lubridate::day(date_time),
    hour = lubridate::hour(date_time),
    minute = lubridate::minute(date_time)
  )

# --------------------------------------------------------------------------------------------------------------------------------------
# write out:
# --------------------------------------------------------------------------------------------------------------------------------------

readr::write_csv(wd, path = "Data/weather_data.csv")
wd1 %>%
  ggplot2::ggplot( ggplot2::aes(rh) ) +
  ggplot2::geom_histogram()
sum(wd$rh < 25, na.rm = T)
summary(wd1)

#x <- cut(weather_data$wind_dir, breaks = seq(0, 360, 5), labels = seq(0, 355, 5)) %>% table() 
#x[1]


# --------------------------------------------------------------------------------------------------------------------------------------
# investigate
# --------------------------------------------------------------------------------------------------------------------------------------

x <- wd %>%
  dplyr::group_by(site_id) %>%
  dplyr::summarise(
    n = dplyr::n(),
    Y_isNA = sum(!leaf_wetness %in% c(0, 1))
  )

weather_data[weather_data$station_id == 328,] %>% summary()

wd_test <- weather_data %>% 
  dplyr::group_by(site_id) %>%
  dplyr::select(site_id, station_id, date_time, rain, wind_speed, wind_dir, "temp"= est_temp, "rh" = est_rh, leaf_wetness, radiation) %>%
  dplyr::mutate(
    rain = ifelse(rain < 0, NA, rain),
    wind_speed = ifelse(wind_speed < 0, NA, wind_speed)
  ) %>%
  dplyr::summarise(
    station_id = stringr::str_c(unique(station_id, collapse = ", ")),
    date_time = stringr::str_c("start: ", min(date_time) %>% lubridate::year(), min(date_time) %>% lubridate::month(),
                               "end: ", max(date_time) %>% lubridate::year(), max(date_time) %>% lubridate::month()),
    rain = stringr::str_c("min: ", min(rain, na.rm = T), ", med: ", median(rain, na.rm = T), ", max: ", max(rain, na.rm = T), ", NA: ", sum(is.na(rain)) ),
    wind_speed = stringr::str_c("min: ", min(wind_speed, na.rm = T), ", med: ", median(wind_speed, na.rm = T), ", max: ", max(wind_speed, na.rm = T), ", NA: ", sum(is.na(wind_speed)) ),
    #wind_dir = 
    temp = stringr::str_c("min: ", min(temp, na.rm = T), ", med: ", median(temp, na.rm = T), ", max: ", max(temp, na.rm = T), ", NA: ", sum(is.na(temp)) ),
    rh = stringr::str_c("min: ", min(rh, na.rm = T), ", med: ", median(rh, na.rm = T), ", max: ", max(rh, na.rm = T), ", NA: ", sum(is.na(rh)) ),
    leaf_wetness = stringr::str_c("wet: ", sum(leaf_wetness > 0, na.rm = T), ", dry: ", sum(leaf_wetness == 0, na.rm = T), ", NA: ", sum(is.na(leaf_wetness))),
    radiation = stringr::str_c("min: ", min(radiation, na.rm = T), ", med: ", median(radiation, na.rm = T), ", max: ", max(radiation, na.rm = T), ", NA: ", sum(is.na(radiation)) )
  )
