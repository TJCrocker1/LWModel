`%>%`  <- magrittr::`%>%`
source('functions.R')
# --------------------------------------------------------------------------------------------------------------------------------------
# assemble the data from raw:
# --------------------------------------------------------------------------------------------------------------------------------------

load("../Data/stat_loc.RData")

wd <- readr::read_csv("../Data/WDdat_bysite_2022-09-16.csv") %>%
  dplyr::select(station_id, date_time, rain, wind_speed, wind_dir, "temp"= est_temp, "rh" = est_rh, leaf_wetness) %>%
  dplyr::mutate(
    rain = ifelse(rain < 0, NA, rain),
    rh = ifelse(rh < 0, NA, rh),
    wind_speed = ifelse(wind_speed < 0, NA, wind_speed),
    wind_dir = ifelse(wind_dir < 0, NA, wind_dir),
    leaf_wetness = ifelse(leaf_wetness < 0, NA, leaf_wetness)
  ) %>% 
  dplyr::left_join(station_locations %>% dplyr::select(station_id, "lat" = station_lat, "lon" = station_lon),
                   by = "station_id") %>%
  dplyr::group_by(station_id, date_time) %>%
  dplyr::summarise(
    temp = unique(temp),
    rain = unique(rain),
    wind_speed = unique(wind_speed),
    wind_dir = unique(wind_dir),
    rh = unique(rh),
    leaf_wetness = unique(leaf_wetness),
    lat = unique(lat),
    lon = unique(lon)
  )

# --------------------------------------------------------------------------------------------------------------------------------------
# Split the data into train, val and test:
# --------------------------------------------------------------------------------------------------------------------------------------
# - test is all the data from two completely unseen sites (~13\% of the data)
# - val is 10\% of every other site, picked from a random starting point
# - training data sets are the rest of the data

test <- c(303, 327)
and_the_rest <- c(326, 338, 335, 336, 302, 341, 304, 328, 330, 312, 329, 324, 309, 331)

wd <- wd %>% 
  dplyr::filter(station_id %in% c(test, and_the_rest)) %>%
  dplyr::group_by(station_id) %>%
  dplyr::arrange(station_id, date_time) %>%
  dplyr::mutate(time_step = (date_time - dplyr::lag(date_time)) %>% as.numeric(., "hours")) %>%
  dplyr::ungroup() %>%
  dplyr::mutate( get_set(station_id, time_step, 0.1, test) )

#wd_test %>%
#  dplyr::group_by(group) %>%
#  dplyr::summarise(
#    n = dplyr::n(),
#    ts = max(time_step, na.rm = T),
#    n_station = length(unique(station_id)),
#    station_ids = stringr::str_c(unique(station_id), collapse = ", ")
#  ) %>%
#  dplyr::filter(n_station>1)

#unique(wd_test$group)


# --------------------------------------------------------------------------------------------------------------------------------------
# Feature engineering:
# --------------------------------------------------------------------------------------------------------------------------------------

wd <- wd %>%
  dplyr::mutate(
    
    vec_sun_position(date_time, lat, lon),
    
    year_sin = sin( as.numeric(date_time) * (2*pi/31556952)),
    year_cos = cos( as.numeric(date_time) * (2*pi/31556952)),
    
    rain = purrr::map_dbl(rain, ~{
      if(is.na(.) | . < 0) {return(0)}
      if(. > 10) {return(10)}
      if(. == 0) {return(0)}
      ceiling(.*5)/5
    }), 
    
    windspeed = ifelse(is.na(wind_speed) | is.na(wind_dir) | wind_speed < 0, 0, wind_speed),
    winddir = ifelse(is.na(wind_dir), 0, (wind_dir/360 * 2 * pi)),
    windX = windspeed*cos(wind_dir),
    windY = windspeed*sin(wind_dir),
    
    leaf_wetness = ifelse(is.na(leaf_wetness), 0.5, as.numeric(leaf_wetness>0))

    
  ) %>%
  dplyr::select(station_id, date_time, lat, lon, year_sin, year_cos, rain, temp, rh, windX, windY, azimuth, elevation, leaf_wetness, set, group)

# --------------------------------------------------------------------------------------------------------------------------------------
# Find mean & sd then re-scale values of weather data:
# --------------------------------------------------------------------------------------------------------------------------------------
wd_train <- wd[wd$set == "train", ]
# year_sin
year_sin_mean <- mean(wd_train$year_sin, na.rm = T)
year_sin_sd <- sd(wd_train$year_sin, na.rm = T)

# year_cos
year_cos_mean <- mean(wd_train$year_cos, na.rm = T)
year_cos_sd <- sd(wd_train$year_cos, na.rm = T)

# rain
rain_mean <- mean(wd_train$rain, na.rm = T)
rain_sd <- sd(wd_train$rain, na.rm = T)

# temp
temp_mean <- mean(wd_train$temp, na.rm = T)
temp_sd <- sd(wd_train$temp, na.rm = T)

# rh
rh_mean <- mean(wd_train$rh, na.rm = T)
rh_sd <- sd(wd_train$rh, na.rm = T)

# windY
windY_mean <- mean(wd_train$windY, na.rm = T)
windY_sd <- sd(wd_train$windY, na.rm = T)

# windX
windX_mean <- mean(wd_train$windX, na.rm = T)
windX_sd <- sd(wd_train$windX, na.rm = T)

# azimuth
azimuth_mean <- mean(wd_train$azimuth, na.rm = T)
azimuth_sd <- sd(wd_train$azimuth, na.rm = T)

# elevation
elevation_mean <- mean(wd_train$elevation, na.rm = T)
elevation_sd <- sd(wd_train$elevation, na.rm = T)

wd <- wd %>%
  dplyr::mutate(
    year_sin = ifelse(is.na(year_sin), 0, (year_sin - year_sin_mean) / year_sin_sd),
    year_cos = ifelse(is.na(year_cos), 0, (year_cos - year_cos_mean) / year_cos_sd),
    rain = ifelse(is.na(rain) | rain < 0, 0, (rain - rain_mean) / rain_sd),
    temp = ifelse(is.na(temp) | temp < -273, 0, (temp - temp_mean) / temp_sd),
    rh = ifelse(is.na(rh) | rh > 100 | rh < 0, 0, (rh-rh_mean) / rh_sd),
    windX = ifelse(is.na(windX), 0, (windX - windX_mean) / windX_sd),
    windY = ifelse(is.na(windY), 0, (windY - windY_mean) / windY_sd),
    azimuth = ifelse(is.na(azimuth), 0, (azimuth - azimuth_mean) / azimuth_sd),
    elevation = ifelse(is.na(elevation), 0, (elevation - elevation_mean) / elevation_sd),
  )

# --------------------------------------------------------------------------------------------------------------------------------------
# write out:
# --------------------------------------------------------------------------------------------------------------------------------------

readr::write_csv(wd, path = "../Data/weather_data.csv")

readr::write_csv(wd[wd$set == "train", ], "../Data/weather_data_train.csv")
readr::write_csv(wd[wd$set == "val", ], "../Data/weather_data_val.csv")
readr::write_csv(wd[wd$set == "test", ], "../Data/weather_data_test.csv")


# --------------------------------------------------------------------------------------------------------------------------------------
# muck about
# --------------------------------------------------------------------------------------------------------------------------------------

wd_val <- wd[wd$set == "val", ]

x <- stringr::str_c(wd_val$group, wd_val$date_time)
length(x)
length(unique(x))


x <- wd %>%
  dplyr::mutate(key = stringr::str_c(group, date_time)) %>%
  dplyr::group_by(key) %>%
  dplyr::summarise(
    g = group,
    n = dplyr::n(),
    id = stringr::str_c(station_id, collapse = ", ")
  ) %>%
  dplyr::filter(n > 1)

x %>%
  dplyr::ungroup() %>%
  dplyr::count(g)

unique(x$g)

wd <- wd %>%
  dplyr::arrange(site_id, date_time) %>%
  dplyr::mutate(
    obs_id = seq_along(site_id),
    
    rain = purrr::map_dbl(rain, ~{
      if(is.na(.)) {return(0)}
      if(. > 10) {return(10)}
      if(. == 0) {return(0)}
      ceiling(.*5)/5
    }),
    temp = ifelse(is.na(temp), temp_missing, (temp - temp_mean)/temp_sd),
    rh = ifelse(is.na(rh), rh_missing, rh/rh_upper),
    
    windspeed = ifelse(is.na(wind_speed) | is.na(wind_dir), 0, wind_speed),
    winddir = ifelse(is.na(wind_dir), 0, (wind_dir/winddir_upper * 2 * pi)),
    windX = windspeed*cos(wind_dir),
    windY = windspeed*sin(wind_dir),

    azimuth = (azimuth/az_upper * 2 * pi)-pi,
    elevation = ((elevation - el_lower)/(el_upper - el_lower) * 2*pi) -pi,
    

  ) %>%
  dplyr::group_by(site_id) %>%
  dplyr::mutate(
    time_diff = (date_time - dplyr::lag(date_time)) %>% as.numeric(., "hours"),
    time_diff = ifelse(is.na(time_diff), 0.25, time_diff)
    ) %>%
  dplyr::select(obs_id, site_id, station_id, lat, lon, date_time, time_diff, azimuth, elevation, windX, windY, temp, rh, rain, leaf_wetness)






# --------------------------------------------------------------------------------------------------------------------------------------
# plot the locations of weather stations
# --------------------------------------------------------------------------------------------------------------------------------------

uk <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf") %>%  
  dplyr::select(name, continent, geometry) %>%  
  dplyr::filter(name == 'United Kingdom')

wd %>%
  dplyr::group_by(station_id) %>%
  dplyr::summarise(
    lat = unique(lat),
    lon = unique(lon),
    n = dplyr::n(),
    time_min = min(date_time),
    time_max = max(date_time),
    ydat = sum(!is.na(leaf_wetness)) / dplyr::n()
  ) %>%
  dplyr::filter(n > 3000) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(lat) %>%
  dplyr::mutate(
    label = stringr::str_c("Start: ", stringr::str_extract(time_min, "^.{7}"),
                           ",\n End: ", stringr::str_extract(time_max, "^.{7}"),
                           ",\n id: ", station_id, ", n: ", n),
    order = seq_along(station_id),
    side = ifelse(order %% 2 == 0, "left", "right"), 
    height = ifelse(side == "left", order %/% 2, 1+order %/% 2 ), 
    lab_lat = 49 + height * (57-49) / sum(side == "right"),
    lab_lon = ifelse(side == "left", -6, 2.5)
  ) %>%
  ggplot2::ggplot(  ) +
  ggplot2::geom_sf( data = uk ) +
  ggplot2::geom_point( ggplot2::aes(lon, lat, colour = ydat) ) +
  ggplot2::geom_segment( ggplot2::aes(lon, lat, xend = lab_lon, yend = lab_lat) ) +
  ggplot2::geom_label( 
    ggplot2::aes(lab_lon, lab_lat, label = label),
    size = 2) +
  ggplot2::coord_sf(xlim = c(-10, 5), ylim = c(49, 57))


wd %>%
dplyr::filter(!rain == 0) %>%
  ggplot2::ggplot( ggplot2::aes(rain) ) +
   ggplot2::geom_histogram(binwidth = .2) +
  ggplot2::facet_wrap(~site_id)
#sum(wd$rh < 25, na.rm = T)
#summary(wd1)

ceiling()

bin

#x <- cut(weather_data$wind_dir, breaks = seq(0, 360, 5), labels = seq(0, 355, 5)) %>% table() 
#x[1]

wd %>% 
  dplyr::mutate(
    windX = ws*cos(wind_dir),
    windY = ws*sin(wind_dir)
  ) %>%
  ggplot2::ggplot( ggplot2::aes(windY) ) +
  ggplot2::geom_histogram(binwidth = .1) +
  ggplot2::coord_cartesian(xlim = c(-25, 25))

wd %>% 
  dplyr::mutate(
    windX = ws*cos(wind_dir),
    windY = ws*sin(wind_dir)
  ) %>%
  ggplot2::ggplot( ggplot2::aes(windX, windY) ) +
  ggplot2::stat_bin2d(binwidth = 0.1) +
  ggplot2::coord_cartesian(xlim = c(-2, 2), ylim = c(-2,2))


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
