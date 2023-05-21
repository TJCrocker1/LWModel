
# --------------------------------------------------------------------------------------------------------------------------------------
# get the sun position
# --------------------------------------------------------------------------------------------------------------------------------------

sun_position <- function(year, month, day, hour, min, sec, lat, lon) {
  
  twopi <- 2 * pi
  deg2rad <- pi / 180
  
  # Get day of the year, e.g. Feb 1 = 32, Mar 1 = 61 on leap years
  month.days <- c(0,31,28,31,30,31,30,31,31,30,31,30)
  day <- day + cumsum(month.days)[month]
  leapdays <- year %% 4 == 0 & (year %% 400 == 0 | year %% 100 != 0) & 
    day >= 60 & !(month==2 & day==60)
  day[leapdays] <- day[leapdays] + 1
  
  # Get Julian date - 2400000
  hour <- hour + min / 60 + sec / 3600 # hour plus fraction
  delta <- year - 1949
  leap <- trunc(delta / 4) # former leapyears
  jd <- 32916.5 + delta * 365 + leap + day + hour / 24
  
  # The input to the Atronomer's almanach is the difference between
  # the Julian date and JD 2451545.0 (noon, 1 January 2000)
  time <- jd - 51545.
  
  # Ecliptic coordinates
  
  # Mean longitude
  mnlong <- 280.460 + .9856474 * time
  mnlong <- mnlong %% 360
  #mnlong[mnlong < 0] <- mnlong[mnlong < 0] + 360
  
  # Mean anomaly
  mnanom <- 357.528 + .9856003 * time
  mnanom <- mnanom %% 360
  mnanom[mnanom < 0] <- mnanom[mnanom < 0] + 360
  mnanom <- mnanom * deg2rad
  
  # Ecliptic longitude and obliquity of ecliptic
  eclong <- mnlong + 1.915 * sin(mnanom) + 0.020 * sin(2 * mnanom)
  eclong <- eclong %% 360
  eclong[eclong < 0] <- eclong[eclong < 0] + 360
  oblqec <- 23.439 - 0.0000004 * time
  eclong <- eclong * deg2rad
  oblqec <- oblqec * deg2rad
  
  # Celestial coordinates
  # Right ascension and declination
  num <- cos(oblqec) * sin(eclong)
  den <- cos(eclong)
  ra <- atan(num / den)
  ra[den < 0] <- ra[den < 0] + pi
  ra[den >= 0 & num < 0] <- ra[den >= 0 & num < 0] + twopi
  dec <- asin(sin(oblqec) * sin(eclong))
  
  # Local coordinates
  # Greenwich mean sidereal time
  gmst <- 6.697375 + .0657098242 * time + hour
  gmst <- gmst %% 24
  gmst[gmst < 0] <- gmst[gmst < 0] + 24.
  
  # Local mean sidereal time
  lmst <- gmst + lon / 15.
  lmst <- lmst %% 24.
  lmst[lmst < 0] <- lmst[lmst < 0] + 24.
  lmst <- lmst * 15. * deg2rad
  
  # Hour angle
  ha <- lmst - ra
  ha[ha < -pi] <- ha[ha < -pi] + twopi
  ha[ha > pi] <- ha[ha > pi] - twopi
  
  # Latitude to radians
  lat <- lat * deg2rad
  
  # Solar zenith angle
  zenithAngle <- acos(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha))
  # Solar azimuth
  az <- acos(((sin(lat) * cos(zenithAngle)) - sin(dec)) / (cos(lat) * sin(zenithAngle)))
  rm(zenithAngle)
  
  # Azimuth and elevation
  el <- asin(sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(ha))
  
  el <- el / deg2rad
  az <- az / deg2rad
  lat <- lat / deg2rad
  
  # -----------------------------------------------
  # Azimuth correction for Hour Angle
  if (ha > 0) az <- az + 180 else az <- 540 - az
  az <- az %% 360
  
  return(list(elevation=el, azimuth=az))
}

# vectorise sun position 
vec_sun_position <- function(date_time, lat, lon) {
  year <- lubridate::year(date_time)
  month <- lubridate::month(date_time)
  day <- lubridate::day(date_time)
  hour <- lubridate::hour(date_time) + lubridate::minute(date_time)/60
  
  azimuth = vector("list", length(year))
  elevation = vector("list", length(year))
  
  for(i in seq_along(date_time)) {
    pos <- sun_position(year[i], month[i], day[i], hour[i], 0, 0, lat[i], lon[i])
    azimuth[[i]] <- pos[["azimuth"]]
    elevation[[i]] <- pos[["elevation"]]
  }
  
  tibble::tibble(
    azimuth = unlist(azimuth),
    elevation = unlist(elevation)
  )
}

#x <- vec_sun_position(wd$date_time[1:10000], wd$lat[1:10000], wd$lon[1:10000])
#wd$azimuth[1:10]
#wd$elevation[1:10]

# --------------------------------------------------------------------------------------------------------------------------------------
# get train / test / val sets and the time series groups
# --------------------------------------------------------------------------------------------------------------------------------------

get_set <- function(station_id, time_step, val_size, test) {
  g <- 0
  set <- vector("list", unique(station_id) %>% length())
  group <- vector("list", unique(station_id) %>% length())
  
  for(station in unique(station_id)) {
    g <- g + 1
    times <- time_step[station_id == station]
    id <- seq_along(times); len <- length(times)
    
    sub_set <- vector("character", len)
    sub_group <- vector("integer", len)
    sub_group[1] <- g
    
    if(station %in% test) {
      sub_set <- rep("test", len)
    } else {
      sub_set[1] <- "train"
      size <- floor(len * val_size)
      start <- sample(id[-c(1,(len-size):len)], 1)
    }
    
    for(i in id[-1]) {
      switch_group <- F
      if(times[i] > 0.25) {switch_group <- T}
      if(!station %in% test) {
        if(i < start | i >= (start + size)) {sub_set[i] <- "train"}
        if(i >= start & i < (start + size)) {sub_set[i] <- "val"}
        if(i == start | i == (start + size)) {switch_group <- T}
      }
      if(switch_group) {g <- g + 1}
      sub_group[i] <- g
    }
    
    set[[length(set)+1]] <- sub_set
    group[[length(group)+1]] <- sub_group
  }
  
  tibble::tibble(
    #station_id = station_id,
    set = unlist(set),
    group = unlist(group)
  )
}
