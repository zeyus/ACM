pacman::p_load(tidyverse,
        here,
        posterior,
        cmdstanr,
        brms, tidybayes)
       

trials <- 120

RandomAgentNoise_f <- function(rate, noise) {

  choice <- rbinom(1, 1, rate) # generating noiseless choices
  
  if (rbinom(1, 1, noise) == 1) {
    choice = rbinom(1, 1, 0.5) # introducing noise
  }
  
  return(choice)
}

d <- NULL
for (noise in seq(0, 0.5, 0.1)) { # looping through noise levels

  for (rate in seq(0, 1, 0.1)) { # looping through rate levels
    randomChoice <- rep(NA, trials)
    
    for (t in seq(trials)) { # looping through trials (to make it homologous to more reactive models)
      randomChoice[t] <- RandomAgentNoise_f(rate, noise)
    }
    temp <- tibble(trial = seq(trials), choice = randomChoice, rate, noise)
    temp$cumulativerate <- cumsum(temp$choice) / seq_along(temp$choice)

    if (exists("d")) {
      d <- rbind(d, temp)
    } else{
      d <- temp
    }
  }
}

write_csv(d, "simdata/W3_randomnoise.csv")
