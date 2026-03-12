# How to build your own image recognition app with R! [Part 1] -----------------
# Reference: https://www.r-bloggers.com/2021/03/how-to-build-your-own-image-recognition-app-with-r-part-1/
options(timeout = max(300, getOption("timeout")))

if (!require("pacman")) 
  install.packages("pacman")

library(tidyverse)
library(tensorflow)
library(keras3)
library(reticulate)
library(pacman)


p_load(conflicted)


conflict_prefer("shape", "keras3", quiet = TRUE)
conflict_prefer("set_random_seed", "keras3", quiet = TRUE)










