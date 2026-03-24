# Setup -----------------------------------------------------------------------
options(timeout = max(300, getOption("timeout")))

#> Reference: Some ideas and code snippers were used from the following GitHub repository:
#> https://github.com/AzKurban-edX-DS/harvardx-movielens
if(!require(matrixStats))
  install.packages("matrixStats")

if(!require(dslabs))
  install.packages("dslabs")
if(!require(tidyverse))
  install.packages("tidyverse")

#> `stringr` library is already included to the `tidyverse` package,
#> there's no need to install `stringr`
# if(!require(stringr))
#   install.packages("stringr")

if(!require("logr")) 
  install.packages("logr")

if(!require(prodlim))
  install.packages("prodlim")
if(!require(caret))
  install.packages("caret")

if(!require(randomForest))
  install.packages("randomForest")

if(!require(remotes))
  install.packages("remotes")

if(!require(doParallel))
  install.packages("doParallel")

if(!require(reticulate))
  install.packages("reticulate")

if(!require(keras3)) {
  install.packages("keras3")
  install_keras()
}


# if(!require(tensorflow))
#   remotes::install_github("rstudio/tensorflow")
# 
# library(tensorflow)
# install_tensorflow(envname = "r-tensorflow")

if (!require(pacman)) 
  install.packages("pacman")

if(!require(imager))
  install.packages("imager")

if(!require(magick))
  install.packages("magick")

if(!require(sqldf))
  install.packages("sqldf")

# if(!require())
#   install.packages("")


#load the packages in R

library(matrixStats)
library(dslabs)
library(tidyverse)

library(prodlim)
library(caret)
library(randomForest)

library(logr)
library(reticulate)

library(pacman)

library(tensorflow)
library(keras3)

library(imager)
library(magick)

library(doParallel)

# Importing sqldf
library(sqldf)

v# py_require(python_version = "3.11")
reticulate::install_python(version = "3.11")
p_load(conflicted)

conflict_prefer("shape", "keras3", quiet = TRUE)
conflict_prefer("set_random_seed", "keras3", quiet = TRUE)

# reticulate::virtualenv_remove("r-tensorflow")
# install_tensorflow(extra_packages="pillow")
# install_tensorflow(envname = "r-tensorflow")

tf$constant("Hello TensorFlow!")
tensorflow::tf_version()

N_pcCores <- detectCores() - 1   # it is convention to leave 1 core for the OS
N_pcCores




