# Setup -----------------------------------------------------------------------
options(timeout = max(300, getOption("timeout")))
## Install Packages ------------------------------------------------------------
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

if(!require(kernlab))
  install.packages("kernlab")

if(!require(utils))
  install.packages("utils")

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

if(!require(abind))
  install.packages("abind")

# if(!require())
#   install.packages("")

## Load Libraries & resolve conflicts ------------------------------------------

library(matrixStats)
library(dslabs)
library(tidyverse)

library(prodlim)
library(caret)
library(randomForest)
library(kernlab)

library(logr)
library(utils)
library(reticulate)

library(pacman)

# detach("package:keras", unload = TRUE)
library(tensorflow)
library(keras3)
library(tfdatasets)

library(imager)
library(magick)
library(abind)

library(doParallel)

# Importing sqldf
#library(sqldf)

p_load(conflicted)

conflict_prefer("shape", "keras3", quiet = TRUE)
conflict_prefer("set_random_seed", "keras3", quiet = TRUE)

# reticulate::virtualenv_remove("r-tensorflow")
# install_tensorflow(extra_packages="pillow")
# install_tensorflow(envname = "r-tensorflow")

# reticulate::install_python(version = "3.11")

tf$constant("Hello TensorFlow!")
tensorflow::tf_version()

N_pcCores <- detectCores() - 1   # it is convention to leave 1 core for the OS
N_pcCores

## Initial Paths ---------------------------------------------------------------
r.path <- "r"

draft_scripts.path <- file.path(r.path, "draft")
stopifnot(dir.exists(draft_scripts.path))
draft_scripts.path

scripts.path <- draft_scripts.path
stopifnot(dir.exists(scripts.path))
scripts.path

models_script.path <- file.path(scripts.path, "models")
stopifnot(dir.exists(models_script.path))
models_script.path

models.cnn_script.path <- file.path(models_script.path, "cnn")
stopifnot(dir.exists(models_script.path))
models.cnn_script.path

support_functions.folder <- "support-functions"
support_scripts.folder <- "support-scripts"

support_scripts.path <- file.path(r.path, support_scripts.folder)
stopifnot(dir.exists(support_scripts.path))
support_functions.path <- file.path(r.path, support_functions.folder)
stopifnot(dir.existssupport_functions.path())
#stopifnot(dir.exists())

setup_script.file_path <- file.path(support_scripts.path, "setup.R")

data.path <- "data"
raw_data.path <- file.path(data.path, "raw")
raw_data.path

raw_data.folder_name <- "Vaibs.HW-Chars"
raw_data.chars.path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.chars.path

img.train.root_path <- file.path(raw_data.chars.path, "Train")
img.train.root_path

img.validation.root_path <- file.path(raw_data.chars.path, "Validation")
img.validation.root_path

dataset.path <- file.path(data.path, "dataset")
dir.create(dataset.path)
dataset.path

train.data.path <- file.path(dataset.path, "train")
dir.create(train.data.path)
train.data.path

final_test.data.path <- file.path(dataset.path, "final_test")
dir.create(final_test.data.path)
final_test.data.path

ds.subsets.path <- file.path(train.data.path, "subsets")
dir.create(ds.subsets.path)
ds.subsets.path

models.path <- file.path(data.path, "models")
dir.create(models.path)
models.path

models.random_forest.path <- file.path(models.path, "random-forest")
dir.create(models.random_forest.path)
models.random_forest.path

models.random_forest.research.path <- file.path(models.random_forest.path, "research")
dir.create(models.random_forest.research.path)
models.random_forest.research.path

knn_pca.path = file.path(models.path, "knn-pca")
if(!dir.exists(knn_pca.path)) {
  dir.create(knn_pca.path)
}


