# %%%%%########%
# Project Setup
# %%%%%########%

options(timeout = max(1000, getOption("timeout")))
options(expressions = 50000) # Increases nesting limit

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




if(!require(tfdatasets))
  install.packages("tfdatasets")

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

if(!require(pROC))
  install.packages("pROC")

if(!require(cvms))
  install.packages("cvms")

if(!require(ggimage))
  install.packages("ggimage")

if(!require(rsvg))
  install.packages("rsvg")

# if(!require())
#   install.packages("")

## Load Libraries & resolve conflicts ------------------------------------------

# 1. Always load reticulate first to set your Python environment
library(reticulate)

py_require(python_version = "3.11")
py_require()
# Tell reticulate to use this new environment for the rest of your session
use_miniconda("mini.r-tensorflow_py3.11", required = TRUE)
py_require()

# 2. Load the core backend engine
library(tensorflow)

# 3. Load the high-level framework
library(keras3)

# 4. Load the automation and tuning packages last
library(kerastuneR)


library(matrixStats)
library(dslabs)
library(tidyverse)

library(prodlim)
library(caret)
library(randomForest)
library(kernlab)
library(ggplot2)

library(logr)
library(utils)

library(pacman)

# detach("package:keras", unload = TRUE)

library(tfdatasets)

library(imager)
library(magick)
library(abind)
library(pROC)
library(cvms)
library(rsvg)
library(ggimage)

library(doParallel)

# Importing sqldf
#library(sqldf)

p_load(conflicted)

conflict_prefer("train", "caret")
conflict_prefer("shape", "keras3", quiet = TRUE)
conflict_prefer("evaluate", "keras3", quiet = TRUE)
conflict_prefer("set_random_seed", "keras3", quiet = TRUE)

# conflict_prefer("save.image", "base")
conflicts_prefer(base::save.image)

# reticulate::virtualenv_remove("r-tensorflow")
# install_tensorflow(extra_packages="pillow")
# install_tensorflow(envname = "r-tensorflow")

# reticulate::install_python(version = "3.11")
# reticulate::install_python(version = "3.6")

tf$constant("Hello TensorFlow!")
tensorflow::tf_version()

N_pcCores <- detectCores() - 1   # it is convention to leave 1 core for the OS
N_pcCores

## Init Global Variables -------------------------------------------------------

n.img_rows <- 28
n.img_cols <- 28

## Load Logging Helper Functions -----------------------------------------------
log_func_script.file_path <- file.path(support_functions.dir, "logging-helper.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Init Project Directories ----------------------------------------------------

put_log("Root directory for the `R` scripts:
%1", r_scripts.dir)

put_log("Root directory for the support scripts:
%1", support_scripts.dir)

put_log("Root directory for the custom functions definition scripts:
%1", support_functions.dir)

model_scripts.dir <- file.path(r_scripts.dir, "models")
stopifnot(dir.exists(model_scripts.dir))

put_log("Root directory for the project models' scripts:
%1", model_scripts.dir)

models.knn_pca_scripts.dir <- file.path(model_scripts.dir, "kNN+PCA")
stopifnot(dir.exists(models.knn_pca_scripts.dir))

put_log("Root directory for the `kNN+PCA MCC` models' scripts:
%1", models.knn_pca_scripts.dir)

models.rf_scripts.dir <- file.path(model_scripts.dir, "random-forest")
stopifnot(dir.exists(models.rf_scripts.dir))

put_log("Root directory for the `RF MCC` model's scripts:
%1", models.rf_scripts.dir)

models.dl_basic.scripts.dir <- file.path(model_scripts.dir, "dl-basic")
stopifnot(dir.exists(models.dl_basic.scripts.dir))

put_log("Root directory for the Basic DL-Based model's scripts:
%1", models.dl_basic.scripts.dir)

models.cnn_scripts.dir <- file.path(model_scripts.dir, "cnn")
stopifnot(dir.exists(model_scripts.dir))

put_log("Root directory for the `CNN-Based MCC` model's scripts:
%1", models.cnn_scripts.dir)

data.dir <- "data"
raw_data.dir <- file.path(data.dir, "raw")

put_log("Root directory for the raw image data:
%1", raw_data.dir)


raw_data.chars.dir <- file.path(raw_data.dir, "Vaibs.HW-Chars")

put_log("Root directory for the raw image data:
%1", raw_data.chars.dir)

img.train_root.dir <- file.path(raw_data.chars.dir, "Train")

put_log("Root directory for the Train raw image data:
%1", img.train_root.dir)


img.validation_root.dir <- file.path(raw_data.chars.dir, "Validation")

put_log("Root directory for the Validation raw image data:
%1", img.validation_root.dir)


dataset.dir <- file.path(data.dir, "dataset")
if(!dir.exists(dataset.dir))
  dir.create(dataset.dir)

put_log("Root directory for the project Dataset:
%1", dataset.dir)


train.data.dir <- file.path(dataset.dir, "train")
if(!dir.exists(train.data.dir))
  dir.create(train.data.dir)

put_log("Root directory for the Train data:
%1", train.data.dir)


final_test.data.dir <- file.path(dataset.dir, "final_test")
if(!dir.exists(final_test.data.dir))
  dir.create(final_test.data.dir)

put_log("Root directory for the Final Test data:
%1", final_test.data.dir)


models_data.dir <- file.path(data.dir, "models")
if(!dir.exists(models_data.dir))
  dir.create(models_data.dir)

put_log("Root directory for the project models data:
%1", models_data.dir)



dl.keras3.dir <- file.path(models_data.dir, "dl.keras3")

if(!dir.exists(dl.keras3.dir))
  dir.create(dl.keras3.dir)

data.dl.cnn.dir <- file.path(dl.keras3.dir, "cnn")

if(!dir.exists(data.dl.cnn.dir))
  dir.create(data.dl.cnn.dir)


## Init Project Script Paths ---------------------------------------------------
prepare_ds.script.path <- file.path(support_scripts.dir, "prepare-input-data.R")
stopifnot(file.exists(prepare_ds.script.path))

model_visualization.shared.script.path <- file.path(model_scripts.dir, 
                                                    "model-visualization.shared.R")
stopifnot(file.exists(model_visualization.shared.script.path))

knn_pca.tune.script.path <- file.path(models.knn_pca_scripts.dir, 
                                      "1.knn+pca.build&tune.R")
stopifnot(file.exists(knn_pca.tune.script.path))

knn_pca.retrain.best_k.script.path <- file.path(models.knn_pca_scripts.dir, 
                                                "2.knn+pca.re-train.best-k.R")
stopifnot(file.exists(knn_pca.retrain.best_k.script.path))

rf_tuning.script.path <- file.path(models.rf_scripts.dir, "rf-tuning.R")
stopifnot(file.exists(rf_tuning.script.path))

rf_retraining.best_par.script.path <- file.path(models.rf_scripts.dir, 
                                                "rf-retraining.best-par.R")
stopifnot(file.exists(rf_retraining.best_par.script.path))

dl_basic.script.path <- file.path(models.dl_basic.scripts.dir, "dl-basic.R")
stopifnot(file.exists(dl_basic.script.path))

dl_basic.tuner.script.path <- file.path(models.dl_basic.scripts.dir, "dl-basic.tuner.R")
stopifnot(file.exists(dl_basic.tuner.script.path))

cnn_multiclass.script.path <- file.path(models.cnn_scripts.dir, "cnn-mcc.basic.R")
stopifnot(file.exists(cnn_multiclass.script.path))

cnn_multiclass.evaluation.script.path <- file.path(models.cnn_scripts.dir, 
                                                   "cnn-multiclass.evaluation.R")
stopifnot(file.exists(cnn_multiclass.evaluation.script.path))

cnn_binary.r_scripts.dir <- file.path(models.cnn_scripts.dir, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.r_scripts.dir))

cnn_binary.ensemble.script.path <- file.path(models.cnn_scripts.dir, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.script.path))

## Load Common Helper Functions ------------------------------------------------
common_helper.funcs.file_path <- file.path(support_functions.dir, "common-helper.R")


source(common_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)



## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.dir, "data-helper.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Model Helper Functions --------------------------------------------------
model_helper.funcs.file_path <- file.path(support_functions.dir, "models-helper.R")


source(model_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

