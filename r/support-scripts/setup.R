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
#py_require()

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
### `R` Script Directories -----------------------------------------------------

put_log("Root directory for the `R` scripts:
%1", r_scripts.dir)

put_log("Root directory for the support scripts:
%1", support_scripts.dir)

put_log("Root directory for the custom functions definition scripts:
%1", support_functions.dir)

#### Directories for Model Scripts ---------------------------------------------
model_scripts.dir <- file.path(r_scripts.dir, "models")
stopifnot(dir.exists(model_scripts.dir))

put_log("Root directory for the project models' scripts:
%1", model_scripts.dir)

##### Directories for Shallow Learning Model Scripts ---------------------------

models.knn_pca_scripts.dir <- file.path(model_scripts.dir, "knn+pca-mcc")
stopifnot(dir.exists(models.knn_pca_scripts.dir))

put_log("Root directory for the `kNN+PCA MCC` models' scripts:
%1", models.knn_pca_scripts.dir)

models.rf_scripts.dir <- file.path(model_scripts.dir, "rf-mcc")
stopifnot(dir.exists(models.rf_scripts.dir))

put_log("Root directory for the `RF MCC` model's scripts:
%1", models.rf_scripts.dir)

##### Directories for Deep Learning Model Scripts ------------------------------
###### Directories for DNN-Based Scripts ---------------------------------------

models.dnn_mcc.scripts.dir <- file.path(model_scripts.dir, "dnn-mcc")
stopifnot(dir.exists(models.dnn_mcc.scripts.dir))

put_log("Root directory for the Basic DNN-Based model's scripts:
%1", models.dnn_mcc.scripts.dir)

dnn_mcc.basic.scripts.dir <- file.path(models.dnn_mcc.scripts.dir, "basic")
stopifnot(dir.exists(dnn_mcc.basic.scripts.dir))

put_log("Root directory for the Basic DNN-Based model's scripts:
%1", dnn_mcc.basic.scripts.dir)

dnn_mcc.tuner.scripts.dir <- file.path(models.dnn_mcc.scripts.dir, "tuner")
stopifnot(dir.exists(dnn_mcc.tuner.scripts.dir))

put_log("Root directory for the Basic DNN-Based model's scripts:
%1", dnn_mcc.tuner.scripts.dir)

###### Directories for CNN-Based Scripts ---------------------------------------

models.cnn_scripts.dir <- file.path(model_scripts.dir, "cnn")
stopifnot(dir.exists(models.cnn_scripts.dir))

put_log("Root directory for the `CNN-Based` model's scripts:
%1", models.cnn_scripts.dir)
#---

cnn._binary.scripts.dir <- file.path(models.cnn_scripts.dir, "binary")
stopifnot(dir.exists(cnn._binary.scripts.dir))

put_log("Root directory for the `CNN-Based MCC` model's scripts:
%1", cnn._binary.scripts.dir)

cnn_mcc.scripts.dir <- file.path(models.cnn_scripts.dir, "mcc")
stopifnot(dir.exists(cnn_mcc.scripts.dir))

put_log("Root directory for the `CNN-Based MCC` model's scripts:
%1", cnn_mcc.scripts.dir)

cnn_mcc.basic.scripts.dir <- file.path(cnn_mcc.scripts.dir, "basic")
stopifnot(dir.exists(cnn_mcc.basic.scripts.dir))

put_log("Root directory for the `CNN-Based Basic MCC` model's scripts:
%1", cnn_mcc.basic.scripts.dir)

cnn_mcc.tuner_scripts.dir <- file.path(cnn_mcc.scripts.dir, "tuner")
stopifnot(dir.exists(cnn_mcc.tuner_scripts.dir))

put_log("Root directory for the `CNN-Based MCC` model tuner's scripts:
%1", cnn_mcc.tuner_scripts.dir)

### Data Directories -----------------------------------------------------------

data.dir <- "data"

#### Raw Data Directories ------------------------------------------------------
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

#### Directories for Project Datasets ------------------------------------------
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

#### Directories for Model Data ------------------------------------------------
models_data.dir <- file.path(data.dir, "models")
if(!dir.exists(models_data.dir))
  dir.create(models_data.dir)

put_log("Root directory for the project models data:
%1", models_data.dir)

##### Directories for Shallow Learning Model Data ------------------------------
###### Directories for kNN+PCA Model Data --------------------------------------
knn_pca.data.dir = file.path(models_data.dir, "knn+pca-mcc")

if(!dir.exists(knn_pca.data.dir))
  dir.create(knn_pca.data.dir)

knn_pca.data.plots.dat.dir <- file.path(knn_pca.data.dir, 
                                        "plots.dat")

if(!dir.exists(knn_pca.data.plots.dat.dir))
  dir.create(knn_pca.data.plots.dat.dir)

###### Directories for Random Forest Model Data --------------------------------

data.models.rf.dir <- file.path(models_data.dir, "rf-mcc")

if(!dir.exists(data.models.rf.dir))
  dir.create(data.models.rf.dir)

data.models.rf.plots.dat.dir <- file.path(data.models.rf.dir, 
                                          "plots.dat")

if(!dir.exists(data.models.rf.plots.dat.dir))
  dir.create(data.models.rf.plots.dat.dir)

##### Directories for Deep Learning Model Data ----------------------------------

dl.keras3.dir <- file.path(models_data.dir, "dl.keras3")

if(!dir.exists(dl.keras3.dir))
  dir.create(dl.keras3.dir)

###### Directories for DNN-Based Data -------------------------------------------
data.dnn_mcc.dir <- file.path(dl.keras3.dir, "dnn-mcc")

if(!dir.exists(data.dnn_mcc.dir))
  dir.create(data.dnn_mcc.dir)

data.dnn_mcc.basic.dir <- file.path(data.dnn_mcc.dir, "basic")

if(!dir.exists(data.dnn_mcc.basic.dir))
  dir.create(data.dnn_mcc.basic.dir)

dnnb_mcc.plots.dat.dir <- file.path(data.dnn_mcc.basic.dir, "plots.dat")

if(!dir.exists(dnnb_mcc.plots.dat.dir))
  dir.create(dnnb_mcc.plots.dat.dir)

dnn_mcc.tuner.dir <- file.path(data.dnn_mcc.dir,
                                  "tuner")
if(!dir.exists(dnn_mcc.tuner.dir))
  dir.create(dnn_mcc.tuner.dir)

dnn_mcc.tuner.plots.dat.dir <- file.path(dnn_mcc.tuner.dir, "plots.dat")

if(!dir.exists(dnn_mcc.tuner.plots.dat.dir))
  dir.create(dnn_mcc.tuner.plots.dat.dir)


###### Directories for CNN-Based Data -------------------------------------------

data.dl.cnn.dir <- file.path(dl.keras3.dir, "cnn")

if(!dir.exists(data.dl.cnn.dir))
  dir.create(data.dl.cnn.dir)

data.cnn_mcc.dir <- file.path(data.dl.cnn.dir, "mcc")

if(!dir.exists(data.cnn_mcc.dir))
  dir.create(data.cnn_mcc.dir)

data.cnn_mcc.basic.dir <- file.path(data.cnn_mcc.dir, "basic")

if(!dir.exists(data.cnn_mcc.basic.dir))
  dir.create(data.cnn_mcc.basic.dir)

cnnb_mcc.plots.dat.dir <- file.path(data.cnn_mcc.basic.dir, "plots.dat")

if(!dir.exists(cnnb_mcc.plots.dat.dir))
  dir.create(cnnb_mcc.plots.dat.dir)

data.cnn_mcc.tuner.dir <- file.path(data.cnn_mcc.dir, "tuner")

if(!dir.exists(data.cnn_mcc.tuner.dir))
  dir.create(data.cnn_mcc.tuner.dir)

data.cnn_mcc.tuner.best.dir <- file.path(data.cnn_mcc.tuner.dir, "best")

if(!dir.exists(data.cnn_mcc.tuner.best.dir))
  dir.create(data.cnn_mcc.tuner.best.dir)

cnn_mcc.best.plots.dat.dir <- file.path(data.cnn_mcc.tuner.best.dir, "plots.dat")

if(!dir.exists(cnn_mcc.best.plots.dat.dir))
  dir.create(cnn_mcc.best.plots.dat.dir)



## Init Input Data Paths -------------------------------------------------------
my_emnist.split.file_path <- file.path(train.data.dir, "my_emnist.20%test-split.rds")

ds28x28.split.train_0.8.backup.file <- file.path(train.data.dir, 
                                                 "ds28x28.split.train_0.8.backup.rds")

ds28x28.split.train_0.1.backup.file <- file.path(train.data.dir, 
                                                 "ds28x28.split.train_0.1.backup.rds")

my_emnist.split.file_path <- file.path(train.data.dir, "my_emnist.20%test-split.rds")

my_emnist.0.1split.file_path <- file.path(train.data.dir, "my_emnist-split(10%train-set).rds")

## Init Project Script Paths ---------------------------------------------------
prepare_ds.script.path <- file.path(support_scripts.dir, "prepare-input-data.R")
stopifnot(file.exists(prepare_ds.script.path))

model_visualization.shared.script.path <- file.path(model_scripts.dir, 
                                                    "model-visualization.shared.R")
stopifnot(file.exists(model_visualization.shared.script.path))

### Shallow Learning MCC Model -------------------------------------------------
#### kNN+PCA MCC Model-Related Scripts -----------------------------------------

knn_pca.tune.script.path <- file.path(models.knn_pca_scripts.dir, 
                                      "1.knn+pca.build&tune.R")

knn_pca.retrain.best_k.script.path <- file.path(models.knn_pca_scripts.dir, 
                                                "2.knn+pca.re-train.best-k.R")

knn_pca.best.eval.script.path <- file.path(models.knn_pca_scripts.dir, 
                                                "3.knn+pca.best-eval.R")

stopifnot(file.exists((knn_pca.tune.script.path),
                      knn_pca.retrain.best_k.script.path,
                      knn_pca.best.eval.script.path))

#### Random Forest (RF) MCC Model-Related Scripts ------------------------------

rf_tuning.script.path <- file.path(models.rf_scripts.dir, "1.rf-tuning.R")

rf_retraining.best_par.script.path <- file.path(models.rf_scripts.dir, 
                                                "2.rf-retraining.best-par.R")
stopifnot(file.exists(rf_tuning.script.path,
                      rf_retraining.best_par.script.path))

### DNN-Based MCC Scripts ------------------------------------------------------
#### DNN-Based Basic MCC Scripts -----------------------------------------------

dnnb_mcc.script.path <- file.path(dnn_mcc.basic.scripts.dir, "1.dnnb-mcc.R")
stopifnot(file.exists(dnnb_mcc.script.path))

dnnb_mcc.eval.script.path <- file.path(dnn_mcc.basic.scripts.dir, "2.dnnb-mcc.eval.R")
stopifnot(file.exists(dnnb_mcc.eval.script.path))

#### DNN-Based MCC Tuner-Related Scripts ---------------------------------

dnn_mcc.tuner.script.path <- file.path(dnn_mcc.tuner.scripts.dir, 
                                       "1.dnn-mcc.tuner.R")
stopifnot(file.exists(dnn_mcc.tuner.script.path))

tdnn_mcc.final.retrain.script.path <- file.path(dnn_mcc.tuner.scripts.dir, 
                                       "2.tdnn-mcc.final.retrain.R")
stopifnot(file.exists(tdnn_mcc.final.retrain.script.path))

tdnn_mcc.final.eval.script.path <- file.path(dnn_mcc.tuner.scripts.dir, 
                                       "3.tdnn-mcc.final.eval.R")
stopifnot(file.exists(tdnn_mcc.final.eval.script.path))

### CNN-Based Scripts ----------------------------------------------------------
#### CNN-Based Basic MCC Scripts -----------------------------------------------
cnnb_mcc.script.path <- file.path(cnn_mcc.basic.scripts.dir, "1.cnnb-mcc.R")
stopifnot(file.exists(cnnb_mcc.script.path))

cnnb_mcc.eval.script.path <- file.path(cnn_mcc.basic.scripts.dir, 
                                            "2.cnnb-mcc.eval.R")
stopifnot(file.exists(cnnb_mcc.eval.script.path))

#### CNN-Based MCC Tuner-Related Scripts ---------------------------------

cnn_mcc.hypermodel.script.path <- file.path(cnn_mcc.tuner_scripts.dir, 
                                            "cnn-mcc.hyper-model.R")
stopifnot(file.exists(cnn_mcc.hypermodel.script.path))

cnn_mcc.model_tuner.script.path <- file.path(cnn_mcc.tuner_scripts.dir, 
                                            "cnn-mcc.model-tuner.R")
stopifnot(file.exists(cnn_mcc.model_tuner.script.path))

cnn_mcc_final.retrain.script.path <- 
  file.path(cnn_mcc.tuner_scripts.dir, 
            "cnn-mcc.final.retrain.R")

stopifnot(file.exists(cnn_mcc_final.retrain.script.path))

cnn_mcc_final.eval.script.path <- 
  file.path(cnn_mcc.tuner_scripts.dir, 
            "cnn-mcc.final.eval.R")

stopifnot(file.exists(cnn_mcc_final.eval.script.path))

# edx.capstone.choose-your-own/r/models/cnn/mcc/tuner/cnn-mcc.retrain-best.R
stopifnot(file.exists(cnn_mcc.model_tuner.script.path))

cnn_binary.r_scripts.dir <- file.path(cnn._binary.scripts.dir, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.r_scripts.dir))

cnn_binary.ensemble.script.path <- file.path(cnn._binary.scripts.dir, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.script.path))

## Load Common Helper Functions ------------------------------------------------
common_helper.funcs.file_path <- file.path(support_functions.dir, 
                                           "common-helper.R")
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


## Load `CNN_MCC.HyperModel` helper class --------------------------------------
stopifnot(file.exists(cnn_mcc.hypermodel.script.path))

put_log("Initializing `CNN_MCC.HyperModel` helper class for tuning the 
CNN-based Multiclass Classifier Model...")

source(cnn_mcc.hypermodel.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

