#####################
Main (Index) Script
#####################

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Logging Helper Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-helper.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Load Common Helper Functions --------------------------------------------------
common_helper.funcs.file_path <- file.path(support_functions.path, "common-helper.R")


source(common_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)



## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.path, "data-helper.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Model Helper Functions --------------------------------------------------
model_helper.funcs.file_path <- file.path(support_functions.path, "models-helper.R")


source(model_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Prepare Input Datasets ------------------------------------------------------
prepare_ds.script.path <- file.path(support_scripts.path, "prepare-input-data.R")
stopifnot(file.exists(prepare_ds.script.path))

source(prepare_ds.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## kNN+PCA & Random Forest Models ----------------------------------------------
knn_pca.rf.script.path <- file.path(models_script.path, "knn+pca&rf.R")
stopifnot(file.exists(knn_pca.rf.script.path))

source(knn_pca.rf.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Basic Deep Learning Models --------------------------------------------------
dl_basic.scripts.path <- file.path(models_script.path, "dl-basic.R")
stopifnot(file.exists(dl_basic.scripts.path))

source(dl_basic.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Multi-class Classifier Model ---------------------------------------
cnn_multiclass.scripts.path <- file.path(models.cnn_script.path, 
                                         "cnn-multiclass.draft2.R")
stopifnot(file.exists(cnn_multiclass.scripts.path))

source(cnn_multiclass.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Binary Classifier Models ------------------------------------------

cnn_binary.scripts.path <- file.path(models.cnn_script.path, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.scripts.path))

source(cnn_binary.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Ensemble based on CNN-based Binary Classifier Models ------------------------

cnn_binary.ensemble.scripts.path <- file.path(models.cnn_script.path, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.scripts.path))

source(cnn_binary.ensemble.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
















