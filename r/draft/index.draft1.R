#%%%%%%%%%%%%%%%%%%%%
Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

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

## Load Flatten Dataset --------------------------------------------------------
# load_flatten_dataset.script.path <- file.path(support_scripts.path, "load-flatten-dataset.R")
# stopifnot(file.exists(load_flatten_dataset.script.path))
# 
# source(load_flatten_dataset.script.path, 
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

## Build kNN+PCA & Random Forest Models ----------------------------------------

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

#### Load Dataset for CNN-Based Models -----------------------------------------
# load.cnn_dataset.script_path <- file.path(support_scripts.path, "load-cnn-dataset.R")
# stopifnot(file.exists(load.cnn_dataset.script_path))
# 
# source(load.cnn_dataset.script_path, 
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

## CNN-based Multi-class Classifier Model ---------------------------------------

### Init CNN-Based Multi-class Model Directories -------------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks
cnn_multiclass.scripts.path <- file.path(models.cnn_script.path, 
                                         "cnn-multiclass.draft3.R")
stopifnot(file.exists(cnn_multiclass.scripts.path))

cnn.train.data.path <- file.path(dl.keras3.path, "cnn")

if(!dir.exists(cnn.train.data.path))
  dir.create(cnn.train.data.path)

cnn.eval.cache.path <- file.path(cnn.train.data.path, "evaluation")

if(!dir.exists(cnn.eval.cache.path))
  dir.create(cnn.eval.cache.path)

cnn.callbacks.path <- file.path(cnn.train.data.path, "callbacks")

if(!dir.exists(cnn.callbacks.path))
  dir.create(cnn.callbacks.path)

cnn.callbacks.checkpoints.path <- file.path(cnn.callbacks.path, "checkpoints")

if(!dir.exists(cnn.callbacks.checkpoints.path))
  dir.create(cnn.callbacks.checkpoints.path)

cnn.callbacks.tensorboard.path <- file.path(cnn.callbacks.path, "tensorboard")

if(!dir.exists(cnn.callbacks.tensorboard.path))
  dir.create(cnn.callbacks.tensorboard.path)

cnn.callbacks.tb_logs.path <- file.path(cnn.callbacks.tensorboard.path, "logs")

if(!dir.exists(cnn.callbacks.tb_logs.path))
  dir.create(cnn.callbacks.tb_logs.path)

input_shape <- c(dim.x_cnn[2], dim.x_cnn[3], 1)
input_shape

### Build CNN-Based Multi-class Model ------------------------------------------
cnn.multiclass.model.file_path <- file.path(cnn.train.data.path, 
                                            "cnn.pre-trained.multiclass.model.keras")

source(cnn_multiclass.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Binary Classifier Models ------------------------------------------
### Init CNN-Based Binary Models Directories -----------------------------------
cnn_binary.scripts.path <- file.path(models.cnn_script.path, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.scripts.path))

cnn.lbl_models.cache.path <- file.path(cnn.train.data.path, "lbl-models")

if(!dir.exists(cnn.lbl_models.cache.path))
  dir.create(cnn.lbl_models.cache.path)

## Build CNN-based Binary Classifier Models ------------------------------------
source(cnn_binary.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Ensemble based on CNN-based Binary Classifier Models ------------------------
### Init CNN-Based Ensemble Model Directories -------------------------------------------
cnn_binary.ensemble.scripts.path <- file.path(models.cnn_script.path, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.scripts.path))
cnn_binary.ensemble.scripts.path

cnn_models.ensemble.cache_file.path <- file.path(cnn.train.data.path,
                                                 "cnn.lbl-models.ensemble.RData")
cnn_models.ensemble.cache_file.path

### Build CNN-Based Ensemble Model ---------------------------------------------
source(cnn_binary.ensemble.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
