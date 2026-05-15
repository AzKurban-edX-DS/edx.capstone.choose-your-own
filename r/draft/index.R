#%%%%%%%%%%%%%%%%%%%%
Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

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
stopifnot(dir.exists(support_functions.path))
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

# ds.subsets.path <- file.path(train.data.path, "subsets")
# dir.create(ds.subsets.path)
# ds.subsets.path

models.path <- file.path(data.path, "models")
dir.create(models.path)
models.path

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


### Deep Learning Models-related paths ----------------------------------------
dl_basic.scripts.path <- file.path(models_script.path, "dl-basic.R")
dl.keras3.path <- file.path(models.path, "dl.keras3")
dir.create(dl.keras3.path)

#### CNN-Based Directories Paths -----------------------------------------------

data.dl.cnn.dir <- file.path(dl.keras3.path, "cnn")

if(!dir.exists(data.dl.cnn.dir))
  dir.create(data.dl.cnn.dir)


##### CNN-Based Multiclass Classifier Model Directories ------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

cnn_multiclass.script.path <- file.path(models.cnn_script.path, 
                                         "cnn-multiclass.R")
stopifnot(file.exists(cnn_multiclass.script.path))

cnn_multiclass.evaluation.script.path <- file.path(models.cnn_script.path, 
                                                   "cnn-multiclass.evaluation.R")
stopifnot(file.exists(cnn_multiclass.evaluation.script.path))

data.dl.cnn.multiclass.dir <- file.path(data.dl.cnn.dir, "multiclass")

if(!dir.exists(data.dl.cnn.multiclass.dir))
  dir.create(data.dl.cnn.multiclass.dir)

data.dl.cnn.multiclass.checkpoints.dir <- file.path(data.dl.cnn.multiclass.dir, "checkpoints")

if(!dir.exists(data.dl.cnn.multiclass.checkpoints.dir))
  dir.create(data.dl.cnn.multiclass.checkpoints.dir)


##### CNN-Based Binary Models Directories -----------------------------------
cnn_binary.scripts.path <- file.path(models.cnn_script.path, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.scripts.path))

data.cnn.binary.dir <- file.path(data.dl.cnn.dir, "binary")

if(!dir.exists(data.cnn.binary.dir))
  dir.create(data.cnn.binary.dir)


data.cnn.binary.models.dir <- file.path(data.cnn.binary.dir, "models")

if(!dir.exists(data.cnn.binary.models.dir))
  dir.create(data.cnn.binary.models.dir)

data.cnn.binary.models.checkpoints.dir <- file.path(data.cnn.binary.models.dir, 
                                                    "checkpoints")
if(!dir.exists(data.cnn.binary.models.checkpoints.dir))
  dir.create(data.cnn.binary.models.checkpoints.dir)

data.cnn.binary.models.evaluation.dir <- file.path(data.cnn.binary.models.dir, 
                                                   "evaluation")
if(!dir.exists(data.cnn.binary.models.evaluation.dir))
  dir.create(data.cnn.binary.models.evaluation.dir)

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

## CNN-based Multiclass Classifier Model ---------------------------------------
open_logfile(".ds.prepare.train&test.balanced_sets")
## Prepare Train+Test Data -----------------------------------------------------
put_log("Preparing Train and Test Sets for training a CNN-based Multiclass Classifier Model...")
start <- put_start_date()

if(!exists("img28x28mx.array")) {
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  put_log("Loading the Train 28x28 Image Data Array from the backup file...")
  img28x28mx.array <- readRDS(train.img28x28mx.array.file_path)
  put_log("The Train 28x28 Image Data Array has been loaded from the following backup file:
%1", train.img28x28mx.array.file_path)
}

put_log("The Train Test Data has the following structure:
%1", capture.output(str(img28x28mx.array)))

#### Split Dataset -------------------------------------------------------------
sample_seed <- length(y.labels) # 39

put_log("Splitting the Train 28x28 Image Data Array into a Train and Test Sets...")

set.seed(sample_seed)
split3d.list <- sample_train_test_sets.x3d(img28x28mx.array)

put_log("The Train 28x28 Image Data Array has been split into a Train and Test Sets,
which are returned in a list object with the following structure:
%1", capture.output(str(split3d.list)))
put_end_date(start)

x3d.train <- split3d.list$train_set
put_log("The Train Set has been saved in the object `x3d.train` with the following shape:
%1", capture.output(shape(x3d.train)))
# shape(132912, 28, 28)

x3d.test <- split3d.list$test_set
put_log("The Test Set has been saved in the object `x3d.test` with the following shape:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

rm(split3d.list)
### Close Log ------------------------------------------------------------------
log_close()

### Build CNN-Based Multiclass Classifier Model --------------------------------

source(cnn_multiclass.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Evaluate pre-trained CNN-Based Multiclass Classifier Model -----------------

source(cnn_multiclass.evaluation.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Binary Classifier Models ------------------------------------------
## Build & Train a CNN-based Binary Classifier Models --------------------------
source(cnn_binary.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Final Test for the CNN-Based Classifier Models ------------------------------
open_logfile(".ds.prepare.final-test.balanced_sets")
### Preparing the Final Test Data -------------------------------------------
put_log("Preparing a Final Test Set for validating the CNN-based Models...")
start <- put_start_date()

if(!exists("ft.img28x28mx.array")) {
  stopifnot(file.exists(final_test.img28x28mx.array.file_path))
  ft.img28x28mx.array <- readRDS(final_test.img28x28mx.array.file_path)
  put_log("The Final Test Data has been loaded from the following backup file:
%1", final_test.img28x28mx.array.file_path)
}

put_log("The Final Test Data has the following structure:
%1", capture.output(str(ft.img28x28mx.array)))

#### Creating Final Test Dataset -----------------------------------------------
final_sample_seed <- length(y.labels) + 1 # 40

put_log("Making a balanced sample from the Validation 28x28 Image Data Array...")

set.seed(final_sample_seed)
#ft.sample_set <- sample_train_test_sets.x3d(img28x28mx.array)
ft.sample_set <- sample_train_test_sets.x3d(ft.img28x28mx.array, test.ratio = 1)

put_log("The Final Test Set sample has been made from the Validation 28x28 Image Data Array,
which is returned in an object with the following structure:
%1", capture.output(str(ft.sample_set)))
put_end_date(start)

# x3d.train <- ft.sample_set$train_set
# put_log("The Train Set has been saved in the object `x3d.train` with the following shape:
# %1", capture.output(shape(x3d.train)))
# # shape(132912, 28, 28)

x3d.test <- ft.sample_set$test_set
put_log("The Test Set has been saved in the object `x3d.test` with the following shape:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

rm(ft.sample_set)
### Close Log ------------------------------------------------------------------
log_close()

### Final Test for pre-trained CNN-Based Multiclass Classifier Model -----------

source(cnn_multiclass.evaluation.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Ensemble based on CNN-based Binary Classifier Models ------------------------
### Init CNN-Based Ensemble Model Directories ----------------------------------
cnn_binary.ensemble.scripts.path <- file.path(models.cnn_script.path, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.scripts.path))
cnn_binary.ensemble.scripts.path

cnn_models.ensemble.cache_file.path <- file.path(cnn.train.data.path,
                                                 "cnn.lbl-models.ensemble.RData")
cnn_models.ensemble.cache_file.path

### Final Test for CNN-Based Ensemble Model ------------------------------------
source(cnn_binary.ensemble.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
