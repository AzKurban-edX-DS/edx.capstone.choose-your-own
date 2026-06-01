#%%%%%%%%%%%%%%%%%%%%
Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

## Initial Paths ---------------------------------------------------------------
# r.path <- "r"
# draft_scripts.path <- file.path(r.path, "draft")
# support_scripts.folder <- "support-scripts"
# support_scripts.path <-  file.path(r.path, support_scripts.folder)

support_scripts.path <-  "r/support-scripts"# file.path(r.path, support_scripts.folder)
stopifnot(dir.exists(support_scripts.path))

setup_script.file_path <- file.path(support_scripts.path, "setup.R")
stopifnot(file.exists(setup_script.file_path))

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

### Load Flatten Dataset --------------------------------------------------------
ds.load_flatten.script.path <- file.path(support_scripts.path, 
                                         "load-flattened-dataset.R")

stopifnot(file.exists(ds.load_flatten.script.path))

source(ds.load_flatten.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Build kNN+PCA & Random Forest Models ----------------------------------------

knn_pca.rf.script.path <- file.path(models_script.path, "knn+pca&rf.R")
stopifnot(file.exists(knn_pca.rf.script.path))

source(knn_pca.rf.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Basic Deep Learning Model --------------------------------------------------
stopifnot(file.exists(dl_basic.scripts.path))

source(dl_basic.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Classifier Models -------------------------------------------------
### CNN-based Multiclass Classifier (CNN MCC) Model ---------------------------------------
open_logfile(".ds.prepare.train&test.balanced_sets")
#### Prepare Training & Testing Sets -----------------------------------------------------
put_log("Preparing Train and Test Sets for training a CNN-based Multiclass Classifier Model...")

start <- put_start_date()
stopifnot(file.exists(train.img28x28mx.array.file_path))
put_log("Loading the Train 28x28 Image Data Array Set from the backup file...")
img_mx.set <- readRDS(train.img28x28mx.array.file_path)
put_log("The Train 28x28 Image Data Array Set has been loading from the following file:
%1", train.img28x28mx.array.file_path)
put_log("The Train 28x28 Image Data Array Set structure:
%1", capture.output(str(img_mx.set)))

put_log("Splitting the Train 28x28 Image Data Array into a Train and Test Sets...")


set.seed(N.classes)
split3d.list <- sample_train_test_sets.x3d(img_mx.set$img28x28mx.array,
                                           img_mx.set$img28x28mx.fpath)
str(split3d.list)

x3d.train_set <- split3d.list$train_set
put_log("The Train Set has been saved in the object `x3d.train_set`, 
which contains a training sample stored in the `x.train` variable having the following shape:
%1", capture.output(shape(x3d.train_set$x.train)))
# shape(132912, 28, 28)

x3d.test_set <- split3d.list$test_set
put_log("The Test Set has been saved in the object `x3d.test_set`, 
which contains a testing sample stored in the `x.test` variable having the following shape:
%1", capture.output(shape(x3d.test_set$x.test)))
# shape(33267, 28, 28)

rm(split3d.list)

#### Init File Paths -------------------------------------------------------------

put_log("Defining and training a CNN-based Multiclass Classifier Model...")

stopifnot(exists("x3d.train_set"))


cnn_multiclass.model.file_path <- file.path(data.dl.cnn.multiclass.dir, 
                                            "cnn.pre-trained.multiclass.model.keras")
cnn_multiclass.train_history.file_path <- file.path(data.dl.cnn.multiclass.dir,
                                                    "cnn_multiclass.train_history.backup.rds")

if(!dir.exists(data.dl.cnn.multiclass.checkpoints.dir))
  dir.create(data.dl.cnn.multiclass.checkpoints.dir)

cnn_multiclass.checkpoint.file_path <- 
  file.path(data.dl.cnn.multiclass.checkpoints.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

log_close()

#### Build CNN-Based Multiclass Classifier (CNN MCC) Model ---------------------

source(cnn_multiclass.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

#### Evaluate pre-trained CNN-Based Multiclass Classifier Model -----------------

source(cnn_multiclass.evaluation.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### CNN-based Binary Classifier Models -----------------------------------------
source(cnn_binary.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Final Test for the CNN-Based Classifier Models -----------------------------
#### Preparing the Final Test Data -------------------------------------------

open_logfile(".ds.prepare.final-test.balanced_sets")

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

##### Creating Final Test Dataset -----------------------------------------------
final_sample_seed <- nrow(ft.img28x28mx.array) # 22524

put_log("Making a balanced sample from the Validation 28x28 Image Data Array...")

set.seed(final_sample_seed)
ft.sample_set <- sample_train_test_sets.x3d(ft.img28x28mx.array, test.ratio = 1)

put_log("The Final Test Set sample has been made from the Validation 28x28 Image Data Array,
which is returned in an object with the following structure:
%1", capture.output(str(ft.sample_set)))
put_end_date(start)

x3d.test <- ft.sample_set$test_set
put_log("The Test Set has been saved in the object `x3d.test` with the following shape:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

rm(ft.sample_set)

log_close()

#### Final Test for pre-trained CNN-Based Multiclass Classifier Model -----------

source(cnn_multiclass.evaluation.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


#### Ensemble based on CNN-based Binary Classifier Models ----------------------
##### Init CNN-Based Ensemble Model Directories --------------------------------
cnn_binary.ensemble.scripts.path <- file.path(models.cnn_script.path, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.scripts.path))
cnn_binary.ensemble.scripts.path

cnn_models.ensemble.cache_file.path <- file.path(cnn.train.data.path,
                                                 "cnn.lbl-models.ensemble.RData")
cnn_models.ensemble.cache_file.path

##### Final Test for CNN-Based Ensemble Model ----------------------------------
source(cnn_binary.ensemble.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
