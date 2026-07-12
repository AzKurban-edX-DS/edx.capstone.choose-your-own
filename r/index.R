#%%%%%%%%%%%%%%%%%%%%
# Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------

r_scripts.dir <- "r"
stopifnot(dir.exists(r_scripts.dir))

support_scripts.dir <-  file.path(r_scripts.dir, "support-scripts")
stopifnot(dir.exists(support_scripts.dir))

support_functions.dir <- file.path(r_scripts.dir, "support-functions")
stopifnot(dir.exists(support_functions.dir))

setup_script.file_path <- file.path(support_scripts.dir, "setup.R")
stopifnot(file.exists(setup_script.file_path))

source(setup_script.file_path, 
       catch.aborts = TRUE,''
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Prepare Input Datasets ------------------------------------------------------
prepare_ds.script.path <- file.path(support_scripts.dir, "prepare-input-data.R")
stopifnot(file.exists(prepare_ds.script.path))

source(prepare_ds.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Load Flatten Datasets ---------------------------------------------------
ds.prepare_flattened.script.path <- file.path(support_scripts.dir, 
                                         "prepare-flattened-datasets.R")

stopifnot(file.exists(ds.prepare_flattened.script.path))

source(ds.prepare_flattened.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## kNN+PCA MCC Model -----------------------------------------------------------
### Build & Tune the kNN+PCA Model ---------------------------------------------

knn_pca.tune.script.path <- file.path(models.knn_pca_scripts.dir, 
                                 "1.knn+pca.build&tune.R")

stopifnot(file.exists(knn_pca.tune.script.path))

source(knn_pca.tune.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Re-Train kNN+PCA Model with the Best `k` Value ------------------------------
knn_pca.retrain.best_k.script.path <- file.path(models.knn_pca_scripts.dir, 
                                                "2.knn+pca.re-train.best-k.R")

stopifnot(file.exists(knn_pca.retrain.best_k.script.path))

source(knn_pca.retrain.best_k.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Random Forest (RF) Model ----------------------------------------------------

### RF Tuning ------------------------------------------------------------------
rf_tuning.script.path <- file.path(models.rf_scripts.dir, 
                                       "rf-tuning.R")

stopifnot(file.exists(rf_tuning.script.path))

source(rf_tuning.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### RF Re-Training with the Best Parameters ------------------------------------
rf_retraining.best_par.script.path <- file.path(models.rf_scripts.dir, 
                                       "rf-retraining.best-par.R")

stopifnot(file.exists(rf_retraining.best_par.script.path))

source(rf_retraining.best_par.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Basic Deep Learning Model ---------------------------------------------------

dl_basic.script.path <- file.path(models.dl_basic.scripts.dir, "dl-basic.R")
stopifnot(file.exists(dl_basic.script.path))

source(dl_basic.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

dl_basic.tuner.script.path <- file.path(models.dl_basic.scripts.dir, "dl-basic.tuner.R")
stopifnot(file.exists(dl_basic.tuner.script.path))

source(dl_basic.tuner.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-Based Classifier Models -------------------------------------------------

### Initial Paths --------------------------------------------------------------
data.dl.cnn.dir <- file.path(dl.keras3.dir, "cnn")

if(!dir.exists(data.dl.cnn.dir))
  dir.create(data.dl.cnn.dir)

### CNN-based Multiclass Classifier (CNN MCC) Model ----------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

#### Initial Paths -------------------------------------------------------------

open_logfile(".ds.prepare.train&test.balanced_sets")

cnn_multiclass.script.path <- file.path(models.cnn_scripts.dir, 
                                         "cnn-multiclass.R")
stopifnot(file.exists(cnn_multiclass.script.path))

cnn_multiclass.evaluation.script.path <- file.path(models.cnn_scripts.dir, 
                                                   "cnn-multiclass.evaluation.R")
stopifnot(file.exists(cnn_multiclass.evaluation.script.path))

data.dl.cnn.multiclass.dir <- file.path(data.dl.cnn.dir, "multiclass")

if(!dir.exists(data.dl.cnn.multiclass.dir))
  dir.create(data.dl.cnn.multiclass.dir)

data.dl.cnn.multiclass.checkpoints.dir <- file.path(data.dl.cnn.multiclass.dir, "checkpoints")

if(!dir.exists(data.dl.cnn.multiclass.checkpoints.dir))
  dir.create(data.dl.cnn.multiclass.checkpoints.dir)

#### Init File Paths -----------------------------------------------------------

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

cnn_binary.r_scripts.dir <- file.path(models.cnn_scripts.dir, "cnn-binary.R")
stopifnot(file.exists(cnn_binary.r_scripts.dir))

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

source(cnn_binary.r_scripts.dir, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Final Test for the Best Models ----------------------------------------------
### Preparing the Final Test Data ----------------------------------------------

open_logfile(".ds.prepare.final-test.balanced_sets")

put_log("Preparing a Final Test Set for validating the CNN-based Models...")
start <- put_start_date()
stopifnot(file.exists(final_test.img28x28mx.array.file_path))

put_log("Loading the Final Test 28x28 Image Data Array Set from the backup file...")
ft.img_mx.set <- readRDS(final_test.img28x28mx.array.file_path)

put_log("The Final Test 28x28 Image Data Array Set has been loading from the following file:
%1", final_test.img28x28mx.array.file_path)

put_log("The Final Test 28x28 Image Data Set structure:
%1", capture.output(str(ft.img_mx.set)))

##### Creating Final Test Dataset -----------------------------------------------
put_log("Making a balanced sample from the Validation 28x28 Image Data Array...")

set.seed(N.classes)
ft.sample_set <- sample_train_test_sets.x3d(ft.img_mx.set$img28x28mx.array,
                                           ft.img_mx.set$img28x28mx.fpath,
                                           test.ratio = 1)

put_log("The Final Test Set sample has been made from the Validation 28x28 Image Data Array,
which is returned in an object with the following structure:
%1", capture.output(str(ft.sample_set)))
put_end_date(start)

ft.x3d.test_set <- ft.sample_set$test_set
put_log("The Test Set has been saved in the object `ft.x3d.test_set`, 
which contains a testing sample stored in the `x.test` variable having the following shape:
%1", capture.output(shape(ft.x3d.test_set$x.test)))
# shape(4641, 28, 28)

# rm(ft.sample_set)

log_close()

### Final Testing of CNN BCC-Based Ensemble ------------------------------------

cnn_binary.ensemble.script.path <- file.path(models.cnn_scripts.dir, 
                                              "cnn-binary.ensemble.R")
stopifnot(file.exists(cnn_binary.ensemble.script.path))

source(cnn_binary.ensemble.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Final Testing of the CNN-Based Multiclass Classifier Model -----------------

x3d.test_set <- ft.x3d.test_set
rm(ft.x3d.test_set)

cnn_multiclass.model.eval.file_path <- file.path(data.dl.cnn.multiclass.dir, 
                                                 "cnn.multiclass.model.final-test.RData")

source(cnn_multiclass.evaluation.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


