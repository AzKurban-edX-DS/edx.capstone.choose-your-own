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
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Prepare Input Datasets ------------------------------------------------------
stopifnot(file.exists(prepare_ds.script.path))

ds28x28.split.train_0.8.backup.file <- file.path(train.data.dir, 
                                                 "ds28x28.split.train_0.8.backup.rds")

ds28x28.split.train_0.1.backup.file <- file.path(train.data.dir, 
                                                 "ds28x28.split.train_0.1.backup.rds")

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

knn_pca.data.dir = file.path(models_data.dir, "knn-pca")

if(!dir.exists(knn_pca.data.dir))
  dir.create(knn_pca.data.dir)

knn_pca.data.plots.dat.dir <- file.path(knn_pca.data.dir, 
                                          "plots.dat")

if(!dir.exists(knn_pca.data.plots.dat.dir))
  dir.create(knn_pca.data.plots.dat.dir)



### Build & Tune the kNN+PCA Model ---------------------------------------------
stopifnot(file.exists(knn_pca.tune.script.path))

source(knn_pca.tune.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Re-Train kNN+PCA Model with the Best `k` Value ------------------------------
stopifnot(file.exists(knn_pca.retrain.best_k.script.path))

source(knn_pca.retrain.best_k.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Random Forest (RF) Model ----------------------------------------------------
data.models.random_forest.dir <- file.path(models_data.dir, "random-forest")

if(!dir.exists(data.models.random_forest.dir))
  dir.create(data.models.random_forest.dir)

data.models.rf.plots.dat.dir <- file.path(data.models.random_forest.dir, 
                                          "plots.dat")

if(!dir.exists(data.models.rf.plots.dat.dir))
  dir.create(data.models.rf.plots.dat.dir)

### RF Tuning ------------------------------------------------------------------
stopifnot(file.exists(rf_tuning.script.path))

source(rf_tuning.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### RF Re-Training with the Best Parameters ------------------------------------
stopifnot(file.exists(rf_retraining.best_par.script.path))

source(rf_retraining.best_par.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Basic Deep Learning Model ---------------------------------------------------
stopifnot(file.exists(dl_basic.script.path))

data.dl_basic.dir <- file.path(dl.keras3.dir, "dl.basic")

if(!dir.exists(data.dl_basic.dir))
  dir.create(data.dl_basic.dir)

source(dl_basic.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

stopifnot(file.exists(dl_basic.tuner.script.path))

source(dl_basic.tuner.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-based Multiclass Classifier (CNN MCC) Model ----------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

#### Initial Paths -------------------------------------------------------------

open_logfile(".ds.prepare.train&test.balanced_sets")

data.cnn_mcc.dir <- file.path(data.dl.cnn.dir, "multiclass")

if(!dir.exists(data.cnn_mcc.dir))
  dir.create(data.cnn_mcc.dir)

cnn_mcc.x3d.test_set.bakup <- file.path(data.cnn_mcc.dir,
                                        "x3d.test_set.rds")

data.cnn_mcc.basic.dir <- file.path(data.cnn_mcc.dir, "basic")

if(!dir.exists(data.cnn_mcc.basic.dir))
  dir.create(data.cnn_mcc.basic.dir)

cnn_mcc.basic.plots.dat.dir <- file.path(data.cnn_mcc.basic.dir, "plots.dat")

if(!dir.exists(cnn_mcc.basic.plots.dat.dir))
  dir.create(cnn_mcc.basic.plots.dat.dir)

cnn_mcc.basic.file_path <- file.path(data.cnn_mcc.basic.dir, 
                                     "cnn.pre-trained.multiclass.model.keras")
cnn_mcc.basic.train_history.file_path <- file.path(data.cnn_mcc.basic.dir,
                                             "cnn_mcc.train_history.backup.rds")

data.cnn_mcc.tuner.dir <- file.path(data.cnn_mcc.dir, "tuner")

if(!dir.exists(data.cnn_mcc.tuner.dir))
  dir.create(data.cnn_mcc.tuner.dir)

data.cnn_mcc.tuner.best.dir <- file.path(data.cnn_mcc.tuner.dir, "best")

if(!dir.exists(data.cnn_mcc.tuner.best.dir))
  dir.create(data.cnn_mcc.tuner.best.dir)

cnn_mcc.best.plots.dat.dir <- file.path(data.cnn_mcc.tuner.best.dir, "plots.dat")

if(!dir.exists(cnn_mcc.best.plots.dat.dir))
  dir.create(cnn_mcc.best.plots.dat.dir)

cnn_mcc.final.file <- file.path(data.cnn_mcc.tuner.best.dir, 
                                     "cnn_mcc.final-model.keras")

cnn_mcc.final.train_history.file <- file.path(data.cnn_mcc.tuner.best.dir, 
                                              "cnn.mcc-final.train-history.rds")

#### Build Basic CNN MCC Model -------------------------------------------------
stopifnot(file.exists(cnn_mcc.script.path))

put_log("Defining and training a CNN-based Multiclass Classifier Model...")

source(cnn_mcc.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


#### Tuning CNN MCC Model ------------------------------------------------------

if(file.exists(cnn_mcc.final.file)) {
  if(file.exists(cnn_mcc.final.train_history.file)){
    put_log("Loading the tuned Final MCC Model Train History...")
    
    cnn_mcc.final.train_history <- readRDS(cnn_mcc.final.train_history.file)
    
    put_log("The Tuned Final MCC Model Training History has been loaded from the backup file:
%1", cnn_mcc.final.train_history.file)
    
    put_log("The Tuned Final MCC Model Training History Summary:
%1", capture.output(cnn_mcc.final.train_history))
    plot(cnn_mcc.final.train_history)
  } else {
    warning("The tuned Final MCC Model backup does not exist:
", cnn_mcc.final.train_history.file)
  }
  
  put_log("Evaluating the tuned and re-trained Final CNN MCC Model")
  stopifnot(file.exists(cnn_mcc_final.eval.script.path))
  
  # source(cnn_mcc_final.eval.script.path, 
  #        catch.aborts = TRUE,
  #        echo = TRUE,
  #        spaced = TRUE,
  #        verbose = TRUE,
  #        keep.source = TRUE)
} else {
    stopifnot(file.exists(cnn_mcc.model_tuner.script.path))
    
    put_log("Tuning the CNN-based Multiclass Classifier Model...")
    source(cnn_mcc.model_tuner.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
}

log_close()

## CNN-based Binary Classifier Models -----------------------------------------
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

# source(cnn_binary.r_scripts.dir,
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

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
stopifnot(file.exists(cnn_binary.ensemble.script.path))


source(cnn_binary.ensemble.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Final Testing of the CNN-Based Multiclass Classifier Model -----------------
stopifnot(file.exists(cnn_mcc.model.final_test.file_path))

x3d.test_set <- ft.x3d.test_set
rm(ft.x3d.test_set)

cnn_mcc.model.final_test.file_path <- file.path(data.cnn_mcc.dir, 
                                                "cnn.multiclass.model.final-test.RData")
stopifnot(file.exists(cnn_mcc.model.final_test.file_path))
saaaaaaaaaaaaaaaaaaaaaaaaaasssssssss
source(cnn_mcc.model.final_test.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

# Appendix: The Device (laptop) Info Where the Project was Build & Tested ---------------

# Processor	13th Gen Intel(R) Core(TM) i7-13620H (2.40 GHz)
# Installed RAM	32.0 GB (31.7 GB usable)
# System type	64-bit operating system, x64-based processor

# Edition	Windows 11 Pro
# Version	25H2
# Installed on	‎12/‎14/‎2024
# OS build	26200.8973
# Experience	Windows Feature Experience Pack 1000.26100.344.0


