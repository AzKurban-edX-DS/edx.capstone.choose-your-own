#%%%%%%%%%%%%%%%%%%%%
# Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------

r_scripts.dir <- "r"
stopifnot(dir.exists(r_scripts.dir))

support_scripts.dir <-  file.path(r_scripts.dir, "support-scripts")
stopifnot(dir.exists(support_scripts.dir))

setup_script.file_path <- file.path(support_scripts.dir, "setup.R")
stopifnot(file.exists(setup_script.file_path))

source(setup_script.file_path, 
       catch.aborts = TRUE,
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
put_log("The Training Set has been saved in the object `x3d.train_set`, 
which contains a training sample stored in the `x.train` variable having the following shape:
%1", capture.output(shape(x3d.train_set$x.train)))
# shape(132912, 28, 28)

x3d.test_set <- split3d.list$test_set
put_log("The Test Set has been saved in the object `x3d.test_set`, 
which contains a testing sample stored in the `x.test` variable having the following shape:
%1", capture.output(shape(x3d.test_set$x.test)))
# shape(33267, 28, 28)

rm(split3d.list)

### Prepare Flatten Datasets ---------------------------------------------------
ds.load_flatten.script.path <- file.path(support_scripts.dir, 
                                         "load-flattened-dataset.R")

stopifnot(file.exists(ds.load_flatten.script.path))

source(ds.load_flatten.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Prepare Flattened Datasets --------------------------------------------------

### Loading Split Flattened Dataset allocated 20% for the Test set (default) ----

open_logfile(".split.20%test.balanced_subset")
start <- put_start_date()

if (!exists("ds.flattened.split_list")) {
  stopifnot(file.exists(my_emnist.split.file_path))
  
  put_log("Loading the Split Flattened Dataset from the backup file...")
  
  ds.flattened.split_list <- readRDS(my_emnist.split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.split.file_path)
}

str(ds.flattened.split_list)

#### Preparing Train Balanced Sample ------------------------
ds.fl <- list()

ds.fl$x.train <- ds.flattened.split_list$train_set$x.train

ds.fl$y.train.groups <- ds.get_classIDs.grouped(ds.fl$x.train)
ds.fl$y.train <- ds.fl$y.train.groups$classID

stopifnot(sum(as.character(ds.fl$y.train) != rownames(ds.fl$x.train)) == 0)

dim(ds.fl$x.train)
#> [1] 132873    784

stopifnot(nrow(ds.fl$x.train) == length(ds.fl$y.train))
# 132873

str(ds.fl)

##### View of the Training Set Grouped by Class ------------------------------------
put_log("The Training Set is balanced by set of Classes:
%1", capture.output(print(ds.fl$y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #        3407
  #  2 $        3407
  #  3 &        3407
  #  4 @        3407
  #  5 0        3407
  #  6 1        3407
  #  7 2        3407
  #  8 3        3407
  #  9 4        3407
  # 10 5        3407
  # 11 6        3407
  # 12 7        3407
  # 13 8        3407
  # 14 9        3407
  # 15 A        3407
  # 16 B        3407
  # 17 C        3407
  # 18 D        3407
  # 19 E        3407
  # 20 F        3407
  # 21 G        3407
  # 22 H        3407
  # 23 I        3407
  # 24 J        3407
  # 25 K        3407
  # 26 L        3407
  # 27 M        3407
  # 28 N        3407
  # 29 P        3407
  # 30 Q        3407
  # 31 R        3407
  # 32 S        3407
  # 33 T        3407
  # 34 U        3407
  # 35 V        3407
  # 36 W        3407
  # 37 X        3407
  # 38 Y        3407
  # 39 Z        3407
  invisible(NULL)
}

#### Preparing Test Balanced Sample --------------------------------------------

ds.fl$x.test <- ds.flattened.split_list$test_set$x.test
ds.fl$x.test.files <- ds.flattened.split_list$test_set$x.files
ds.fl$y.test.groups <- ds.get_classIDs.grouped(ds.fl$x.test)
ds.fl$y.test <- y.test.groups$classID

stopifnot(sum(as.character(ds.fl$y.test) != rownames(ds.fl$x.test)) == 0)

dim(ds.fl$x.test)

stopifnot(nrow(ds.fl$x.test) == length(ds.fl$y.test))
#> [1] 33228

str(ds.fl)

##### View of the Test Set Grouped by Class ------------------------------------
put_log("The Test Set is balanced by set of Classes:
%1", capture.output(print(ds.fl$y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         852
  #  2 $         852
  #  3 &         852
  #  4 @         852
  #  5 0         852
  #  6 1         852
  #  7 2         852
  #  8 3         852
  #  9 4         852
  # 10 5         852
  # 11 6         852
  # 12 7         852
  # 13 8         852
  # 14 9         852
  # 15 A         852
  # 16 B         852
  # 17 C         852
  # 18 D         852
  # 19 E         852
  # 20 F         852
  # 21 G         852
  # 22 H         852
  # 23 I         852
  # 24 J         852
  # 25 K         852
  # 26 L         852
  # 27 M         852
  # 28 N         852
  # 29 P         852
  # 30 Q         852
  # 31 R         852
  # 32 S         852
  # 33 T         852
  # 34 U         852
  # 35 V         852
  # 36 W         852
  # 37 X         852
  # 38 Y         852
  # 39 Z         852  
  invisible(NULL)
}

#### Finalize Preparing Datasets -----------------------------------------------
rm(ds.flattened.split_list)
log_close()

## Build & Tune kNN+PCA Model --------------------------------------------------

knn_pca.tune.script.path <- file.path(models.knn_pca_scripts.dir, 
                                 "1.knn+pca.build&tune.R")

stopifnot(file.exists(knn_pca.tune.script.path))

source(knn_pca.tune.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Re-Train kNN+PCA Model with the Best `k` Value ------------------------------
knn_pca.retrain.best_k.script.path <- file.path(models.knn_pca_scripts.dir, 
                                                "2.knn+pca.re-train.best-k.R")

stopifnot(file.exists(knn_pca.retrain.best_k.script.path))

source(knn_pca.retrain.best_k.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Build Random Forest Model ---------------------------------------------------

random_forest.script.path <- file.path(models.rf_scripts.dir, 
                                       "random-forest.R")

stopifnot(file.exists(random_forest.script.path))

source(random_forest.script.path, 
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


