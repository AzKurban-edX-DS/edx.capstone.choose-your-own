#%%%%%%%%%%%%%%%%%%%%
# Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------

scripts.path <- "r"
stopifnot(dir.exists(scripts.path))

support_scripts.path <-  file.path(scripts.path, "support-scripts")
stopifnot(dir.exists(support_scripts.path))

setup_script.file_path <- file.path(support_scripts.path, "setup.R")
stopifnot(file.exists(setup_script.file_path))

source(setup_script.file_path, 
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

### Prepare Flatten Datasets ---------------------------------------------------
ds.load_flatten.script.path <- file.path(support_scripts.path, 
                                         "load-flattened-dataset.R")

stopifnot(file.exists(ds.load_flatten.script.path))

source(ds.load_flatten.script.path, 
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

## Prepare Flatten Datasets ----------------------------------------------------
### Loading Split Flatten Dataset allocated 10% for the Train Set ---------------
open_logfile(".split.10%train.balanced_subset")
start <- put_start_date()


if (!exists("ds_flatten.0.1split_list")) {
  stopifnot(file.exists(my_emnist.0.1split.file_path))
  
  put_log("Loading the Split Flattened Dataset from the backup file...")
  
  ds_flatten.0.1split_list <- readRDS(my_emnist.0.1split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.0.1split.file_path)
} 

str(ds_flatten.0.1split_list)

#### Preparing Train Balanced Sample --------------------------------------------

x0.1.train.flatten <- ds_flatten.0.1split_list$train_set$x.train

y0.1.train.flatten.groups <- ds.get_classIDs.grouped(x0.1.train.flatten)
y0.1.train.flatten <- y0.1.train.flatten.groups$classID

stopifnot(sum(as.character(y0.1.train.flatten) != rownames(x0.1.train.flatten)) == 0)

str(x0.1.train.flatten)
dim(x0.1.train.flatten)
#> [1] 16653   784

str(y0.1.train.flatten)

stopifnot(nrow(x0.1.train.flatten) == length(y0.1.train.flatten))

##### View of the Train Set Grouped by Class ------------------------------------
put_log("The Train Set is balanced by set of Classes:
%1", capture.output(print(y0.1.train.flatten.groups$groupByClass, n = N.classes)))
{  
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         425
  #  2 $         425
  #  3 &         425
  #  4 @         425
  #  5 0         425
  #  6 1         425
  #  7 2         425
  #  8 3         425
  #  9 4         425
  # 10 5         425
  # 11 6         425
  # 12 7         425
  # 13 8         425
  # 14 9         425
  # 15 A         425
  # 16 B         425
  # 17 C         425
  # 18 D         425
  # 19 E         425
  # 20 F         425
  # 21 G         425
  # 22 H         425
  # 23 I         425
  # 24 J         425
  # 25 K         425
  # 26 L         425
  # 27 M         425
  # 28 N         425
  # 29 P         425
  # 30 Q         425
  # 31 R         425
  # 32 S         425
  # 33 T         425
  # 34 U         425
  # 35 V         425
  # 36 W         425
  # 37 X         425
  # 38 Y         425
  # 39 Z         425
}

#### Preparing Test Balanced Sample ---------------------------------------------

x0.9.test.flatten <- ds_flatten.0.1split_list$test_set$x.test
x0.9.test.flatten.files <- ds_flatten.0.1split_list$test_set$x.files

y0.9.test.flatten.groups <- ds.get_classIDs.grouped(x0.9.test.flatten)
y0.9.test.flatten <- y0.9.test.flatten.groups$classID

stopifnot(sum(as.character(y0.9.test.flatten) != rownames(x0.9.test.flatten)) == 0)

str(x0.9.test.flatten)
dim(x0.9.test.flatten)
# [1] 149526    784

str(y0.9.test.flatten)

stopifnot(nrow(x0.9.test.flatten) == length(y0.9.test.flatten))
#> [1] 149526

##### View of the Test Set Grouped by Class -------------------------------------
put_log("The Test Set is balanced by set of Classes:
%1", capture.output(print(y0.9.test.flatten.groups$groupByClass, n = N.classes)))
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
}

#### Finalize Preparing Datasets ------------------------------------------------
rm(ds_flatten.0.1split_list)
log_close()

### Loading Split Flatten Dataset allocated 20% for the Test set (default) -----

open_logfile(".split.20%test.balanced_subset")
start <- put_start_date()

if (!exists("ds_flatten.split_list")) {
  stopifnot(file.exists(my_emnist.split.file_path))
  
  put_log("Loading the Split Flattened Dataset from the backup file...")
  
  ds_flatten.split_list <- readRDS(my_emnist.split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.split.file_path)
}

str(ds_flatten.split_list)

#### Preparing Train Balanced Sample --------------------------------------------

x.train.flatten <- ds_flatten.split_list$train_set$x.train

y.train.flatten.groups <- ds.get_classIDs.grouped(x.train.flatten)
y.train.flatten <- y.train.flatten.groups$classID

stopifnot(sum(as.character(y.train.flatten) != rownames(x.train.flatten)) == 0)

dim(x.train.flatten)
#> [1] 132873    784

str(x.train.flatten)

str(y.train.flatten)

stopifnot(nrow(x.train.flatten) == length(y.train.flatten))
# 132873

##### View of the Train Set Grouped by Class ------------------------------------
put_log("The Train Set is balanced by set of Classes:
%1", capture.output(print(y.train.flatten.groups$groupByClass, n = N.classes)))
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
}

#### Preparing Test Balanced Sample --------------------------------------------

x.test.flatten <- ds_flatten.split_list$test_set$x.test
x.test.flatten.files <- ds_flatten.split_list$test_set$x.files
y.test.flatten.groups <- ds.get_classIDs.grouped(x.test.flatten)
y.test.flatten <- y.test.flatten.groups$classID

stopifnot(sum(as.character(y.test.flatten) != rownames(x.test.flatten)) == 0)

str(x.test.flatten)
dim(x.test.flatten)

str(y.test.flatten)
stopifnot(nrow(x.test.flatten) == length(y.test.flatten))
#> [1] 33228

##### View of the Test Set Grouped by Class ------------------------------------
put_log("The Test Set is balanced by set of Classes:
%1", capture.output(print(y.test.flatten.groups$groupByClass, n = N.classes)))
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
}

#### Finalize Preparing Datasets ------------------------------------------------
rm(ds_flatten.split_list)
log_close()

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

dl_basic.scripts.path <- file.path(models_script.path, "dl-basic.R")
dl.keras3.path <- file.path(models.path, "dl.keras3")
dir.create(dl.keras3.path)

stopifnot(file.exists(dl_basic.scripts.path))

source(dl_basic.scripts.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## CNN-Based Classifier Models -------------------------------------------------

### Initial Paths --------------------------------------------------------------
data.dl.cnn.dir <- file.path(dl.keras3.path, "cnn")

if(!dir.exists(data.dl.cnn.dir))
  dir.create(data.dl.cnn.dir)

### CNN-based Multiclass Classifier (CNN MCC) Model ----------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

#### Initial Paths -------------------------------------------------------------

open_logfile(".ds.prepare.train&test.balanced_sets")

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

source(cnn_binary.scripts.path, 
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

cnn_binary.ensemble.script.path <- file.path(models.cnn_script.path, 
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


