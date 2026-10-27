#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Training Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Prepare Training Datasets ---------------------------------------------------
open_logfile(".prepare-train-set")
stopifnot(file.exists(img.std_bin.ls.nested.file))

### Preparing Default Split Datasets (of 80%-size Training Set) for NN-Based Models ----

put_log("Preparing Default Split Datasets (of 80%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds28x28.split.train_0.8.file)

start <- put_start_date()

if(!file.exists(ds28x28.split.train_0.8.file)) {
  stopifnot(file.exists(img.std_bin.array.file))
  
  put_log("Loading and splitting the Train 28x28 Image Data Array 
into default Train and Test Sets...")
  
  ds28x28.split.train_0.8 <- 
    split.img28x28mx_array(img.std_bin.array.file,
                           test_ratio = 0.2)
  
  put_log("The Default Split Dataset object structure:
%1", capture.output(str(ds28x28.split.train_0.8)))

  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds28x28.split.train_0.8, 
          file = ds28x28.split.train_0.8.file)
  
  rm(ds28x28.split.train_0.8)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds28x28.split.train_0.8.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds28x28.split.train_0.8.file)
}

put_end_date(start)

### Preparing Split Datasets of 10%-size Training Set for NN-Based Models ----

put_log("Preparing Split Datasets of 10%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds28x28.split.train_0.1.file)

start <- put_start_date()

if(!file.exists(ds28x28.split.train_0.1.file)) {
  stopifnot(file.exists(img.std_bin.array.file))
  
  put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")
  
  ds28x28.split.train_0.1 <- 
    split.img28x28mx_array(img.std_bin.array.file,
                           test_ratio = 0.9)
  
  put_log("The Default Split Dataset object structure:
%1", capture.output(str(ds28x28.split.train_0.1)))

  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds28x28.split.train_0.1, 
          file = ds28x28.split.train_0.1.file)
  
  rm(ds28x28.split.train_0.1)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds28x28.split.train_0.1.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds28x28.split.train_0.1.file)
}

### Preparing a Split Imbalanced Datasets for Final Retraining of the Best Model ----

put_log("Preparing a (Default) split imbalanced datasets (of 80%-size Training Set) for 
the final retraining ot the Best Model...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds.imbalanced.final_retraining.file)

start <- put_start_date()

if(!file.exists(ds.imbalanced.final_retraining.file)) {
  stopifnot(file.exists(img.std_bin.array.file))
  
  put_log("Loading and splitting (without balancing) the array of training images (28x28) 
into standard training and test sets...")
  
  ds.split_imbalanced <- 
    split.img28x28mx_array(img.std_bin.array.file,
                           test_ratio = 0.2,
                           balanced = FALSE)
  
  put_log("The Split Imbalanced Dataset object structure:
%1", capture.output(str(ds.split_imbalanced)))
  
  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds.split_imbalanced, 
          file = ds.imbalanced.final_retraining.file)
  
  rm(ds.split_imbalanced)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds.imbalanced.final_retraining.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds.imbalanced.final_retraining.file)
}

put_end_date(start)

### Preparing the Flattened EMNIST-Like Datasets -------------------------------

put_log("Preparing the Flattened EMNIST-Like Dataset for the DL-Based Basic and Non-NN-Based Models...")

my_emnist.file <- file.path(train.data.dir, "my_emnist.rds")

put_log("The path for the backup file to save the Flattened EMNIST-Like Dataset:
%1", my_emnist.file)


if(!file.exists(my_emnist.file)){
  if(!exists("img.std_bin.ls.nested")) {
    stopifnot(file.exists(img.std_bin.ls.nested.file))
    img.std_bin.ls.nested <- readRDS(img.std_bin.ls.nested.file)
  }
  
  Y.Labels <- img.std_bin.ls.nested$label.list

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  my_emnist.set <- img.list2flatten_matrix(img.std_bin.ls.nested$img.list)
  rm(img.std_bin.ls.nested)

  put_log("The flatten dataset have been created with the following structure:
  %1", capture.output(str(my_emnist.set)))

  
  put_log("Saving flatten training dataset to the backup file: 
%1", my_emnist.file)
  saveRDS(my_emnist.set,
       file = my_emnist.file)
  
  rm(my_emnist.set)
  
  put_log("The flatten training dataset has been saved to the backup file:
%1", my_emnist.file)
  put_end_date(start)
  
} else {
  put_log("The flatten training dataset has already been constructed 
and backed up to the following file:
%1", my_emnist.file)
}

#### Load Flatten Dataset ------------------------------------------------------

ds.load_flattened.script.path <- file.path(support_scripts.dir, 
                                           "load-flattened-dataset.R")

stopifnot(file.exists(ds.load_flattened.script.path))

source(ds.load_flattened.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

stopifnot(exists("my_emnist.set"))
my_emnist.seed <- nrow(my_emnist.set$img.mx)

#### Prepare Split Training Datasets -------------------------------------------

stopifnot(dir.exists(support_scripts.dir))
ds.prepare_flattened.script.path <- file.path(support_scripts.dir, 
                                              "prepare-flattened-datasets.R")

stopifnot(file.exists(ds.prepare_flattened.script.path,
                      ds.load_flattened.script.path))

source(ds.prepare_flattened.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

put_end_date(start)
log_close()
# =========================================================================
# Log End Time: 2026-10-13 12:39:50.727976
# Log Elapsed Time: 0 00:06:20
# =========================================================================

