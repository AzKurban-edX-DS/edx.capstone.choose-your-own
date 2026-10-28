#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Training Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Prepare Training Datasets ---------------------------------------------------
open_logfile(".prepare-train-set")
stopifnot(file.exists(train.img.std_bin.ls.nested.file))

### Preparing Default Split Datasets (of 80%-size Training Set) for NN-Based Models ----

put_log("Preparing Default Split Datasets (of 80%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds.split.train_0.8.backup.path)

start <- put_start_date()

if(!file.exists(ds.split.train_0.8.backup.path)) {
  stopifnot(file.exists(train.img.std_bin.array.file))
  
  put_log("Loading and splitting the Train Image Data Array 
into default Train and Test Sets...")
  
  ds.split.train_0.8 <- 
    ds.split.img_mx.array(train.img.std_bin.array.file,
                           test_ratio = 0.2)
  
  put_log("The Default Split Dataset object structure:
%1", capture.output(str(ds.split.train_0.8)))

  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds.split.train_0.8, 
          file = ds.split.train_0.8.backup.path)
  
  rm(ds.split.train_0.8)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds.split.train_0.8.backup.path)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds.split.train_0.8.backup.path)
}

put_end_date(start)

### Preparing Split Datasets of 10%-size Training Set for NN-Based Models ----

put_log("Preparing Split Datasets of 10%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds.split.train_0.1.file)

start <- put_start_date()

if(!file.exists(ds.split.train_0.1.file)) {
  stopifnot(file.exists(train.img.std_bin.array.file))
  
  put_log("Loading and splitting the Train Image Data Array 
into a Default Train and Test Sets...")
  
  ds.split.train_0.1 <- 
    ds.split.img_mx.array(train.img.std_bin.array.file,
                           test_ratio = 0.9)
  
  put_log("The Default Split Dataset object structure:
%1", capture.output(str(ds.split.train_0.1)))

  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds.split.train_0.1, 
          file = ds.split.train_0.1.file)
  
  rm(ds.split.train_0.1)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds.split.train_0.1.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds.split.train_0.1.file)
}

### Preparing a Split Imbalanced Datasets for Final Retraining of the Best Model ----

put_log("Preparing a (Default) split imbalanced datasets (of 80%-size Training Set) for 
the final retraining ot the Best Model...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds.imbalanced.final_retraining.file)

start <- put_start_date()

if(!file.exists(ds.imbalanced.final_retraining.file)) {
  stopifnot(file.exists(train.img.std_bin.array.file))
  
  put_log("Loading and splitting (without balancing) the array of training images 
into standard training and test sets...")
  
  ds.split_imbalanced <- 
    ds.split.img_mx.array(train.img.std_bin.array.file,
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

ds.train.flatten.file <- file.path(train.data.dir, "ds.flatten.rds")

put_log("The path for the backup file to save the Flattened EMNIST-Like Dataset:
%1", ds.train.flatten.file)


if(!file.exists(ds.train.flatten.file)){
  stopifnot(file.exists(train.img.std_bin.ls.nested.file))
  
  img.std_bin.ls.nested <- readRDS(train.img.std_bin.ls.nested.file)

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  ds.train.flatten <- img.list2flatten_matrix(img.std_bin.ls.nested$img.list)
  rm(img.std_bin.ls.nested)

  put_log("The flatten dataset have been created with the following structure:
  %1", capture.output(str(ds.train.flatten)))

  
  put_log("Saving flatten training dataset to the backup file: 
%1", ds.train.flatten.file)
  saveRDS(ds.train.flatten,
       file = ds.train.flatten.file)
  
  rm(ds.train.flatten)
  
  put_log("The flatten training dataset has been saved to the backup file:
%1", ds.train.flatten.file)
  put_end_date(start)
  
} else {
  put_log("The flatten training dataset has already been constructed 
and backed up to the following file:
%1", ds.train.flatten.file)
}

#### Prepare Split Training Datasets -------------------------------------------

stopifnot(dir.exists(support_scripts.dir))
ds.prepare_flattened.script.path <- file.path(support_scripts.dir, 
                                              "prepare-flattened-datasets.R")

stopifnot(file.exists(ds.prepare_flattened.script.path))

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

