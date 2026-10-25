#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Training Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Transforming into a List of Binary standardized Image Data Objects ----------

open_logfile(".prepare-bin.train-dat")
start <- put_start_date()


put_log("Preparing a List of Binary standardized Image Data Objects...")

train.img_bin.mx_list.file <- file.path(train.data.dir, 
                                              "train.img_bin.mx_list.rds")

put_log("The path for the backup file to save the Binary standardized Image Object list:
%1", train.img_bin.mx_list.file)

if (!file.exists(train.img_bin.mx_list.file)) {
  stopifnot(file.exists(train.img_dat.list.file))
  
  put_log("Loading list of Image objects...")
  img_dat.list <- readRDS(train.img_dat.list.file)
  put_log("The list of Image objects has been loading from the following file:
%1", train.img_dat.list.file)

### Define the Standard Image Size Geometry ------------------------------------
  
  x <- lapply(img_dat.list$img.list, function(ch.class) {
    img_dat.ls <- ch.class$img.list
    x <- lapply(img_dat.ls, function(img_dat) {
      dim(img_dat[1,,])
    })
  }) |> unlist() |>
    matrix(ncol = 2, 
           byrow = T, 
           dimnames = list(NULL, 
                           c('width', 'height')))
  
  put_log("Structure of the Image Size Array:
%1", capture.output(str(x)))
  
  y <- as.data.frame(x)
  
  y.grouped <- y |> group_by(width) |>
    summarise(min.height = min(height))
  
  print(y.grouped)
  
  y.size <- y.grouped$width * y.grouped$min.height
  img.min_size <- y.grouped[which.min(y.size),]
  
  ds.img_min.geometry  <- paste0(img.min_size$width,
                                 'x',
                                 img.min_size$min.height,
                                 '!')[1]
  #rm(x,y)
  
  put_log("The geometry for standardizing images to a uniform size, 
based on the smallest existing image, is as follows: 
%1", ds.img_min.geometry)
  
### Standardize the Image Data -------------------------------------------------    
  
  put_log("Creating a Standardized Image Binary Data list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train_root.dir)
  
  start <- put_start_date()
  
  train.img_bin.mx_list <- 
    ds.standardize.img_dat(img_dat.list,
                           function(image.dat) {
                             image.dat2bin_mx(image.dat,
                                              ds.img_min.geometry, # '28x28!'
                                              trim_image = TRUE)
                           })
  
  put_log("The Standardized Image Binary Data list has been created 
with the following structure:
%1", capture.output(str(train.img_bin.mx_list)))
  put_end_date(start)

  put_log("Saving Standardized Image Binary Data list to the backup file...")
  saveRDS(train.img_bin.mx_list,
       file = train.img_bin.mx_list.file)
  put_log("Standardized Image Binary Data list has been saved to the following file:
%1", train.img_bin.mx_list.file)
  put_end_date(start)
  
} else {
    put_log("The Standardized Image Binary Data list has already been constructed 
and backed up to the following file:
%1", train.img_bin.mx_list.file)
}

log_close()
# =========================================================================
# Log End Time: 2026-10-08 07:37:53.041816
# Log Elapsed Time: 0 00:24:25
# =========================================================================

## Prepare Training Datasets ---------------------------------------------------
open_logfile(".prepare-train-set")
stopifnot(file.exists(train.img_bin.mx_list.file))

### Preparing the Multiclass Classifier Class Label List -----------------------
start <- put_start_date()

put_log("Preparing the Project Multiclass Classifier Class Label List...")

classifier.label_list.file_path <- file.path(dataset.dir, 
                                             "classifier.label-list.rds")

put_log("The path for the backup file to save the list of the Multiclass Classifier Class Labels:
%1", classifier.label_list.file_path)

if(!file.exists(classifier.label_list.file_path)){
  if(!exists("train.img_bin.mx_list")) {
    put_log("Loading the Standardized Image Binary Data list from the backup file...")
    train.img_bin.mx_list <- readRDS(train.img_bin.mx_list.file)
    put_log("The Standardized Image Binary Data list has been loaded from the following file:
%1", train.img_bin.mx_list.file)
  }
 
  Y.Labels <- train.img_bin.mx_list$label.list
  
  put_log("Saving Classifier Label List to the backup file...")
  saveRDS(Y.Labels,
          file = classifier.label_list.file_path)
  put_log("The Classifier Label List has been saved to the following file:
%1", classifier.label_list.file_path)
  put_end_date(start)
} else {
  put_log("The Classifier Label List has already been extracted 
and backed up to the following file:
%1", classifier.label_list.file_path)
  
  if(!exists("Y.Labels"))
    Y.Labels <- readRDS(classifier.label_list.file_path)
}

N.classes <- length(Y.Labels)

# put_log("The Classifier Label List contains the following labels:
# %1", capture.output(Y.Labels))
put_log("The Classifier Handwritten Character List contains the following labels:
%1", Y.Labels, .sep = " ")

### Preparing a List of Binary standardized Image  Matrix ----------------------

put_log("Preparing a List of the Binary standardized Image  Matrix...")

img.bin_mx.list.file <- file.path(train.data.dir, "train.img_bin.mx.list.rds")

put_log("The path for the backup file to save the Binary standardized Image  Matrix object:
%1", img.bin_mx.list.file)

if(!file.exists(img.bin_mx.list.file)){
  if(!exists("train.img_bin.mx_list")) {
    stopifnot(file.exists(train.img_bin.mx_list.file))
    train.img_bin.mx_list <- readRDS(train.img_bin.mx_list.file)
  }

  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- train.img_bin.mx_list$img.list
  # names(img.nested_list)
  
  
  img28x28mx.list <- lapply(names(img.nested_list), function(label){
    put_log("Creating array of images of character `%1`", label)
    item <- img.nested_list[[label]]
    fpath_list.size = length(item$fpath.list)
    
    stopifnot(length(item$img.list) == fpath_list.size)
    
    img.array <- abind(item$img.list, rev.along = 3)
    dimnames(img.array) <- list(base::rep(label, 
                                          times = length(item$img.list)),
                                NULL,
                                NULL)
    
    stopifnot(dim(img.array)[1] == fpath_list.size)
    put_log("The array of images of character `%1` has been created.", label)
    
    list(img.array = img.array,
         file.path = item$fpath.list)
  })
  names(img28x28mx.list) <- as.character(Y.Labels)
  rm(img.nested_list)
  
  put_log("A Standardized Image Binary Data Matrix list has been created with the following structure:
  %1", capture.output(str(img28x28mx.list)))
  put_end_date(start)
  
  put_log("Saving Standardized Image Binary Data Matrix list to the backup file...")
  saveRDS(img28x28mx.list,
          file = img.bin_mx.list.file)
  
  # rm(img28x28mx.list)
  
  put_log("Standardized Image Binary Data Matrix list has been saved to the following file:
%1", img.bin_mx.list.file)

} else {
  put_log("The Standardized Image Binary Data Matrix list has already been constructed 
and backed up to the following file:
%1", img.bin_mx.list.file)
}

### Preparing an Array of the Binary standardized Image  Matrix ------------------

put_log("Preparing an Array of the Binary standardized Image  Matrix...")

train.img28x28mx.array.file_path <- file.path(train.data.dir, "train.img28x28mx.array.rds")

put_log("The path for the backup file to save the Array of the Binary Image  Matrix:
%1", train.img28x28mx.array.file_path)

if(!file.exists(train.img28x28mx.array.file_path)){
  if(!exists("img28x28mx.list")) {
    stopifnot(file.exists(img.bin_mx.list.file))
    
    put_log("Loading the Train Standardized Image Binary Data Matrix list...")
    img28x28mx.list <- readRDS(img.bin_mx.list.file)

    put_log("The Train Standardized Image Binary Data Matrix list has been loaded from the following file:
%1", img.bin_mx.list.file)
  }
  
  put_log("Combining Binary image 28x28x matrix list to array...")
  
  img28x28mx.array.list <- lapply(img28x28mx.list, function(item) {
    item$img.array
  })
  img28x28mx.file.list <- lapply(img28x28mx.list, function(item) {
    item$file.path
  })

  img28x28mx.array <- abind(img28x28mx.array.list, along = 1)
  rm(img28x28mx.array.list)
  
  img28x28mx.fpath <- abind(img28x28mx.file.list)
  rm(img28x28mx.file.list)
  
  names(img28x28mx.fpath) <- rownames(img28x28mx.array)
  
  rm(img28x28mx.list)
  
  stopifnot(length(img28x28mx.fpath) == nrow(img28x28mx.array))
  
  put_log("images File Path list has the following structure:
  %1", capture.output(str(img28x28mx.fpath)))
  
  put_log("Combined Binary image matrix 28x28 array has the following structure:
  %1", capture.output(str(img28x28mx.array)))
  
  put_log("Combined Binary image matrix 28x28 array has the following dimentions:
  %1", capture.output(dim(img28x28mx.array)))
  
  put_log("Saving Standardized Image Binary Data array set to the backup file...")
  saveRDS(list(img28x28mx.array = img28x28mx.array,
               img28x28mx.fpath = img28x28mx.fpath),
          file = train.img28x28mx.array.file_path)
  
  put_log("The Standardized Image Binary Data array set has been saved to the following file:
%1", train.img28x28mx.array.file_path)
  
  rm(img28x28mx.array,
     img28x28mx.fpath)

  put_end_date(start)
} else {
  put_log("The Standardized Image Binary Data array has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.array.file_path)
}

if (exists("img28x28mx.list")) rm(img28x28mx.list)

### Preparing Default Split Datasets (of 80%-size Training Set) for NN-Based Models ----

put_log("Preparing Default Split Datasets (of 80%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds28x28.split.train_0.8.file)

start <- put_start_date()

if(!file.exists(ds28x28.split.train_0.8.file)) {
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  
  put_log("Loading and splitting the Train 28x28 Image Data Array 
into default Train and Test Sets...")
  
  ds28x28.split.train_0.8 <- 
    split.img28x28mx_array(train.img28x28mx.array.file_path,
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
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  
  put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")
  
  ds28x28.split.train_0.1 <- 
    split.img28x28mx_array(train.img28x28mx.array.file_path,
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
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  
  put_log("Loading and splitting (without balancing) the array of training images (28x28) 
into standard training and test sets...")
  
  ds.split_imbalanced <- 
    split.img28x28mx_array(train.img28x28mx.array.file_path,
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

my_emnist.file_path <- file.path(train.data.dir, "my_emnist.rds")

put_log("The path for the backup file to save the Flattened EMNIST-Like Dataset:
%1", my_emnist.file_path)


if(!file.exists(my_emnist.file_path)){
  if(!exists("train.img_bin.mx_list")) {
    stopifnot(file.exists(train.img_bin.mx_list.file))
    train.img_bin.mx_list <- readRDS(train.img_bin.mx_list.file)
  }
  
  Y.Labels <- train.img_bin.mx_list$label.list

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  my_emnist.set <- img.list2flatten_matrix(train.img_bin.mx_list$img.list)
  rm(train.img_bin.mx_list)

  put_log("The flatten dataset have been created with the following structure:
  %1", capture.output(str(my_emnist.set)))

  
  put_log("Saving flatten training dataset to the backup file: 
%1", my_emnist.file_path)
  saveRDS(my_emnist.set,
       file = my_emnist.file_path)
  
  rm(my_emnist.set)
  
  put_log("The flatten training dataset has been saved to the backup file:
%1", my_emnist.file_path)
  put_end_date(start)
  
} else {
  put_log("The flatten training dataset has already been constructed 
and backed up to the following file:
%1", my_emnist.file_path)
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

