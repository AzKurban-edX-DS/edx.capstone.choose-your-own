#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input Data Preparing Script 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Download the Kaggle Dataset -------------------------------------------------

# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters

kaggle_dataset <- "vaibhao/handwritten-characters"

open_logfile(".download-kaggle-dataset")
start.download <- put_start_date()

if(!dir.exists(raw_data.chars.dir)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.dir, unzip = TRUE)
  print_log1("The Kaggle image files have been downloaded 
and unziped to the following directory: 
`%1`", raw_data.chars.dir)
} else {
  warning(str.build("The Kaggle image files have already been downloaded 
and saved to the following directory: 
`%1`.
If you need to rerun the download, delete the root folder and rerun this script.", 
                   raw_data.chars.dir))
}

# Remove duplicate files:
dir.to_remove <- file.path(raw_data.chars.dir, "dataset")
dir.to_remove

if (dir.exists(dir.to_remove)) {
  print_log1("Deleting the folder with duplicate files: `%1`...", dir.to_remove)
  unlink(dir.to_remove, recursive = TRUE, force = TRUE)
  print_log1("Directory removed: `%1`", dir.to_remove)
} else {
  warning(str.build("Couldn't delete the folder:
`%1`  
It has already been deleted or moved.", 
                   dir.to_remove))
}

put_end_date(start.download)
log_close()

## Prepare Train Data ------------------------------------------------------------

open_logfile(".prepare-train-data")
start <- put_start_date()

### Preparing a List of Binary 28x28-size Image Presentation Objects -----------

put_log("Preparing a List of Binary 28x28-size Image Presentation Objects...")

train.img28x28bin.list.file_path <- file.path(train.data.dir, 
                                              "train.img28x28bin.list.rds")

put_log("The path for the backup file to save the Binary 28x28-size Image Object list:
%1", train.img28x28bin.list.file_path)

if (!file.exists(train.img28x28bin.list.file_path)) {
  put_log("Creating a Binary Image 28x28 list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train_root.dir)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  img28x28bin.list <- img.load.bin28x28mx.list(img.train_root.dir)
  
  put_log("The Binary Image 28x28 list has been created with the following structure:
%1", capture.output(str(img28x28bin.list)))
  put_end_date(start)

  put_log("Saving Binary Image 28x28 list to the backup file...")
  saveRDS(img28x28bin.list,
       file = train.img28x28bin.list.file_path)
  put_log("Binary Image 28x28 list has been saved to the following file:
%1", train.img28x28bin.list.file_path)

} else {
    put_log("The Binary Image 28x28 list has already been constructed 
and backed up to the following file:
%1", train.img28x28bin.list.file_path)
}

### Preparing the Multiclass Classifier Class Label List -----------------------

put_log("Preparing the Project Multiclass Classifier Class Label List...")

classifier.label_list.file_path <- file.path(dataset.dir, 
                                             "classifier.label-list.rds")

put_log("The path for the backup file to save the list of the Multiclass Classifier Class Labels:
%1", classifier.label_list.file_path)

if(!file.exists(classifier.label_list.file_path)){
  if(!exists("img28x28bin.list")) {
    stopifnot(file.exists(train.img28x28bin.list.file_path))
    
    put_log("Loading the Binary Image 28x28 list from the backup file...")
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
    put_log("The Binary Image 28x28 list has been loaded from the following file:
%1", train.img28x28bin.list.file_path)
  }
 
  Y.Labels <- img28x28bin.list$label.list
  
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

### Preparing a List of Binary 28x28-size Image  Matrix ------------------------

put_log("Preparing a List of the Binary 28x28-size Image  Matrix...")

train.img28x28mx.list.file_path <- file.path(train.data.dir, "train.img28x28mx.list.rds")

put_log("The path for the backup file to save the Binary 28x28-size Image  Matrix object:
%1", train.img28x28mx.list.file_path)

if(!file.exists(train.img28x28mx.list.file_path)){
  if(!exists("img28x28bin.list")) {
    stopifnot(file.exists(train.img28x28bin.list.file_path))
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }

  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- img28x28bin.list$img.list
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
  
  put_log("A Binary Image 28x28 Matrix list has been created with the following structure:
  %1", capture.output(str(img28x28mx.list)))
  put_end_date(start)
  
  put_log("Saving Binary Image 28x28 Matrix list to the backup file...")
  saveRDS(img28x28mx.list,
          file = train.img28x28mx.list.file_path)
  
  # rm(img28x28mx.list)
  
  put_log("Binary Image 28x28 Matrix list has been saved to the following file:
%1", train.img28x28mx.list.file_path)

} else {
  put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.list.file_path)
}

### Preparing an Array of the Binary 28x28-size Image  Matrix ------------------

put_log("Preparing an Array of the Binary 28x28-size Image  Matrix...")

train.img28x28mx.array.file_path <- file.path(train.data.dir, "train.img28x28mx.array.rds")

put_log("The path for the backup file to save the Array of the Binary Image  Matrix:
%1", train.img28x28mx.array.file_path)

if(!file.exists(train.img28x28mx.array.file_path)){
  if(!exists("img28x28mx.list")) {
    stopifnot(file.exists(train.img28x28mx.list.file_path))
    
    put_log("Loading the Train Binary Image 28x28 Matrix list...")
    img28x28mx.list <- readRDS(train.img28x28mx.list.file_path)

    put_log("The Train Binary Image 28x28 Matrix list has been loaded from the following file:
%1", final_test.img28x28mx.list.file_path)
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
  
  put_log("Saving Binary Image 28x28 array set to the backup file...")
  saveRDS(list(img28x28mx.array = img28x28mx.array,
               img28x28mx.fpath = img28x28mx.fpath),
          file = train.img28x28mx.array.file_path)
  
  put_log("The Binary Image 28x28 array set has been saved to the following file:
%1", train.img28x28mx.array.file_path)
  
  rm(img28x28mx.array,
     img28x28mx.fpath)

  put_end_date(start)
} else {
  put_log("The Binary Image 28x28 array has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.array.file_path)
}

if (exists("img28x28mx.list")) rm(img28x28mx.list)

### Preparing Default Split Datasets (of 80%-size Training Set) for NN-Based Models ----

put_log("Preparing Default Split Datasets (of 80%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds28x28.split.train_0.8.backup.file)

start <- put_start_date()

if(!file.exists(ds28x28.split.train_0.8.backup.file)) {
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  
  put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")
  
  ds28x28.split.train_0.8 <- 
    split.img28x28mx_array(train.img28x28mx.array.file_path,
                           test_ratio = 0.2)
  
  put_log("The Default Split Dataset object structure:
%1", capture.output(str(ds28x28.split.train_0.8)))

  put_log("Saving the Split Dataset List object in the backup file...")
  
  saveRDS(ds28x28.split.train_0.8, 
          file = ds28x28.split.train_0.8.backup.file)
  
  rm(ds28x28.split.train_0.8)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds28x28.split.train_0.8.backup.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds28x28.split.train_0.8.backup.file)
}

put_end_date(start)
log_close()

### Preparing Split Datasets of 10%-size Training Set for NN-Based Models ----

put_log("Preparing Split Datasets of 10%-size Training Set) for training 
the NN-based Multiclass Classifier (MCC) Models...")


put_log("The path for the backup file to save the list of the Split Datasets: 
%1", ds28x28.split.train_0.1.backup.file)

start <- put_start_date()

if(!file.exists(ds28x28.split.train_0.1.backup.file)) {
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
          file = ds28x28.split.train_0.1.backup.file)
  
  rm(ds28x28.split.train_0.1)
  
  put_log("The Split Dataset List object has been backed up in the following file:
`%1`", ds28x28.split.train_0.1.backup.file)
} else {
  put_log("The Split Datasets for training the NN-based Multiclass Classifier Models 
has already been constructed and backed up to the following file:
%1", ds28x28.split.train_0.1.backup.file)
}

put_end_date(start)
log_close()

### Preparing the Flattened EMNIST-Like Dataset --------------------------------

put_log("Preparing the Flattened EMNIST-Like Dataset for the DL-Based Basic and Non-NN-Based Models...")

my_emnist.file_path <- file.path(train.data.dir, "my_emnist.rds")

put_log("The path for the backup file to save the Flattened EMNIST-Like Dataset:
%1", my_emnist.file_path)


if(!file.exists(my_emnist.file_path)){
  if(!exists("img28x28bin.list")) {
    stopifnot(file.exists(train.img28x28bin.list.file_path))
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }
  
  Y.Labels <- img28x28bin.list$label.list

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  my_emnist.set <- img.list2flatten_matrix(img28x28bin.list$img.list)
  rm(img28x28bin.list)

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

## Prepare Final Test Data ----------------------------------------------------

open_logfile(".prepare-final_test-data")
start <- put_start_date()

final_test.img28x28bin.list.file_path <- file.path(final_test.data.dir, 
                                                   "final_test.img28x28bin.list.rds")

put_log("The path for the backup file to save the Final Test
List of Binary 28x28-size Image Objects:
%1", final_test.img28x28bin.list.file_path)

final_test.img28x28mx.list.file_path <- file.path(final_test.data.dir, 
                                                  "final_test.img28x28mx.list.rds")

put_log("The path for the backup file to save the Final Test 
List of Binary 28x28-size Image  Matrix:
%1", final_test.img28x28mx.list.file_path)

final_test.img28x28mx.array.file_path <- file.path(final_test.data.dir, 
                                                   "final_test.img28x28mx.array.rds")

put_log("The path for the backup file to save the Final Test 
Array of the Binary 28x28-size Image  Matrix:
%1", final_test.img28x28mx.array.file_path)

if (!file.exists(final_test.img28x28bin.list.file_path)) {
  put_log("Creating the Final Test Binary Image 28x28 list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.validation_root.dir)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  ft.img28x28bin.list <- img.load.bin28x28mx.list(img.validation_root.dir)
  put_end_date(start)
  
  put_log("The Final Test Binary Image 28x28 list has been created with the following structure:
%1", capture.output(str(ft.img28x28bin.list)))

  put_log("Saving the Final Test Binary Image 28x28 list...")
  
  saveRDS(ft.img28x28bin.list,
       file = final_test.img28x28bin.list.file_path)
  
  put_log("The Final Test Binary Image 28x28 list has been saved to the following file:
%1", final_test.img28x28bin.list.file_path)
} else {
    put_log("The Binary Image 28x28 list has already been constructed 
and backed up in the following file:
%1", final_test.img28x28bin.list.file_path)
}
put_end_date(start)

if(!file.exists(final_test.img28x28mx.list.file_path)){
  if(!exists("ft.img28x28bin.list")) {
    stopifnot(file.exists(final_test.img28x28bin.list.file_path))
    ft.img28x28bin.list <- readRDS(final_test.img28x28bin.list.file_path)
  }

  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- ft.img28x28bin.list$img.list
  rm(ft.img28x28bin.list)
  
  length(img.nested_list)
  names(img.nested_list)
  
  
  ft.img28x28mx.list <- lapply(names(img.nested_list), function(label){
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
  rm(img.nested_list)
  
  names(ft.img28x28mx.list) <- as.character(Y.Labels)
  
  put_log("The Final Test Binary Image 28x28 Matrix list has been created with the following structure:
  %1", capture.output(str(ft.img28x28mx.list)))
  put_end_date(start)
  
  put_log("Saving Binary Image 28x28 Matrix list to the backup file...")
  saveRDS(ft.img28x28mx.list,
          file = final_test.img28x28mx.list.file_path)
  
  put_log("Binary Image 28x28 Matrix list has been saved to the following file:
%1", final_test.img28x28mx.list.file_path)
  put_end_date(start)
} else {
  put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", final_test.img28x28mx.list.file_path)
}

if(!file.exists(final_test.img28x28mx.array.file_path)){
  if (!exists("ft.img28x28mx.list")) {
    stopifnot(file.exists(final_test.img28x28mx.list.file_path))

    put_log("Loading the Final Binary Image 28x28 Matrix list...")
    ft.img28x28mx.list <- readRDS(final_test.img28x28mx.list.file_path)
    
    put_log("The Final Test Binary Image 28x28 Matrix list has been loaded from the following file:
%1", final_test.img28x28mx.list.file_path)
  }
  
  put_log("Combining Binary image 28x28x matrix list to array...")
  
  img28x28mx.array.list <- lapply(ft.img28x28mx.list, function(item) {
    item$img.array
  })
  img28x28mx.file.list <- lapply(ft.img28x28mx.list, function(item) {
    item$file.path
  })

  img28x28mx.array <- abind(img28x28mx.array.list, along = 1)
  rm(img28x28mx.array.list)
  
  img28x28mx.fpath <- abind(img28x28mx.file.list)
  rm(img28x28mx.file.list)
  
  names(img28x28mx.fpath) <- rownames(img28x28mx.array)
  
  stopifnot(length(img28x28mx.fpath) == nrow(img28x28mx.array))

  put_log("Combined Final Test Binary image matrix 28x28 array has the following structure:
  %1", capture.output(str(img28x28mx.array)))
  
  put_log("Combined Final Test Binary image matrix 28x28 array has the following dimentions:
  %1", capture.output(dim(img28x28mx.array)))
  
  put_log("Saving Final Test Binary Image 28x28 array to the backup file...")
  saveRDS(list(img28x28mx.array = img28x28mx.array,
               img28x28mx.fpath = img28x28mx.fpath),
          file = final_test.img28x28mx.array.file_path)
  
  put_log("The Final Test Binary Image 28x28 array has been saved to the following file:
%1", final_test.img28x28mx.array.file_path)
  
  rm(img28x28mx.array)
  rm(img28x28mx.fpath)
} else {
  put_log("The Binary Image 28x28 array has already been constructed 
and backed up to the following file:
%1", final_test.img28x28mx.array.file_path)
}

put_end_date(start)
log_close()
