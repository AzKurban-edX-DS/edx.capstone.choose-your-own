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

if(!dir.exists(raw_data.chars.path)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.path, unzip = TRUE)
  print_log1("The Kaggle image files have been downloaded 
and unziped to the following directory: 
`%1`", raw_data.chars.path)
} else {
  warning(str.build("The Kaggle image files have already been downloaded 
and saved to the following directory: 
`%1`.
If you need to rerun the download, delete the root folder and rerun this script.", 
                   raw_data.chars.path))
}

# Remove duplicate files:
dir.to_remove <- file.path(raw_data.chars.path, "dataset")
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

## Init Backup File Path -------------------------------------

classifier.label_list.file_path <- file.path(dataset.path, "classifier.label-list.rds")
classifier.label_list.file_path

### Init Train Backup File Paths ----------------------------------------------
train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.rds")
train.img28x28bin.list.file_path

train.img28x28mx.list.file_path <- file.path(train.data.path, "train.img28x28mx.list.rds")
train.img28x28mx.list.file_path

train.img28x28mx.array.file_path <- file.path(train.data.path, "train.img28x28mx.array.rds")
train.img28x28mx.array.file_path

my_emnist.file_path <- file.path(train.data.path, "my_emnist.rds")
my_emnist.file_path

### Init Final Test Backup File Paths -----------------------------------------
final_test.img28x28bin.list.file_path <- file.path(final_test.data.path, "final_test.img28x28bin.list.rds")
final_test.img28x28bin.list.file_path

final_test.img28x28mx.list.file_path <- file.path(final_test.data.path, "final_test.img28x28mx.list.rds")
final_test.img28x28mx.list.file_path

final_test.img28x28mx.array.file_path <- file.path(final_test.data.path, "final_test.img28x28mx.array.rds")
final_test.img28x28mx.array.file_path

## Prepare Train Data ------------------------------------------------------------

open_logfile(".prepare-train-data")
start <- put_start_date()

if (!file.exists(train.img28x28bin.list.file_path)) {
  put_log("Creating a Binary Image 28x28 list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train.root_path)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  img28x28bin.list <- img.load.bin28x28mx.list(img.train.root_path)
  
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

if(!file.exists(classifier.label_list.file_path)){
  if(!exists("img28x28bin.list")) {
    stopifnot(file.exists(train.img28x28bin.list.file_path))
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }
 
  y.labels <- img28x28bin.list$label.list
  
  put_log("Saving Classifier Label List to the backup file...")
  saveRDS(y.labels,
          file = classifier.label_list.file_path)
  put_log("The Classifier Label List has been saved to the following file:
%1", classifier.label_list.file_path)
  put_end_date(start)
} else {
  put_log("The Classifier Label List has already been extracted 
and backed up to the following file:
%1", classifier.label_list.file_path)
  
  if(!exists("y.labels"))
    y.labels <- readRDS(classifier.label_list.file_path)
}

N.classes <- length(y.labels)


# put_log("The Classifier Label List contains the following labels:
# %1", capture.output(y.labels))
put_log("The Classifier Handwritten Character List contains the following labels:
%1", y.labels, .sep = " ")

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
  names(img28x28mx.list) <- as.character(y.labels)
  
  put_log("A Binary Image 28x28 Matrix list has been created with the following structure:
  %1", capture.output(str(img28x28mx.list)))
  put_end_date(start)
  
  put_log("Saving Binary Image 28x28 Matrix list to the backup file...")
  saveRDS(img28x28mx.list,
          file = train.img28x28mx.list.file_path)
  put_log("Binary Image 28x28 Matrix list has been saved to the following file:
%1", train.img28x28mx.list.file_path)

} else {
  put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.list.file_path)
}

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
  
# rm(img28x28mx.list)
  
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
  put_end_date(start)
  
  rm(img28x28mx.array)
  rm(img28x28mx.fpath)
} else {
  put_log("The Binary Image 28x28 array has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.array.file_path)
}

if(!file.exists(my_emnist.file_path)){
  if(!exists("img28x28bin.list")) {
    stopifnot(file.exists(train.img28x28bin.list.file_path))
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }
  
  y.labels <- img28x28bin.list$label.list

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  my_emnist.set <- img.list2flatten_matrix(img28x28bin.list$img.list)
  rm(img28x28bin.list)

  put_log("The flatten dataset have been created with the following structure:
  %1", capture.output(str(my_emnist.set)))

  
  put_log("Saving flatten training dataset to the backup file: 
%1", my_emnist.file_path)
  # start <- put_start_date()
  saveRDS(my_emnist.set,
       file = my_emnist.file_path)
  put_log("The flatten training dataset has been saved to the backup file:
%1", my_emnist.file_path)
  put_end_date(start)
  
} else {
  put_log("The flatten training dataset has already been constructed 
and backed up to the following file:
%1", my_emnist.file_path)
}

put_end_date(start)
log_close()

## Prepare Final Test Data ----------------------------------------------------

open_logfile(".prepare-final_test-data")
start <- put_start_date()

if (!file.exists(final_test.img28x28bin.list.file_path)) {
  put_log("Creating the Final Test Binary Image 28x28 list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.validation.root_path)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  ft.img28x28bin.list <- img.load.bin28x28mx.list(img.validation.root_path)
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
  
  names(ft.img28x28mx.list) <- as.character(y.labels)
  rm(img.nested_list)
  
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
