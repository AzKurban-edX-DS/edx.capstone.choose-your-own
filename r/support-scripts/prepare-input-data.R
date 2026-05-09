#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input Data Preparing Script 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Open log: Download Kaggle Dataset -------------------------------------------
open_logfile(".download-kaggle-dataset")
## Download the Kaggle Dataset -------------------------------------------------

# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters
kaggle_dataset <- "vaibhao/handwritten-characters"

start.download <- put_start_date()

if(!dir.exists(raw_data.chars.path)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.path, unzip = TRUE)
  print_log1("The Kaggle image files have been downloaded 
and unziped to the following directory: 
`%1`", raw_data.chars.path)
} else {
  warning(get_log1("The Kaggle image files have already been downloaded 
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
  warning(get_log1("Couldn't delete the folder:
`%1`  
It has already been deleted or moved.", 
                   dir.to_remove))
}

put_end_date(start.download)

### Close Log ------------------------------------------------------------------
log_close()

### Open log: Prepare Train Data --------------------------------------------------
open_logfile(".prepare-train-data")
### Prepare Train Data ------------------------------------------------------------
train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.rds")
train.img28x28bin.list.file_path

train.img28x28mx.list.file_path <- file.path(train.data.path, "train.img28x28mx.list.rds")
train.img28x28mx.list.file_path

train.img28x28mx.array.file_path <- file.path(train.data.path, "train.img28x28mx.array.rds")
train.img28x28mx.array.file_path

classifier.label_list.file_path <- file.path(train.data.path, "classifier.label-list.rds")
classifier.label_list.file_path

my_emnist.file_path <- file.path(train.data.path, "my_emnist.rds")
my_emnist.file_path

start <- put_start_date()

if (!file.exists(train.img28x28bin.list.file_path)) {
  put_log("Creating Binary Image 28x28 list from raw data files from root directory:
%1", img.train.root_path)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  img28x28bin.list <- img.load.bin28x28mx.list(img.train.root_path)

  put_log("Saving Binary Image 28x28 list to the backup file...")
  saveRDS(img28x28bin.list,
       file = train.img28x28bin.list.file_path)
  put_log("Binary Image 28x28 list has been saved to the following file:
%1", train.img28x28bin.list.file_path)

  put_log("Binary Image 28x28 list summary:
%1", capture.output(summary(img28x28bin.list)))
  
} else {
    put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", train.img28x28bin.list.file_path)
}

if(!file.exists(classifier.label_list.file_path)){
  if(!exists("img28x28bin.list")) {
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

# put_log("The Classifier Label List contains the following labels:
# %1", capture.output(y.labels))
put_log("The Classifier Handwritten Character List contains the following labels:
%1", y.labels, .sep = " ")

if(!file.exists(train.img28x28mx.list.file_path)){
  if(!exists("img28x28bin.list")) {
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }

  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- img28x28bin.list$img.list
  length(img.nested_list)
  names(img.nested_list)
  
  
  img28x28mx.list <- lapply(names(img.nested_list), function(label){
    item <- img.nested_list[[label]]
    img.array <- abind(item$img.list, rev.along = 3)
    dimnames(img.array) <- list(base::rep(label, 
                                          times = length(item$img.list)),
                                NULL,
                                NULL)
    img.array
  })
  names(img28x28mx.list) <- as.character(y.labels)
  
  put_log("Combined image data matrix has been created with the following structure:
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
    img28x28bin.list <- readRDS(train.img28x28mx.list.file_path)
  }
  
  put_log("Combining Binary image 28x28x matrix list to array...")
  
  img28x28mx.array <- abind(img28x28mx.list, along = 1)
  
  put_log("Combined Binary image matrix 28x28 array has been created with the following dimentions:
  %1", capture.output(dim(x_cnn)))
  
  put_log("Saving Binary Image 28x28 array to the backup file...")
  saveRDS(img28x28mx.array,
          file = train.img28x28mx.array.file_path)
  put_log("The Binary Image 28x28 array has been saved to the following file:
%1", train.img28x28mx.array.file_path)
  put_end_date(start)
  
  rm(img28x28mx.array)
} else {
  put_log("The Binary Image 28x28 array has already been constructed 
and backed up to the following file:
%1", train.img28x28mx.array.file_path)
}

if(!file.exists(my_emnist.file_path)){
  if(!exists("img28x28bin.list")) {
    img28x28bin.list <- readRDS(train.img28x28bin.list.file_path)
  }
  
  y.labels <- img28x28bin.list$label.list

  put_log("Building flatten (`EMNIST`-like) dataset...")
  
  my_emnist <- img28x28.list2flatten.mx(img28x28bin.list$img.list)

  put_log("The flatten dataset have been created with the following structure:
  %1", capture.output(str(my_emnist)))

  # rm(img28x28bin.list)
  
  put_log("Saving flatten training dataset to the backup file: 
%1", my_emnist.file_path)
  # start <- put_start_date()
  saveRDS(my_emnist,
       file = my_emnist.file_path)
  put_log("The flatten training dataset has been saved to the backup file:
%1", my_emnist.file_path)
  put_end_date(start)

} else {
  put_log("The flatten training dataset has already been constructed 
and backed up to the following file:
%1", my_emnist.file_path)
}

rm(img28x28bin.list)
put_end_date(start)

### Close Log ---------------------------------------------------------------
log_close()

