##########################
# Load Input Data Script 
##########################

## Open log: Download Kaggle Dataset -------------------------------------------
open_logfile(".download-kaggle-dataset")
## Download the Kaggle Dataset -------------------------------------------------

# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters
kaggle_dataset <- "vaibhao/handwritten-characters"

if(!dir.exists(raw_data.chars.path)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.path, unzip = TRUE)
  print_log1("The Kaggle dataset has been downloaded and unzip to folder: `%1`", raw_data.chars.path)
} else {
  warning(get_log1("Nothing to do: directory already exists: `%1`.
If you need to rerun the download, delete this folder and rerun this script.", 
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
  warning(get_log1("Nothing to do: directory does not exist: `%1`", dir.to_remove))
}

### Close Log ------------------------------------------------------------------
log_close()

### Open log: Load Train Data --------------------------------------------------
open_logfile(".load-train-data")
### Load Train Data ------------------------------------------------------------
train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.RData")
train.img28x28bin.list.file_path

if (!file.exists(train.img28x28bin.list.file_path)) {
  put_log("Creating Binary Image 28x28 list from raw data files from root directory:
%1", img.train.root_path)
  start <- put_start_date()
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  img28x28bin.list <- img.load.bin28x28mx.list(img.train.root_path)

  put_log("Saving Binary Image 28x28 list to the backup file...")
  start <- put_start_date()
  save(img28x28bin.list,
       file = train.img28x28bin.list.file_path)
  put_log("Binary Image 28x28 list has been saved to the following file:
%1", train.img28x28bin.list.file_path)
  put_end_date(start)
  
put_log("Binary Image 28x28 list summary:
%1", capture.output(summary(img28x28bin.list)))

rm(img28x28bin.list)
} else {
    put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", train.img28x28bin.list.file_path)
  
  
#   start <- put_start_date()
#   put_log("Loading Binary Image 28x28 Matrix list from the backup file...")
#   load(train.img28x28bin.list.file_path)
#   put_log("The Binary Image 28x28 Matrix list has been loaded from the following file:
# %1", train.img28x28bin.list.file_path)
#   put_end_date(start)
} 


### Close Log ---------------------------------------------------------------
log_close()

