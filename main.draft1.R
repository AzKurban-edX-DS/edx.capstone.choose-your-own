# Main Script ------------------------------------------------------------------

## Initial Paths ---------------------------------------------------------------
r.path <- "r"

support_functions.folder <- "support-functions"
support_scripts.folder <- "support-scripts"

support_scripts.path <- file.path(r.path, support_scripts.folder)
support_functions.path <- file.path(r.path, support_functions.folder)

setup_script.file_path <- file.path(support_scripts.path, "setup.R")

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Logging Helper Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-functions.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.path, "data-helper.funcs.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Download the Kaggle Dataset -------------------------------------------------

kaggle_dataset <- "tarunkumarkaggle/english-alphabets-28-and-64-px-handwritten-for-ocr"
raw_data.path <- "data/raw"

raw_data.folder_name <- "handwritten.enChars"
raw_data.folder_path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.folder_path

kaggle_cli.download(kaggle_dataset, raw_data.folder_path)

A_Z.img_path <- file.path(raw_data.folder_path, "A-Z/Augmentated 28 X 28")
A_Z.img_path

charLabels_A_Z28 <- dir(A_Z.img_path)
charLabels_A_Z28

output_n <- length(charLabels_A_Z28)
output_n

#list all the img files in img path
A_Z.imgFile.list <- lapply(charLabels_A_Z28, function(label){
  filePath.list <- list.files(file.path(A_Z.img_path, label), 
             full.names = TRUE, 
             pattern = "png",
             recursive = FALSE) 
  # lapply(filePath.list, function(filePath) {
  #   c(filePath = filePath, label = label)
  # })
  
  list(file_list = filePath.list,
       label = label)
})

str(A_Z.imgFile.list)

charLabels_A_Z28[[4]]
# head(A_Z.imgFile.list[[1]])
# head(A_Z.imgFile.list[[2]])
fileList4 <- A_Z.imgFile.list[[4]]
#str(as.data.frame(fileList4))
#head(fileList4)
head(fileList4$file_list)
fileList4$label
