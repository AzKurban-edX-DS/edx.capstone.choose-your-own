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
raw_data.chars.path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.chars.path

kaggle_cli.download(kaggle_dataset, raw_data.chars.path)

Upper.Img28.dir_path <- file.path(raw_data.chars.path, "A-Z/Augmentated 28 X 28")
Upper.Img28.dir_path

# Upper.Img28.labels <- dir(Upper.Img28.dir_path)
# Upper.Img28.labels

Upper.Img28File.list <- img.file_path.get_list(Upper.Img28.dir_path)
# str(Upper.Img28File.list)

fileList4 <- Upper.Img28File.list[[4]]
#str(as.data.frame(fileList4))

head(fileList4$file_list)
fileList4$label
fileList4$label |> substr(1,1)
fileList4$label |> str_sub(end = 1)

lower.Img28.dir_path <- file.path(raw_data.chars.path, "a_z/Augmentation 28 X 28")
lower.Img28.dir_path

# lower.Img28.labels <- dir(lower.Img28.dir_path)
# lower.Img28.labels

lower.Img28File.list <- img.file_path.get_list(lower.Img28.dir_path)
# str(lower.Img28File.list)

file.list4 <- lower.Img28File.list[[4]]
head(file.list4$file_path.list)
file.list4$label
file.list4$label |> substr(1,1)
file.list4$label |> str_sub(end = 1)



# output_n <- length(Upper.Img28.labels)
# output_n

#list all the img files in img path
# Upper.Img28File.list <- lapply(Upper.Img28.labels, function(label){
#   file_path.list <- list.files(file.path(Upper.Img28.dir_path, label), 
#              full.names = TRUE, 
#              pattern = "png",
#              recursive = FALSE) 
# 
#   list(file_list = filePath.list,
#        label = label)
# })

