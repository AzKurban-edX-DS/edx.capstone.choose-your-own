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
# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
s
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters
kaggle_dataset <- "vaibhao/handwritten-characters"
raw_data.path <- "data/raw"

raw_data.folder_name <- "Vaibs.HW-Chars"
raw_data.chars.path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.chars.path

if(!dir.exists(raw_data.chars.path)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.path, unzip = TRUE)
  print_log1("The Kaggle dataset has been downloaded and unzip to folder: `%1`", raw_data.chars.path)
} else {
  warning(get_log1("Nothing to do: directory already exists: `%1`", raw_data.chars.path))
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

# ------------------------------------------------------------------------------

train_img32.dir_path <- file.path(raw_data.chars.path, "Train")
train_img32.dir_path

# Upper.Img28.labels <- dir(train_img32.dir_path)
# Upper.Img28.labels

train_img32.file_list <- img.file_path.get_list(train_img32.dir_path)
# str(train_img32.file_list)

fileList4 <- train_img32.file_list[[4]]

head(fileList4$file_path.list)
fileList4$label

fileList5 <- train_img32.file_list[[5]]

head(fileList5$file_path.list)
fileList5$label


img_list5 <- map_il(fileList5$file_path.list[1:5], imager::load.image)
img_list5
str(img_list5)













