# Setup -----------------------------------------------------------------------
options(timeout = max(300, getOption("timeout")))

#> Reference: Some ideas and code snippers were used from the following GitHub repository:
#> https://github.com/AzKurban-edX-DS/harvardx-movielens

if(!require(tidyverse))
  install.packages("tidyverse")

library(tidyverse)

#> `stringr` library is already included to the `tidyverse` package,
#> there's no need to install `stringr`
# if(!require(stringr))
#   install.packages("stringr")

# if(!require())
#   install.packages("")
# if(!require())
#   install.packages("")

if(!require(remotes))
  install.packages("remotes")

# if(!require(tensorflow))
#   remotes::install_github("rstudio/tensorflow")

# if(!require(tensorflow))
#   remotes::install_github("rstudio/tensorflow")
# 
# library(tensorflow)
# install_tensorflow(envname = "r-tensorflow")


if(!require(reticulate))
  install.packages("reticulate")

library(reticulate)

# py_require(python_version = ">=3.11")
reticulate::install_python(version = "3.11")

if(!require(keras3))
  install.packages("keras3")

library(keras3)
install_keras()

# reticulate::virtualenv_remove("r-tensorflow")
# install_tensorflow(extra_packages="pillow")
# install_tensorflow(envname = "r-tensorflow")

library(tensorflow)
tf$constant("Hello TensorFlow!")
tensorflow::tf_version()





# Raw Data Loading -------------------------------------------------------------
kaggle_dataset <- "maryamlsgumel/drone-detection-dataset"
raw_data_path <- "data/raw"
# zip_file_name <- "drone-detection-dataset.zip"

# full_zip_file_name <- file.path(raw_data_path, zip_file_name)
raw_data_folder_name <- "BirdVsDroneVsAirplane"
raw_data_folder_path <- file.path(raw_data_path, raw_data_folder_name)


if(!dir.exists(raw_data_folder_path)) {
# if(!file.exists(full_zip_file_name)) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(paste("kaggle datasets download", kaggle_dataset, "--path", raw_data_path, "--unzip")) != 0) {
    stop("Failed to download the dataset with Kaggle CLI.")
  }
  
}

if(!dir.exists(raw_data_folder_path)) {
  stop("Failed to download and unzip the dataset `BirdVsDroneVsAirplane`.")
  
}
  
bdaLabels <- dir(raw_data_folder_path)
