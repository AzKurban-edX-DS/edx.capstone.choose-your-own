#load the packages in R

library(imager)
library(magick)

# Raw Data Loading -------------------------------------------------------------
kaggle_dataset <- "tarunkumarkaggle/english-alphabets-28-and-64-px-handwritten-for-ocr"
raw_data_path <- "learning/cv/data/raw"
# zip_file_name <- "drone-detection-dataset.zip"

# full_zip_file_name <- file.path(raw_data_path, zip_file_name)
raw_data_folder_name <- "hw-en-chars"
raw_data_folder_path <- file.path(raw_data_path, raw_data_folder_name)


if(!dir.exists(raw_data_folder_path)) {
  # if(!file.exists(full_zip_file_name)) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(paste("kaggle datasets download", kaggle_dataset, "--path", raw_data_folder_path, "--unzip")) != 0) {
    stop("Failed to download the dataset with Kaggle CLI.")
  }
  
}

if(!dir.exists(raw_data_folder_path)) {
  stop("Failed to download and unzip the dataset `BirdVsDroneVsAirplane`.")
  
}

charLabels <- dir(raw_data_folder_path)
charLabels

output_n <- length(charLabels)
