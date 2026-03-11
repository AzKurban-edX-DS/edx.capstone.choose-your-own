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
bdaLabels

output_n <- length(bdaLabels)
# save(bdaLabels, file=file.path(raw_data_folder_path, "bdaLabels-list.rds"))

# width <- 224
# height<- 224
# target_size <- c(height, width)
# rgb <- 3 #color channels

# path_train <- "/train/"

bda_dat <- image_dataset_from_directory(raw_data_folder_path,
                                               seed = 31026,
                                               subset = "both",
                                               # crop_to_aspect_ratio = FALSE,
                                               pad_to_aspect_ratio = TRUE,
                                               validation_split = .2)

str(bda_dat[[1]])






