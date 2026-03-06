options(timeout = max(300, getOption("timeout")))

kaggle_dataset <- "maryamlsgumel/drone-detection-dataset"
local_filepath <- "data/raw"
zip_file_name <- "drone-detection-dataset.zip"

full_zip_file_name <- file.path(local_filepath, zip_file_name)

if(!file.exists(full_zip_file_name)) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(paste("kaggle datasets download", kaggle_dataset, "--path", local_filepath)) != 0) {
    stop("Failed to download the dataset with Kaggle CLI.")
  }
  
}
  
