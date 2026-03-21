kaggle_cli.download <- function(dataset.path, data.local_path) {
  if(!dir.exists(data.local_path)) {
    if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
      stop("Kaggle CLI is not installed or not in the PATH.")
    }
    
    if (system(paste("kaggle datasets download", kaggle_dataset, "--path", data.local_path, "--unzip")) != 0) {
      stop(get_log1("Failed to download the dataset with Kaggle CLI: `%1`.", dataset.path))
    }
    
  }
  
  if(!dir.exists(data.local_path)) {
    stop(get_log1("Failed to download and unzip the Kaggle dataset: `%1`.", dataset.path))
  }
}
