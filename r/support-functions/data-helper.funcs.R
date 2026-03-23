kaggle_cli.download <- function(dataset.path, data.local_path, unzip = FALSE) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(trimws(paste("kaggle datasets download", kaggle_dataset, "--path", 
                          data.local_path, ifelse(unzip, "--unzip", "")))) != 0) {
    stop(get_log1("Failed to download the dataset with Kaggle CLI: `%1`.", dataset.path))
  }
  
  if(!dir.exists(data.local_path)) {
    stop(get_log1("Failed to download and unzip the Kaggle dataset: `%1`.", dataset.path))
  }
}

img.file_path.get_list <- function(dir_path) {
  folder.list <- dir(dir_path)

  lapply(folder.list, function(folder.name){
    label <- folder.name |> substr(1,1)
    file.dir_path <- file.path(dir_path, label)
    file_path.list <- list.files(file.dir_path, 
                                 full.names = TRUE, 
                                 recursive = FALSE) 
    
    list(file_path.list = file_path.list,
         dir_path = file.dir_path,
         label = label)
  })
}

