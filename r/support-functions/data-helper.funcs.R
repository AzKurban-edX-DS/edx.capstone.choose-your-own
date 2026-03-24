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

img.file_path.get_list <- function(dir_path, folder.list) {
  # folder.list <- dir(dir_path)

  lapply(folder.list, function(folder.name){
    # label <- folder.name |> substr(1,1)
    file.dir_path <- file.path(dir_path, folder.name)
    file_path.list <- list.files(file.dir_path, 
                                 full.names = TRUE)
    
    list(file_path.list = file_path.list,
         dir_path = file.dir_path)
         # label = label)
  })
}

load.kaggle_img <- function(file) {
  imager::load.image(file) |> 
    grayscale() |> 
    resize(size_x = 28,
           size_y = 28)
}

reshape.cimg_tensor <- function(cimg) {
  as.vector(cimg[,,1,1])
}

reshape.cimg_tensor.list <- function(cimg.list) {
  map(cimg.list, reshape.cimg_tensor)
}

reshape.cimg_tensor.mx <- function(cimg.list, label) {
  mx <- map(cimg.list, reshape.cimg_tensor) |>
    unlist() |>
    matrix(ncol = 28*28)
  
  rownames(mx) <- base::rep(label, times = nrow(mx))
  mx
}


