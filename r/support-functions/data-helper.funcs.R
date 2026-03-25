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

img.file_path.get_list <- function(root_path, label.list = NULL) {
  folder.list <- dir(root_path)
  
  if(is.null(label.list))
    label.list <- folder.list
  
  path_list <- lapply(dir(root_path), function(folder_name){
    file.root_path <- file.path(root_path, folder_name)
    list(root_path = file.root_path,
         file_path.list = list.files(file.root_path, 
                                     full.names = TRUE))
  })
  
  names(path_list) <- label.list
  path_list
}

load.kaggle_img <- function(file) {
  imager::load.image(file) |> 
    grayscale() |> 
    resize(size_x = 28,
           size_y = 28)
}

as.vector.cimg <- function(cimg) {
  as.vector(cimg[,,1,1])
}

as.matrix.cimg <- function(cimg.list, label) {
  mx.ncols <- 28*28
  
  map(cimg.list, as.vector.cimg) |>
    unlist() |>
    matrix(ncol = mx.ncols, 
           dimnames = list(base::rep(label, times = length(cimg.list)),
                           1:mx.ncols))
}


char.image <- function(char.vector) {
  image(matrix(char.vector, nrow = 28)[, 28:1])
}

create.file_path.list <- function() {
  
}

create.hwChar_dataset <- function(root_path){
  
}
