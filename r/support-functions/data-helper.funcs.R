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

img.file_path.get_list <- function(root_path, 
                                   folder.list = NULL, 
                                   char_files.max = NA) {
  if(is.null(folder.list))
    folder.list <- dir(root_path)

  path_list <- lapply(folder.list, function(folder_name){
    file.root_path <- file.path(root_path, folder_name)
    
    put_log1("Getting file path list from the following char's root folders:
%1", file.root_path)

        fpath.list <- list.files(file.root_path, 
                             full.names = TRUE)
    
    if(!is.na(char_files.max) && length(fpath.list) > char_files.max){
      fpath.list <- fpath.list[seq_len(char_files.max)]
    }
    
    put_log2("%1 files in folder: %2", length(fpath.list), folder_name)

    list(root_path = file.root_path,
         file_path.list = fpath.list)
  })
  
  names(path_list) <- folder.list |> substr(1,1)
  path_list
}

load.kaggle_img <- function(file) {
  img <- imager::load.image(file)
  
  if(dim(img)[4] > 1) {
    img <- grayscale(img)
  }

  img |> 
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
           byrow = TRUE,
           dimnames = list(base::rep(label, times = length(cimg.list)),
                           1:mx.ncols))
}

char.image <- function(char.vector) {
  image(matrix(char.vector, nrow = 28)[, 28:1])
}

## Data processing functions -------------------------------
hwChar_data.load <- function(root_path, 
                             folder.list = NULL, 
                             char_files.max = NA,
                             shuffle.rows = FALSE,
                             shuffle.seed = NA){
  start <- put_start_date()
  put_log("Getting file path lists...")
  img.file_list <- img.file_path.get_list(root_path, 
                                          folder.list,
                                          char_files.max)
  put_end_date(start)
  put_log("File path lists have been created")
  put(str(img.file_list))
  
  start <- put_start_date()
  put_log("Loading image files...")
  img_list <- lapply(img.file_list, function(img_f){
    list(cimg.list = map_il(img_f$file_path.list, load.kaggle_img),
         fpath.list = img_f$file_path.list)
  })
  put_end_date(start)
  put_log("Image files have been loaded.")
  put(str(img_list))
  
  start <- put_start_date()
  put_log("Converting image lists to matrices...")
  char_matrix.list <- lapply(names(img_list), function(label){
    img_list[[label]]$cimg.list |> 
      as.matrix.cimg(label)
  })
  put_end_date(start)
  put_log("Image matrix list has been created.")
  put(str(char_matrix.list))
  
  start <- put_start_date()
  put_log("Combining image data to single matrix...")
  img.mx <- do.call(rbind, char_matrix.list)
  put_end_date(start)
  put_log("Image dataset matrix has been created.")
  put(dim(img.mx))
  
  label_list <- as.factor(names(img.file_list))
  # str(label_list)

  if (shuffle.rows) {
    img.mx <- shuffle.mxrows(img.mx, shuffle.seed)
  }
  
  list(img.files = img.file_list,
       label.list = label_list,
       img.list = char_matrix.list,
       my_mnist = img.mx)
}

sample_train_test_sets.mx <- function(data, 
                                      seed, 
                                      test.ratio = 0.2,
                                      shuffle.test_rows = FALSE,
                                      shuffle.seed = NA){ 
  put_log("Function: `sample_train_test_sets.mx`: Sampling 20% of the `data` data...")

  idx_group.list <- split(seq_len(nrow(data)), 
                           as.factor(rownames(data)))
  #str(idx_group.list)
  set.seed(seed)
  
  test.idx <-
    sapply(idx_group.list,
           function(idx_group) 
             sample(idx_group, 
                    ceiling(length(idx_group)*test.ratio))) |>
    unlist(use.names = FALSE) |>
    sort()
  
  #str(test.idx)
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 80% of the original index set of `data` not used for the test Set...")
  
  train.set <- data[-test.idx,]
  # str(train.set)
  # dim(train.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `train.set`")

  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 20% of the original index set of `data` used for the test Set.")
  
  test.set <- data[test.idx,]
  # str(test.set)
  # dim(test.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `test.set`")

  if (shuffle.test_rows) {
    test.set <- shuffle.mxrows(test.set, shuffle.seed)
  }
  # Return result datassets
  list(train_set = train.set,
       test_set = test.set)
}

shuffle.mxrows <- function(mx, seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
  random.idx <- sample(nrow(mx))
  mx[random.idx,]
}
