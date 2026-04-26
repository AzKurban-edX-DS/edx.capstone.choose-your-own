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
                                   char_files.max = NA,
                                   seed = NA,
                                   random_sample = FALSE) {
  if(is.null(folder.list))
    folder.list <- dir(root_path)

  path_list <- lapply(folder.list, function(folder_name){
    file.root_path <- file.path(root_path, folder_name)
    folder.idx <- which(folder.list == folder_name)
    
    put_log1("Function: `img.file_path.get_list`:
Getting file path list from the following char's root folders:
%1", file.root_path)

    fpath.list <- list.files(file.root_path, 
                             full.names = TRUE)
    
    fpath.len <- length(fpath.list)

    if(!is.na(char_files.max) && fpath.len > char_files.max){
      if (!is.na(seed)) {
        set.seed(seed + folder.idx)
      }
      
      if(random_sample) {
        file.idx <- sample(fpath.len, size = char_files.max)
      } else {
        file.idx <- seq_len(char_files.max)
      }
      
      fpath.list <- fpath.list[file.idx]
    }
    
    put_log2("Function: `img.file_path.get_list`:
%1 files in folder: %2", length(fpath.list), folder_name)

    list(root_path = file.root_path,
         file_path.list = fpath.list)
  })
  
  names(path_list) <- folder.list |> substr(1,1)
  path_list
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

as.matrix.img28x28.list <- function(img.list, label) {
  mx.ncols <- 28*28
  
  map(img.list, as.vector) |>
    unlist() |>
    matrix(ncol = mx.ncols, 
           byrow = TRUE,
           dimnames = list(base::rep(label, times = length(img.list)),
                           1:mx.ncols))
}

## Data Visualization ---------------------------------
image.mx <- function(mx) {
  image(mx[, seq(ncol(mx), 1)])
}
char.image <- function(char.vector) {
  image(matrix(char.vector, nrow = 28)[, 28:1])
}

## Data processing functions -------------------------------
magick_img2matrix <- function(img){
  img.dat <- image_data(img)
  mx.bin <- img.dat[1,,]
  mode(mx.bin) <- "numeric"
  mx.bin /255
}

image.load_bin.shape28x28 <- function(file_path) {
  img0 <- image_read(file_path)
  # plot(img0)
  
  img0.mx <- magick_img2matrix(img0)
  
  if (max(img0.mx) > 0) {
    img0.trimmed <- image_trim(img0)
    # plot(img0.trimmed)
    
    im0.dim <- dim(image_data(img0.trimmed)) 
    im0.dim
    
    img28x28 <- image_resize(img0.trimmed, '28x28!')
    # dim(image_data(img28x28))
    # plot(img28x28)
    
    img.sharpen <- image_convolve(img28x28, 'DoG:0,0,2', scaling = '100, 100%')
    # plot(img.sharpen)
    
    img.mx <- magick_img2matrix(img.sharpen)

    img.bin <- img.mx > 0.5
    # image.mx(img.bin)
    img.bin
  }
}


img28x28mx2flatten.list <- function(img_list){
  start <- put_start_date()
  put_log("Function `img28x28mx2flatten.list`:
Converting image lists to matrices...")
  char_matrix.list <- lapply(names(img_list), function(label){
    put_log("Function `img28x28mx2flatten.list`:
Processing label: `%1`...", label)
    img_list[[label]]$img.list |> 
      as.matrix.img28x28.list(label)
  })
  
  put_log("Function `img28x28mx2flatten.list`:
Image matrix list has been created with the following structure:
%1", capture.output(str(char_matrix.list)))
  put_end_date(start)

  char_matrix.list
}

img28x28.list2flatten.mx <- function(img_list,
                                 shuffle.rows = FALSE,
                                 shuffle.seed = NA){
  
  char_matrix.list <- img28x28mx2flatten.list(img_list)

  start <- put_start_date()
  put_log("Function `img28x28.list2matrix`:
Combining image data to single matrix...")
  img.mx <- do.call(rbind, char_matrix.list)
  put_log("Function `img28x28.list2matrix`:
Combined image data matrix has been created with the following structure:
%1", capture.output(str(img.mx)))
  put_end_date(start)

  if (shuffle.rows) {
    put_log("Shuffling flatten matrix rows...")
    img.mx <- shuffle.mxrows(img.mx, shuffle.seed)
    put_log("The flatten matrix rows have been shuffled.")
  }
  
  img.mx
}

img.load.bin28x28mx.list <- function(root_path, 
                                     folder.list = NULL, 
                                     char_files.max = NA,
                                     char_files.seed = NA,
                                     random_sample = FALSE) {
  start <- put_start_date()
  put_log("Getting file path lists...")
  img.file_list <- img.file_path.get_list(root_path, 
                                          folder.list,
                                          char_files.max,
                                          char_files.seed,
                                          random_sample)
  put_log("File path lists have been created. The output list structure:
%1", capture.output(str(img.file_list)))
  put_end_date(start)
  
  label_list <- as.factor(names(img.file_list))
  
  start <- put_start_date()
  put_log("Loading image files...")
  img_list <- lapply(img.file_list, function(char.dir){
    list(img.list = map(char.dir$file_path.list, 
                        image.load_bin.shape28x28) |> compact(),
         fpath.list = char.dir$file_path.list)
  })
  put_end_date(start)
  put_log("Image files have been loaded. The output list structure:
%1", capture.output(str(img_list)))
  
  list(img.files = img.file_list,
       label.list = label_list,
       img.list = img_list)
}

split2bagging_sets <- function() {
  
}

ds.get_classIDs.grouped <- function(x) {
  y <- as.factor(rownames(x))
  
  g <- data.frame(classID = y) |> 
    group_by(classID) |>
    summarise(n = n())
  
  list(classID = y,
       groupByClass = g)
}

sample_test.idx <- function(x, 
                            seed = NA, 
                            test.ratio = 0.2,
                            train.balanced = TRUE,
                            bootstap.sample = FALSE) { 
  y.groups <- ds.get_classIDs.grouped(x)
  # str(y.groups)
  y.chars <- y.groups$groupByClass
  # str(y.chars)
  
  g_len.min <- min(y.chars$n)
  # 4261
  N <- nrow(x)
  
  put_log("Function: `sample_train_test_sets.mx`: 
Sampling 20% of the `x` x...")
  
  idx_group.list <- split(seq_len(N), y)
  # str(idx_group.list)
  
  if(train.balanced) {
    train.size <- g_len.min * (1 - test.ratio)
  }
  
  test.idx <-
    sapply(idx_group.list,
           function(idx_group) {
             if (!is.na(seed)) {
               set.seed(seed + idx_group[1])
             }
             grp_lenth <- length(idx_group)
             
             sample_size <- ifelse(train.balanced, 
                                   grp_lenth - train.size, 
                                   ceiling(grp_lenth * test_ratio))
             sample(idx_group, 
                    size = sample_size,
                    replace = bootstap.sample)
           }) |>
    unlist(use.names = FALSE) |>
    sort()
  
  test.idx
} 

sample_train_test_sets.mx <- function(x, 
                                      seed = NA, 
                                      test.ratio = 0.2,
                                      shuffle.test_rows = TRUE,
                                      shuffle.seed = NA,
                                      train.balanced = TRUE,
                                      bootstap.sample = FALSE) { 
  test.idx <- sample_test.idx(x, 
                              seed, 
                              test.ratio, 
                              train.balanced,
                              bootstap.sample)
  #str(test.idx)
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 80% of the original index set of `x` not used for the test Set...")
  
  train.set <- x[-test.idx,]
  # str(train.set)
  # dim(train.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `train.set`")

  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 20% of the original index set of `x` used for the test Set.")
  
  test.set <- x[test.idx,]
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

sample_train_test_sets.x3d <- function(x, 
                                      seed = NA, 
                                      test.ratio = 0.2,
                                      shuffle.test_rows = TRUE,
                                      shuffle.seed = NA,
                                      train.balanced = TRUE,
                                      bootstap.sample = FALSE) { 
  test.idx <- sample_test.idx(x, 
                              seed, 
                              test.ratio, 
                              train.balanced,
                              bootstap.sample)
  #str(test.idx)
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 80% of the original index set of `x` not used for the test Set...")
  
  train.set <- x[-test.idx,,]
  # str(train.set)
  # dim(train.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `train.set`")

  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 20% of the original index set of `x` used for the test Set.")
  
  test.set <- x[test.idx,,]
  # str(test.set)
  # dim(test.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `test.set`")

  if (shuffle.test_rows) {
    test.set <- shuffle.rows.x3d(test.set, shuffle.seed)
  }
  
  # Return result datasets
  list(train_set = train.set,
       test_set = test.set)
}

shuffle.mxrows <- function(x, seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
  random.idx <- sample(nrow(x))
  x[random.idx,]
}

shuffle.rows.x3d <- function(x, seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
  random.idx <- sample(nrow(x))
  x[random.idx,,]
}

splitDataset <- function(x, n.parts){
  x.size <- nrow(x)
  part.size <- as.integer(x.size / n.parts)
  
  start.idx <- 1
  end.idx <- start.idx + part.size - 1
  
  part.list <- list(x = list(), y = list())
  
  for (i in seq_len(n.parts)) {
    end.idx <-  ifelse(end.idx > x.size, x.size, end.idx)
    x.part <- x[seq(start.idx,end.idx, 1),]
    
    part.list$x[[i]] <- x.part
    part.list$y[[i]] <- as.factor(rownames(x.part))
    
    start.idx <- end.idx + 1
    
    if (start.idx > x.size) {
      break
    }
    end.idx <- start.idx + part.size - 1
  }
  part.list
}
