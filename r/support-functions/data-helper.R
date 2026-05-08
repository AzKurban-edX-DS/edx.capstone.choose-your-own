###########################
# Data Helper Functions 
###########################

## Data loading ----------------------------------------------------------------
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

load.img28x28bin.list <- function(file) {
  if(!file.exists(file)) {
    stop("Function `load.img28x28bin.list`:
File for Binary Image 28x28 Matrix list DOES NOT EXIST:", file)
  }
  start <- put_start_date()
  put_log("Function `load.img28x28bin.list`:
Loading Binary Image 28x28 Matrix list from the backup file...")
  load(file, envir = .GlobalEnv)

  if(!exists("img28x28bin.list"))
    stop("Function `load.img28x28bin.list`:
File `", file, 
         "` does not contain the object `img28x28bin.list`.")
  
  put_log("Function `load.img28x28bin.list`:
The Binary Image 28x28 Matrix list has been loaded from the following file:
%1", file)
  put_log("Function `load.img28x28bin.list`:
The Binary Image 28x28 Matrix list Summary:
%1", capture.output(summary(img28x28bin.list)))
  put_end_date(start)
}


load.img_data.train.flatten <- function(file_path) {
  start <- put_start_date()
  put_log("Loading flatten training dataset from the backup file...")
  load.data( "my_emnist", file = file_path)
  put_log("The flatten training dataset has been loaded from the following file:
%1", file_path)
  
  x <- my_emnist
  # class identifies
  y.groups <- ds.get_classIDs.grouped(x)
  put_end_date(start)
  
  list(x = x,
       y_grouped = y.groups)
}

load.data.global <- function (..., file) {
  load.data(..., file = file, envir = .GlobalEnv)
}

load.data <- function(..., file, envir = parent.frame()) {
  stopifnot(file.exists(file))
  load(file, envir = envir)
  
  arg_ls <- list(...)
  
  for (object.name in arg_ls) {
    put_log("Function `load.data`:
Checking for object existence: `%1`", object.name)
    stopifnot(exists(object.name, envir = envir))
    put_log("Function `load.data`:
The `%1` object exists.", object.name)
  }
}
## Image Processing ------------------------------------------------------------
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

## Data Visualization ---------------------------------
image.mx <- function(mx) {
  image(mx[, seq(ncol(mx), 1)])
}
char.image <- function(char.vector) {
  image(matrix(char.vector, nrow = 28)[, 28:1])
}

data.plot <- function(data, 
                      title, 
                      xname, 
                      yname, 
                      xlabel = NULL, 
                      ylabel = NULL,
                      line_col = "blue",
                      # scale = 1,
                      normalize = FALSE) {
  y <- data[, yname]
  
  if (normalize) {
    y <- y - min(y)
  }
  
  if (is.null(xlabel)) {
    xlabel = xname
  }
  if (is.null(ylabel)) {
    ylabel = yname
  }
  
  aes_mapping <- aes(x = data[, xname], y = y)
  
  data |> 
    ggplot(mapping = aes_mapping) +
    ggtitle(title) +
    xlab(xlabel) +
    ylab(ylabel) +
    geom_point() + 
    geom_line(color=line_col)
}

## Data processing functions ---------------------------------------------------
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

img28x28mx2flatten.list <- function(img_list){
  start <- put_start_date()
  put_log("Function `img28x28mx2flatten.list`:
Converting image lists to matrices...")
  char_matrix.list <- lapply(names(img_list), function(label){
    put_log("Function `img28x28mx2flatten.list`:
Processing label: `%1`...", label)
    img_list[[label]]$img.list |> 
      img28x28.list2matrix(label)
  })
  names(char_matrix.list) <- names(img_list)
  
  put_log("Function `img28x28mx2flatten.list`:
Image matrix list has been created with the following structure:
%1", capture.output(str(char_matrix.list)))
  put_end_date(start)

  char_matrix.list
}

img28x28.list2matrix <- function(img.list, label) {
  mx.ncols <- 28*28
  
  map(img.list, as.vector) |>
    unlist() |>
    matrix(ncol = mx.ncols, 
           byrow = TRUE,
           dimnames = list(base::rep(label, times = length(img.list)),
                           1:mx.ncols))
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

get_sample.idx <- function(x, 
                           seed = NA, 
                           test.ratio = 0.2,
                           shuffle_rows = TRUE,
                           balanced = TRUE,
                           bootstap.sample = FALSE) {
  y <- as.factor(rownames(x))
  y.groups <- ds.get_classIDs.grouped(x)
  # str(y.groups)
  y.chars <- y.groups$groupByClass
  # str(y.chars)
  
  grp_len.min <- min(y.chars$n)
  # 4261
  N <- nrow(x)
  
  put_log("Function: `get_sample.idx`: 
Sampling %1% of the dataset indices for a Test Set...", test.ratio)
  
  y.idx <- seq_len(N)
  idx_group.list <- split(y.idx, y)
  # str(idx_group.list)
  
  
  test.size.balanced <- ceiling(grp_len.min * test.ratio)
  train.size.balanced <- grp_len.min - test.size.balanced
  set_seed(seed)

  test.idx.groups <-
    lapply(idx_group.list,
           function(idx_group) {
             sample(idx_group, 
                    size = ifelse(balanced, 
                                  test.size.balanced,
                                  ceiling(length(idx_group) * test.ratio)),
                    replace = bootstap.sample)
           })
  
  test.idx <- test.idx.groups |>
    unlist(use.names = FALSE) |>
    sort()

  if(balanced) {
    train.idx <-
      sapply(test.idx.groups,
             function(idx_group) {
               sample(idx_group, 
                      size = train.size.balanced,
                      replace = bootstap.sample)
             }) |>
      unlist(use.names = FALSE) |>
      sort()
  } else {
    train.idx <- y.idx[-test.idx]
  }
  
  if(shuffle_rows) {
    train.idx <- sample(train.idx)
    test.idx <- sample(test.idx)
  }
  
  c(train = train.idx,
    test = test.idx)
} 

binclass.get_sample.idx <- function(x,
                                    label,
                                    seed = NA, 
                                    test.ratio = 0.2,
                                    bootstap.sample = FALSE) {
  y <- as.factor(rownames(x))
  N <- length(y)
  
  put_log("Function: `binclass.get_sample.idx`: 
Sampling %1% of the dataset indices for a Test Set...", test.ratio*100)
  
  y.idx <- seq_len(N)
  idx_group.list <- split(y.idx, y)
  # str(idx_group.list)
  y_lbl.idx <- y.idx[y == label]
  N.lbl <- length(y_lbl.idx) 
  set_seed(seed)

  y_other.idx <- sample(y.idx[-y_lbl.idx])
  N.no_lbl <- length(y_other.idx)

  lbl.test.size <- ceiling(N.lbl * test.ratio)
  lbl.train.size <- N.lbl - lbl.test.size

  t.idx <- sample(seq_len(N.lbl), 
                  size = lbl.test.size,
                  replace = bootstap.sample)
  
  lbl.test.idx = y_lbl.idx[t.idx]
  
  put_log("Function: `binclass.get_sample.idx`: 
Size of labeled part of Test Set is %1", length(lbl.test.idx))
  
  no_lbl.t.idx <- sample(seq_len(N.no_lbl),
                         size = lbl.test.size,
                         replace = bootstap.sample)
  
   other.test.idx <- y_other.idx[no_lbl.t.idx]
  
  put_log("Function: `binclass.get_sample.idx`: 
Size of other part of Test Set is %1", length(other.test.idx))

  test.idx <- list(lbl.test.idx, other.test.idx) |>
    unlist() |>
    sample()
  
  put_log("Function: `binclass.get_sample.idx`: 
Total size of Test Set  is %1", length(test.idx))
  
  put_log("Function: `binclass.get_sample.idx`: 
The size of the labeled part of the Test Set is equal to the size of the remaining part: 
%1", sum(y[test.idx] == label) == sum(y[test.idx] != label))
  
  lbl.train.idx <- y_lbl.idx[-t.idx]
  
  put_log("Function: `binclass.get_sample.idx`: 
Size of labeled part of Train Set is %1", length(lbl.train.idx))
  
  other.train.idx <- sample(y_other.idx[-no_lbl.t.idx],
                            size = lbl.train.size,
                            replace = bootstap.sample)
  
  put_log("Function: `binclass.get_sample.idx`: 
Size of other part of Train Set is %1", length(other.train.idx))

  train.idx <- list(lbl.train.idx, other.train.idx) |>
    unlist() |>
    sample()
  
  put_log("Function: `binclass.get_sample.idx`: 
Total size of Train Set  is %1", length(train.idx))
  
  put_log("Function: `binclass.get_sample.idx`: 
The size of the labeled part of the Train Set is equal to the size of the remaining part: 
%1", sum(y[train.idx] == label) == sum(y[train.idx] != label))

  list(train = train.idx,
       test = test.idx)
} 

cnn.binclass.sample_sets <- function(x, 
                                 label,
                                 seed = NA, 
                                 test.ratio = 0.2,
                                 bootstap.sample = FALSE) { 
  dim.x <- dim(x)
  dim.x
  stopifnot(length(dim.x) == 3)

  sample.idx <- binclass.get_sample.idx(x,
                                        label,
                                        seed, 
                                        test.ratio,
                                        bootstap.sample)
  
  put_log("Function: `binclass.sample_sets`: 
Extracting %1% of the original index set of `x` not used for the test Set...",
          (1 - test.ratio)*100)
  
  x_train <- x[sample.idx$train,,]
  # browser()
  put_log("Function: `binclass.sample_sets`: 
`x_train` object structure is as follows:
%1", capture.output(str(x_train)))

  put_log("Function: `binclass.sample_sets`: 
`x_train` object dimentions are as follows:
%1", capture.output(dim(x_train)))

  y_train <- (rownames(x_train))
  y.train <- (y_train  == label) |> as.integer() 
  
  str(y.train)
  sum(y.train)
  length(y.train)
    
  # Add channel into the dimension
  x.train <- array_reshape(x_train, 
                           c(nrow(x_train), 
                             dim.x[2], 
                             dim.x[3], 
                             1))
  
  rownames(x.train) <- y_train
  
  put_log("Function: `binclass.sample_sets`:
`x.train` dataset object created with the following structure:
%1", capture.output(str(x.train)))

  put_log("Function: `binclass.sample_sets`: 
`x.train` object dimentions are as follows:
%1", capture.output(dim(x.train)))
  
  put_log("Function: `binclass.sample_sets`: 
Extracting 20% of the original index set of `x` used for the test Set.")
  
  x_test <- x[sample.idx$test,,]
  # browser()
  put_log("Function: `binclass.sample_sets`: 
`x_test` object structure is as follows:
%1", capture.output(str(x_test)))
  
  put_log("Function: `binclass.sample_sets`: 
`x_test` object dimentions are as follows:
%1", capture.output(dim(x_test)))
  # browser()

  y_test <- (rownames(x_test))
  y.test <- (y_test  == label) |> as.integer() 
  
  str(y.test)
  sum(y.test)
  length(y.test)
  
  
  # Add channel into the dimension
  x.test <- array_reshape(x_test, 
                          c(nrow(x_test), 
                            dim.x[2], 
                            dim.x[3], 
                            1))

  rownames(x.test) <- y_test
  
  put_log("Function: `binclass.sample_sets`:
`x.test` dataset object created with the following structure:
%1", capture.output(str(x.test)))
  
  put_log("Function: `binclass.sample_sets`: 
`x.test` object dimentions are as follows:
%1", capture.output(dim(x.test)))
  
  put_log("Function: `binclass.sample_sets`: Dataset created: `test.set`")
  
  # Return result datasets
  list(x.train = x.train,
       y.train = y.train,
       x.test = x.test,
       y.test = y.test)
}

sample_train_test_sets.mx <- function(x, 
                                      seed = NA, 
                                      test.ratio = 0.2,
                                      shuffle_rows = TRUE,
                                      balanced = TRUE,
                                      bootstap.sample = FALSE) { 
  sample.idx <- get_sample.idx(x, 
                              seed, 
                              test.ratio,
                              shuffle_rows,
                              balanced,
                              bootstap.sample)
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 80% of the original index set of `x` not used for the test Set...")
  
  train.set <- x[sample.idx["train"],]
  # str(train.set)
  # dim(train.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `train.set`")

  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 20% of the original index set of `x` used for the test Set.")
  
  test.set <- x[sample.idx["test"],]
  # str(test.set)
  # dim(test.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: `test.set`")

  # Return result datassets
  list(train_set = train.set,
       test_set = test.set)
}

sample_train_test_sets.x3d <- function(x, 
                                      seed = NA, 
                                      test.ratio = 0.2,
                                      shuffle_rows = TRUE,
                                      balanced = TRUE,
                                      bootstap.sample = FALSE) { 
  sample.idx <- get_sample.idx(x, 
                               seed, 
                               test.ratio, 
                               shuffle_rows,
                               balanced,
                               bootstap.sample)
  put_log("Function: `sample_train_test_sets.x3d`: 
Extracting 80% of the original index set of `x` not used for the test Set...")
  
  train.set <- x[sample.idx["train"],,]
  # str(train.set)
  # dim(train.set)
  
  put_log("Function: `sample_train_test_sets.x3d`: Dataset created: `train.set`")

  put_log("Function: `sample_train_test_sets.x3d`: 
Extracting 20% of the original index set of `x` used for the test Set.")
  
  test.set <- x[sample.idx["test"],,]
  # str(test.set)
  # dim(test.set)
  
  put_log("Function: `sample_train_test_sets.x3d`: Dataset created: `test.set`")

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

## Utility Functions -----------------------------------------------------------

set_seed <- function(seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
}
set_seed.increment <- function(seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
    seed <- seed + 1
  }
  seed
}
