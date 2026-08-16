#%%%%%%%%%%%%%%%%%%%%%%%
# Data Helper Functions 
#%%%%%%%%%%%%%%%%%%%%%%%

## Data loading ----------------------------------------------------------------
kaggle_cli.download <- function(dataset.dir, data.local_path, unzip = FALSE) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(trimws(paste("kaggle datasets download", kaggle_dataset, "--path", 
                          data.local_path, ifelse(unzip, "--unzip", "")))) != 0) {
    stop(str.build("Failed to download the dataset with Kaggle CLI: `%1`.", dataset.dir))
  }
  
  if(!dir.exists(data.local_path)) {
    stop(str.build("Failed to download and unzip the Kaggle dataset: `%1`.", dataset.dir))
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
  x <- readRDS(file_path)
  put_log("The flatten training dataset has been loaded from the following file:
%1", file_path)
  
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

image.load_bin.shape28x28 <- function(file_path) {
  img0 <- magick::image_read(file_path)
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
    put_log("Loading files from root directory:
%1...",char.dir$root_path)

    img_ls <- map(char.dir$file_path.list, 
                        image.load_bin.shape28x28) # |> compact(),
    
    valid.idx <- sapply(img_ls, function(img){
      sum(img) > 0
    })
    
    # sum(ivalid.idx)
    # bad_img <- img_ls[!ivalid.idx]
    # length(bad_img)
    # 
    # char.image(bad_img[[1]])

    ls <- list(img.list = img_ls[valid.idx], 
               fpath.list = char.dir$file_path.list[valid.idx])
    
    stopifnot(length(ls$img.list) == length(ls$fpath.list))

    put_log("Completed Loading files from directory:
%1.",char.dir$root_path)
    put_end_date(start)
    ls
  })
  
  put_log("Image files have been loaded. The output list structure:
%1", capture.output(str(img_list)))
  
  list(label.list = label_list,
       img.list = img_list)
}

load_datasets <- function(ds.file) {
  put_log("Function `load_datasets`:
Loading dataset from the backup file...")
  ds <- readRDS(ds.file)
  put_log("Function `load_datasets`:
The dataset has been loaded from the following backup file:
%1", ds.file)
  
  train <- list()
  
  train$x <- ds$train_set$x.train
  train$class_groups <- ds.get_classIDs.grouped(train$x)

  test <- list()
  
  test$x <- ds$test_set$x.test
  test$files <- ds$test_set$x.files
  
  test$class_groups <- ds.get_classIDs.grouped(test$x)

  list(train = train,
       test = test)
}

load.train_set <- function(backup.file) {
  ds <- load_datasets(backup.file)
  return(ds$train)
}

load.test_set <- function(backup.file) {
  ds <- load_datasets(backup.file)
  return(ds$test)
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
    
    put_log("Function: `img.file_path.get_list`:
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
    
    put_log("Function: `img.file_path.get_list`:
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

as.matrix.cimg <- function(cimg.list, class.label) {
  mx.ncols <- 28*28
  
  map(cimg.list, as.vector.cimg) |>
    unlist() |>
    matrix(ncol = mx.ncols, 
           byrow = TRUE,
           dimnames = list(base::rep(class.label, times = length(cimg.list)),
                           1:mx.ncols))
}

magick_img2matrix <- function(img){
  img.dat <- image_data(img)
  mx.bin <- img.dat[1,,]
  mode(mx.bin) <- "numeric"
  mx.bin /255
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

recognition_err.table <- function(predicted.values, 
                                  actual.values, 
                                  img.file_paths) {
  
  err.idx <- which(predicted.values != actual.values)
  
  data.frame(predicted = predicted.values[err.idx],
             actual = actual.values[err.idx],
             file = img.file_paths[err.idx])
}

print.image_grid <- function(err.table,
                             err_index.range = 1:30,
                             image.scale ="100",
                             font.size = 10,
                             font.color = "red4",
                             box.color = colors()[492], #"#F5F5DC",
                             tile = "5x6",
                             geometry = "x160+5+5"
                             ) {
  
  img_grid <- lapply(err_index.range, function(i) {
    image_read(err.table$file[i]) |>
      image_scale(image.scale) |>
      image_annotate(str_flatten(c(as.character(err.table$predicted[i]),
                                   " vs ",
                                   as.character(err.table$actual[i]),
                                   " (actual)")),
                     size = font.size, color = font.color,
                     boxcolor = box.color,
                     # location = "+1+2",
                     #gravity = "southwest"
      )
  }) |> 
    image_join() |>
    image_montage(tile = tile, geometry = geometry)
  
  image_info(img_grid)
  
  print(img_grid)
  # dev.off()
  img_grid
}

plot_image <- function(image_file) {

  img <- magick::image_read(image_file)
  plot(img)
}

## Data processing -------------------------------------------------------------
split.img28x28mx_array <- function(backup.file,
                                   seed = NA,
                                   test_ratio = 0.2) {
  stopifnot(file.exists(backup.file))
  
  put_log("Function `split.img28x28mx_array`:
Loading the Binary Image 28x28 array set from the backup file...")
  img28x28mx.set <- readRDS(backup.file)
  put_log("Function `split.img28x28mx_array`:
The Binary Image 28x28 array set has been loaded from the following file:
%1", backup.file)
  
  put_log("Function `split.img28x28mx_array`:
Splitting the Train 28x28 Image Data Array into a Train and Test Sets...")
  
  if(!is.na(seed)) {
    set.seed(seed)
  }

  split.list <- sample_train_test_sets.x3d(img28x28mx.set$img28x28mx.array,
                                           img28x28mx.set$img28x28mx.fpath,
                                           test.ratio = test_ratio)

  put_log("Function `split.img28x28mx_array`:
The Result Split Dataset object structure:
%1", capture.output(str(split.list)))
  
  return(split.list)
}

img.list2flatten_matrix <- function(img_list,
                                 shuffle.rows = FALSE,
                                 shuffle.seed = NA){
  
  char_matrix.list <- img.list2flatten_matrix.list(img_list)

  start <- put_start_date()
  put_log("Function `class_img.list2flatten_matrix`:
Combining image data to single matrix...")
  img.mx <- do.call(rbind, char_matrix.list)
  put_log("Function `class_img.list2flatten_matrix`:
Combined image data matrix has been created with the following structure:
%1", capture.output(str(img.mx)))
  put_end_date(start)
  
  file_path.ls <- lapply(img_list, function(item){
    item$fpath.list
  })
  
  img.file_path <- abind(file_path.ls)
  names(img.file_path) <- rownames(img.mx)

  if (shuffle.rows) {
    put_log("Shuffling flatten matrix rows...")
    img.mx <- shuffle.rows(img.mx, img.file_path, shuffle.seed)
    put_log("The flatten matrix rows have been shuffled.")
  }

  list(img.mx = img.mx,
       img.file_path = img.file_path)
}

img.list2flatten_matrix.list <- function(img_list){
  start <- put_start_date()
  put_log("Function `img.list2flatten_matrix.list`:
Converting image lists to matrices...")
  char_matrix.list <- lapply(names(img_list), function(class.label){
    put_log("Function `img.list2flatten_matrix.list`:
Processing class.label: `%1`...", class.label)
    img_list[[class.label]]$img.list |> 
      class_img.list2flatten_matrix(class.label)
  })
  names(char_matrix.list) <- names(img_list)
  
  put_log("Function `img.list2flatten_matrix.list`:
Image matrix list has been created with the following structure:
%1", capture.output(str(char_matrix.list)))
  put_end_date(start)

  char_matrix.list
}

class_img.list2flatten_matrix <- function(img.list, class.label) {
  mx.ncols <- n.img_rows*n.img_cols
  
  map(img.list, as.vector) |>
    unlist() |>
    matrix(ncol = mx.ncols, 
           byrow = TRUE,
           dimnames = list(base::rep(class.label, times = length(img.list)),
                           1:mx.ncols))
}

split2bagging_sets <- function() {
  
}

ds.get_classIDs.grouped <- function(x) {
  y <- as.factor(rownames(x))
  y.get_classIDs.grouped(y)
}

y.get_classIDs.grouped <- function(y) {
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
Sampling %1% of the dataset indices for a Test Set...", test.ratio*100)
  
  y.idx <- seq_len(N)
  idx_group.list <- split(y.idx, y)
  # str(idx_group.list)
  
  
  test.size.balanced <- ceiling(grp_len.min * test.ratio)
  train.size.balanced <- grp_len.min - test.size.balanced
  
  if(!is.na(seed))
    set.seed(seed)

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
      sapply(y.chars$classID,
               function(class.label) {
                 idx_grp <- idx_group.list[[class.label]]
                 test.idx_grp <- test.idx.groups[[class.label]]
                 # train.idx_grp <- setdiff(idx_grp, test.idx_grp)
                 setdiff(idx_grp, test.idx_grp) |>
                   sample(size = train.size.balanced,
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
  
  list(train.index = train.idx,
       test.index = test.idx)
} 

binClass.get_sample.idx.balanced <- function(x,
                                             class.label,
                                             seed = NA, 
                                             test.ratio = 0.2) {
  y <- as.factor(rownames(x))
  N <- nrow(x)
  
  put_log("Function: `binClass.get_sample.idx.balanced`: 
Sampling %1% of the dataset indices for a Test Set...", test.ratio*100)
  
  y.idx <- seq_len(N)
  idx_group.list <- split(y.idx, y)
  # str(idx_group.list)
  
  idx_group.class_excluded.idx <- idx_group.list[Y.Labels != class.label]
  
  y.class.idx <- y.idx[y == class.label]
  N.class <- length(y.class.idx) 
  
  if(!is.na(seed))
    set.seed(seed)
  
  y_other.idx <- y.idx[-y.class.idx]
  N.no.class <- length(y_other.idx)
  
  y_other.groups <- y.get_classIDs.grouped(y[y_other.idx])
  # str(y.groups)
  y_other.chars <- y_other.groups$groupByClass
  # str(y.chars)
  
  grp_len.min <- min(y_other.chars$n)
  
  N.class.balanced <- min(N.class, grp_len.min*(N.classes - 1))
  class.test.size.balanced <- ceiling(N.class.balanced * test.ratio)

  N.other_class.balanced <- ceiling(N.class.balanced / (N.classes - 1))
  other_class.test.size <- as.integer(class.test.size.balanced / (N.classes - 1))
  
  test_size.diff <- class.test.size.balanced - other_class.test.size * (N.classes - 1)  

  other.test_sizes <- rep(other_class.test.size, times = N.classes - 1)
  others.idx <- seq_len(N.classes - 1)
  
  increment.idx <- sample(others.idx, size = test_size.diff)

  other.test_sizes[increment.idx] <-  other_class.test.size + 1
  
  stopifnot(sum(other.test_sizes) == class.test.size.balanced)
  
  
  class.idx <- sample(y.class.idx, size = N.class.balanced)

  other_classes.idx.groups <- lapply(idx_group.class_excluded.idx, function(idx_group) {
    sample(idx_group, size = N.other_class.balanced)
  })

  class.test.idx <- sample(class.idx, size = class.test.size.balanced)
  
  put_log("Function: `binClass.get_sample.idx.balanced`: 
The size of the part of the Test Set belonging to the Class `%1` is %2.", 
          class.label,
          length(class.test.idx))
  
  class.train.idx <- setdiff(class.idx, class.test.idx)

  put_log("Function: `binClass.get_sample.idx.balanced`: 
The size of the part of the Training Set belonging to the Class `%1` is %2.", 
          class.label,
          length(class.train.idx))

  other_classes.test.idx <- lapply(others.idx, function(i) {
    sample(other_classes.idx.groups[[i]], size = other.test_sizes[i])
  }) |> unlist(use.names = FALSE) |> sort()
  
  stopifnot(length(class.test.idx) == length(other_classes.test.idx))
  
  put_log("Function: `binClass.get_sample.idx.balanced`: 
The size of the part of the Test Set belonging to all other classes (all except `%1`) is %2.", 
          class.label,
          length(other_classes.test.idx))

  other_classes.train.idx <- other_classes.idx.groups |>
    unlist(use.names = FALSE) |> 
    sort() |>
    setdiff(other_classes.test.idx)

  put_log("Function: `binClass.get_sample.idx.balanced`: 
The size of the part of the Training Set belonging to all other classes (all except `%1`) is %2.", 
          class.label,
          length(other_classes.train.idx))
  
  test.idx <- list(class.test.idx, 
                   other_classes.test.idx) |>
    unlist() |>
    sample()
  
  put_log("Function: `binClass.get_sample.idx.balanced`: 
The total size of the Test Set is %1.", 
          length(test.idx))
  
  
  train.idx <- list(class.train.idx, 
                    other_classes.train.idx) |>
    unlist() |>
    sample()
  
  put_log("Function: `binClass.get_sample.idx.balanced`: 
The total size of the Training Set is %1.", 
          length(train.idx))
  
  list(train.index = train.idx,
       test.index = test.idx)
} 

binClass.sample_train_test_sets.x3d <- function(x,
                                                x.files = NULL,
                                                class.label,
                                                seed = NA, 
                                                test.ratio = 0.2) { 
  dim.x <- dim(x)
  dim.x
  stopifnot(length(dim.x) == 3)

  sample.idx <- binClass.get_sample.idx.balanced(x,
                                                 class.label,
                                                 seed, 
                                                 test.ratio)
  
  if(!is.null(x.files)) {
    train.files <- x.files[sample.idx$train.index]
    test.files <- x.files[sample.idx$test.index]
  } else {
    train.files <- NULL
    test.files <- NULL
  }
  
  x_train <- x[sample.idx$train.index,,]
  y_train <- rownames(x_train)
  
  x_test <- x[sample.idx$test.index,,]
  y_test <- rownames(x_test)
  # browser()

  dim.x = dim(x_train)
    
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
The Train and Test Sets has been created with the following dimensions:
%1", capture.output(dim.x))

  # Add channel into the dimension
  x.train <- array_reshape(x_train, 
                           c(nrow(x_train), 
                             dim.x[2], 
                             dim.x[3], 
                             1))

  rownames(x.train) = y_train
  y.train <- (y_train  == class.label) |> as.integer() 
  
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
The Training Set has been reshaped as follows:
%1", capture.output(shape(x.train)))
  

    # Add channel into the dimension
  x.test <- array_reshape(x_test, 
                          c(nrow(x_test), 
                            dim.x[2], 
                            dim.x[3], 
                            1))

  rownames(x.test) <- y_test
  y.test <- (y_test  == class.label) |> as.integer() 
  
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
The Test Set has been reshaped as follows:
%1", capture.output(shape(x.test)))

  train.set <- list(x.train = x.train,
                    y.train = y.train,
                    x.files = train.files)
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
A training sample of size %1 rows has been generated with the following structure:
%2", nrow(train.set$x.train), capture.output(str(train.set)))
  
  test.set <- list(x.test = x.test,
                   y.test = y.test,
                   x.files = test.files)
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
A testing sample of size %1 rows has been generated with the following structure:
%2", nrow(test.set$x.test), capture.output(str(test.set)))

  list(train_set = train.set,
       test_set = test.set)
}

sample_train_test_sets.mx <- function(x,
                                      x.files = NULL,
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
  
  if(!is.null(x.files)) {
    train.files <- x.files[sample.idx$train.index]
    test.files <- x.files[sample.idx$test.index]
  } else {
    train.files <- NULL
    test.files <- NULL
  }
  
  train.size <- (1 - test.ratio) * 100
  test.size <- test.ratio * 100
  
  put_log("Function: `sample_train_test_sets.mx`: 
Generating a training sample of size %1% from the original dataset...",
          train.size)
  
  train.set <- list(x.train = x[sample.idx$train.index,],
                    x.files = train.files)

  put_log("Function: `sample_train_test_sets.mx`: 
A training sample of size %1% has been made with the following structure:
%2", train.size, capture.output(str(train.set)))
  
  put_log("Function: `sample_train_test_sets.mx`: 
Generating a testing sample of size %1% from the original dataset...",
          test.size)
  
  test.set <- list(x.test = x[sample.idx$test.index,],
                   x.files = test.files)
  
  put_log("Function: `sample_train_test_sets.x3d`: 
A testing sample of size %1% has been made with the following structure:
%2", test.size, capture.output(str(test.set)))
  
  
  list(train_set = train.set,
       test_set = test.set)
}

sample_train_test_sets.x3d <- function(x,
                                       x.files = NULL,
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

  if(!is.null(x.files)) {
    train.files <- x.files[sample.idx$train.index]
    test.files <- x.files[sample.idx$test.index]
  } else {
    train.files <- NULL
    test.files <- NULL
  }

  train.size <- (1 - test.ratio) * 100
  test.size <- test.ratio * 100

  put_log("Function: `sample_train_test_sets.x3d`: 
Generating a training sample of size %1% from the original dataset...",
          train.size)

  train.set <- list(x.train = x[sample.idx$train.index,,],
                    x.files = train.files)
  
  put_log("Function: `sample_train_test_sets.x3d`: 
A training sample of size %1% has been made with the following structure:
%2", train.size, capture.output(str(train.set)))
  
  put_log("Function: `sample_train_test_sets.mx`: 
Generating a testing sample of size %1% from the original dataset...",
          test.size)

  test.set <- list(x.test = x[sample.idx$test.index,,],
                    x.files = test.files)
  
  put_log("Function: `sample_train_test_sets.x3d`: 
A testing sample of size %1% has been made with the following structure:
%2", test.size, capture.output(str(test.set)))
  
  
  list(train_set = train.set,
       test_set = test.set)
}

shuffle.rows <- function(x,
                         x.files = NULL,
                         seed = NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
  random.idx <- sample(nrow(x))
  
  random.files <- ifelse(!is.null(x.files),
                         x.files[random.idx])
  list(x = x[random.idx,],
       x.files = random.files)
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
