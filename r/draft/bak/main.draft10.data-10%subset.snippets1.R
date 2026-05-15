# ---
ch_A.list <- img28x28bin.list[["A"]]$img.list
str(ch_A.list)

ch_A.mx <- class_img.list2flatten_matrix(ch_A.list, "A")
dim(ch_A.mx)
ch_A.dist <- dist(ch_A.mx)
str(ch_A.dist)
# 'dist' num [1:147997410] 9.11 9.85 12.04 10.77 12.08 ...
# - attr(*, "Size")= int 17205
# - attr(*, "Labels")= chr [1:17205] "A" "A" "A" "A" ...
# - attr(*, "Diag")= logi FALSE
# - attr(*, "Upper")= logi FALSE
# - attr(*, "method")= chr "euclidean"
# - attr(*, "call")= language dist(x = ch_A.mx)

ch_A.dist.mx <- as.matrix(ch_A.dist)

image(ch_A.dist.mx)


# ---------------------
ds.train.list.file_path <- file.path(train.data.path, "train-data-list.RData")
ds.train.list.file_path

if (file.exists(ds.train.list.file_path)) {
  put_log1("Loading Train Data from cache file: 
%1", ds.train.list.file_path)
  
  start <- put_start_date()
  load(ds.train.list.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  
  # img.train.dat <- hwChar_data.load(img.train.root_path)
  put_log("Train Data list has been created from raw data files.")
  put_end_date(start)
  
  
  img.train.files <- img.train.dat$img.files
  y.labels <- img.train.dat$label.list
  img.train.list <- img.train.dat$img.list
  my_emnist.train <- img.train.dat$my_emnist
  
  
  put_log1("Saving Train Data to the cache file: 
%1", ds.train.list.file_path)
  start <- put_start_date()
  save(img.train.files,
       y.labels,
       img.train.list,
       my_emnist.train, 
       file = ds.train.list.file_path)
  put_log("Train Data list has been cached to the File System.")
  put_end_date(start)
  
  rm(img.train.dat)
}


#start <- put_start_date()
put_log("Building flatten training dataset (`my_emnist`)...")
my_emnist <- class_img.list2flatten_matrix(img28x28bin.list$img.list)
put_end_date(start)
str(my_emnist)

# Converting image lists to char matrix list
flatten_mx.list <- img.list2flatten_matrix.list(img28x28bin.list$img.list)
# str(flatten_mx.list)

