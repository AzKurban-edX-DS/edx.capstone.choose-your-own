### Open log: Load Train Data --------------------------------------------------
open_logfile(".load-train-data")
### Load Train Data ------------------------------------------------------------
train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.RData")
train.img28x28bin.list.file_path

my_emnist.file_path <- file.path(train.data.path, "my_emnist.RData")
my_emnist.file_path

if (!file.exists(train.img28x28bin.list.file_path)) {
  put_log1("Creating Binary Image 28x28 list from raw data files from root directory:
%1", img.train.root_path)
  start <- put_start_date()
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  img28x28bin.list <- img.load.bin28x28mx.list(img.train.root_path)

  put_log1("Saving Binary Image 28x28 list to the cache file: 
%1", train.img28x28bin.list.file_path)
  start <- put_start_date()
  save(img28x28bin.list,
       file = train.img28x28bin.list.file_path)
  put_log("Binary Image 28x28 list has been saved to the cache file:
%1", train.img28x28bin.list.file_path)
  put_end_date(start)
  
} else {
  start <- put_start_date()
  put_log("Loading Binary Image 28x28 Matrix list from cache.")
  load(train.img28x28bin.list.file_path)
  put_log("The Binary Image 28x28 Matrix list has been loaded from cache.")
  put_end_date(start)
} 

if(!file.exists(my_emnist.file_path)){
  put_log1("LoadingBinary Image 28x28 list from cache file: 
%1", train.img28x28bin.list.file_path)

  put_log("Building flatten (`EMNIST`-like) dataset...")
  my_emnist <- img28x28.list2flatten.mx(img28xc28bin.list$img.list)
  put_log("The flatten dataset have been created.")
  str(my_emnist)

  put_log1("Saving flatten training dataset to the cache file: 
%1", my_emnist.file_path)
  start <- put_start_date()
  save(my_emnist,
       file = my_emnist.file_path)
  put_log("The flatten training dataset has been saved to the cache file:
%1", my_emnist.file_path)
  put_end_date(start)
} else {
  start <- put_start_date()
  put_log("Loading flatten training dataset from cache.")
  load(my_emnist.file_path)
  put_log("The flatten training dataset has been loaded from cache.")
  put_end_date(start)
}

put_log("Binary Image 28x28 list structure:
%1", capture.output(str(img28x28bin.list)))

y.labels <- img28x28bin.list$label.list

put_log("Train dataset labels:
%1", y.labels, .sep = " ")


### Close Log ---------------------------------------------------------------
log_close()

