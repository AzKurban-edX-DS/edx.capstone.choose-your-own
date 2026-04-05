# Main Script (Data Subset) ------------------------------------------------------------------

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Logging Helper Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-functions.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.path, "data-helper.funcs.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


### Open log: Load Train Data Subset (Max 512 files per char class) -------------
open_logfile(".load-train-data.subset512-rows")
### Load Train Data Subset (Max 512 files per char class) -----------------------
char_files.max512 <- 512 
char_files.max512

ds.train.subset512.file_path <- file.path(ds.subsets.path, "train-data-subset512.RData")
ds.train.subset512.file_path

if (file.exists(ds.train.subset512.file_path)) {
  put_log1("Loading Train Data subset (Max 512 files per char class) from cache file: 
%1", ds.train.subset512.file_path)
  
  start <- put_start_date()
  load(ds.train.subset512.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  put_log1("Creating Train Data subset (Max 512 files per char class) list 
from raw data files from root directory:
%1", img.train.root_path)
  
  start <- put_start_date()
  train.dat.subset512 <- hwChar_data.load(img.train.root_path, 
                                          char_files.max = char_files.max512,
                                          char_files.seed = char_files.max512)
  
  put_log1("Train Data subset (Max 512 files per char class) list structure:
%1", capture.output(str(train.dat.subset512)))
  
  put_log("Train Data subset (Max 512 files per char class) list 
has been created from raw data files.")
  put_end_date(start)
  
  train.files.subset512 <- train.dat.subset512$img.files
  train.labels512 <- train.dat.subset512$label.list
  train.images.subset512 <- train.dat.subset512$img.list
  hwChars.mnist.train.subset512 <- train.dat.subset512$hwChars.mnist
  
  rm(train.dat.subset512)
  
  put_log1("Saving Train Data subset (Max 512 files per char class) to the cache file: 
%1", ds.train.subset512.file_path)
  start <- put_start_date()
  save(train.files.subset512,
       train.labels512,
       train.images.subset512,
       hwChars.mnist.train.subset512, 
       file = ds.train.subset512.file_path)
  put_log("Train Data subset (Max 512 files per char class) list has been cached to the File System.")
  put_end_date(start)
}

put_log1("Train image file list subset (Max 512 files per char class) structure:
%1", capture.output(str(train.files.subset512)))

put_log1("Train dataset labels:
%1", train.labels512, .sep = " ")

put_log1("`train.images.subset512` data structure:
%1", capture.output(str(train.images.subset512)))

put_log1("`hwChars.mnist.train.subset512` dataset matrix dimensions: 
%1", dim(hwChars.mnist.train.subset512), .sep = " ")

# Visualize the first char:
char.image(hwChars.mnist.train.subset512[1,])

# rm(train.files.subset512)
# rm(train.images.subset512)
# rm(hwChars.mnist.train.subset512)

#### Init `x512` & `y512` variables (Max 512 items per char class) -------------------
ch.labels <- train.labels512
x512 <- hwChars.mnist.train.subset512
dim(x512)
class(x512)
str(x512)

y512 <- as.factor(rownames(x512))
str(y512)
length(y512)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset (Max 512 files per char class) -------------
open_logfile(".split-train-data-subset512")
### Split Train Dataset --------------------------------------------------------
dim.x512 <- dim(x512)
dim.x512
dim.x512[1]
dim.x512[2]

x512.sample.seed <- dim.x512[1]
x512.sample.seed
test.sample.seed <- as.integer(x512.sample.seed*0.2)
test.sample.seed

train.dataset.list <- sample_train_test_sets.mx(x512, 
                                                x512.sample.seed,
                                                shuffle.test_rows = TRUE,
                                                shuffle.seed = test.sample.seed)
str(train.dataset.list)

x512.train <- train.dataset.list$train_set
dim(x512.train)

y512.train <- as.factor(rownames(x512.train))
str(y512.train)
length(y512.train)

x512.test <- train.dataset.list$test_set
dim(x512.test)
#> [1] 166823    784

y512.test <- as.factor(rownames(x512.test))
str(y512.test)
length(y512.test)

### Close Log ------------------------------------------------------------------
log_close()

## Model Building -------------------------------------------------------------

### Open log: Dimension reduction with PCA -------------------------------------
open_logfile(".model.dim-reduction-pca")
#### Dimension reduction with PCA --------------------------------
# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca


start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

train_knn_pca <- caret::train(x512.train, y512.train, method = "knn", 
                       preProcess = c("nzv", "pca"),
                       trControl = trainControl("cv", number = 20, p = 0.95,
                                                preProcOptions = list(thresh = 0.9)),
                       tuneGrid = data.frame(k = seq(1, 7, 2)))
stopCluster(cl)
stopImplicitCluster()
print_end_date(start)
train_knn_pca

start <- put_start_date()
y_hat_knn_pca <- stats::predict(train_knn_pca, x512.test, type = "raw")
print_end_date(start)

mean(y_hat_knn_pca == y512.test)
#> [1] 0.7806821

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment --------------------------------------------------------
rm(x512)
rm(x512.train)
rm(x512.test)

rm(y512)
rm(y512.train)
rm(y512.test)
# ---------------------------
# Reference:
#
# 































