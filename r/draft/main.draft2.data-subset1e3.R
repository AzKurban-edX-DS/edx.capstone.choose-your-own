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


### Open log: Load Train Data Subset (Max 1e3 files per char class) -------------
open_logfile(".load-train-data.subset1e3-rows")
### Load Train Data Subset (Max 1e3 files per char class) -----------------------
char_files.max1e3 <- 1e3 
char_files.max1e3

ds.train.subset1e3.file_path <- file.path(ds.subsets.path, "train-data-subset1e3.RData")
ds.train.subset1e3.file_path

if (file.exists(ds.train.subset1e3.file_path)) {
  put_log1("Loading Train Data subset (Max 1e3 files per char class) from cache file: 
%1", ds.train.subset1e3.file_path)
  
  start <- put_start_date()
  load(ds.train.subset1e3.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  put_log1("Creating Train Data subset (Max 1e3 files per char class) list 
from raw data files from root directory:
%1", img.train.root_path)
  
  start <- put_start_date()
  train.dat.subset1e3 <- hwChar_data.load(img.train.root_path, 
                                          char_files.max = char_files.max1e3,
                                          char_files.seed = char_files.max1e3)
  
  put_log1("Train Data subset (Max 1e3 files per char class) list structure:
%1", capture.output(str(train.dat.subset1e3)))
  
  put_log("Train Data subset (Max 1e3 files per char class) list 
has been created from raw data files.")
  put_end_date(start)
  
  train.files.subset1e3 <- train.dat.subset1e3$img.files
  train.labels1e3 <- train.dat.subset1e3$label.list
  train.images.subset1e3 <- train.dat.subset1e3$img.list
  hwChars.mnist.train.subset1e3 <- train.dat.subset1e3$hwChars.mnist
  
  rm(train.dat.subset1e3)
  
  put_log1("Saving Train Data subset (Max 1e3 files per char class) to the cache file: 
%1", ds.train.subset1e3.file_path)
  start <- put_start_date()
  save(train.files.subset1e3,
       train.labels1e3,
       train.images.subset1e3,
       hwChars.mnist.train.subset1e3, 
       file = ds.train.subset1e3.file_path)
  put_log("Train Data subset (Max 1e3 files per char class) list has been cached to the File System.")
  put_end_date(start)
}

put_log1("Train image file list subset (Max 1e3 files per char class) structure:
%1", capture.output(str(train.files.subset1e3)))

put_log1("Train dataset labels:
%1", train.labels1e3, .sep = " ")

put_log1("`train.images.subset1e3` data structure:
%1", capture.output(str(train.images.subset1e3)))

put_log1("`hwChars.mnist.train.subset1e3` dataset matrix dimensions: 
%1", dim(hwChars.mnist.train.subset1e3), .sep = " ")

# Visualize the first char:
char.image(hwChars.mnist.train.subset1e3[1,])

# rm(train.files.subset1e3)
# rm(train.images.subset1e3)
# rm(hwChars.mnist.train.subset1e3)

#### Init `x1e3` & `y1e3` variables (Max 1e3 items per char class) -------------------
ch.labels <- train.labels1e3
x1e3 <- hwChars.mnist.train.subset1e3
dim(x1e3)
class(x1e3)
str(x1e3)

y1e3 <- as.factor(rownames(x1e3))
str(y1e3)
length(y1e3)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset (Max 1e3 files per char class) -------------
open_logfile(".split-train-data-subset1e3")
### Split Train Dataset --------------------------------------------------------
dim.x1e3 <- dim(x1e3)
dim.x1e3
dim.x1e3[1]
dim.x1e3[2]

x1e3.sample.seed <- dim.x1e3[1]
x1e3.sample.seed
test.sample.seed <- as.integer(x1e3.sample.seed*0.2)
test.sample.seed

train.dataset.list <- sample_train_test_sets.mx(x1e3, 
                                                x1e3.sample.seed,
                                                shuffle.test_rows = TRUE,
                                                shuffle.seed = test.sample.seed)
str(train.dataset.list)

x1e3.train <- train.dataset.list$train_set
dim(x1e3.train)

y1e3.train <- as.factor(rownames(x1e3.train))
str(y1e3.train)
length(y1e3.train)

x1e3.test <- train.dataset.list$test_set
dim(x1e3.test)
#> [1] 166823    784

y1e3.test <- as.factor(rownames(x1e3.test))
str(y1e3.test)
length(y1e3.test)

### Close Log ------------------------------------------------------------------
log_close()

## Model Building -------------------------------------------------------------

### Open log: Dimension reduction with PCA -------------------------------------
open_logfile(".model.dim-reduction-pca")
### Dimension reduction with PCA --------------------------------
# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca


dim_reduction.x1e3.train_knn_pca.file_path <- file.path(models.path, "dim-reduction.x1e3.train_knn_pca.RData")

if (file.exists(dim_reduction.x1e3.train_knn_pca.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", dim_reduction.x1e3.train_knn_pca.file_path)
  
  start <- put_start_date()
  load(dim_reduction.x1e3.train_knn_pca.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train_knn_pca <- caret::train(x1e3.train, y1e3.train, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = 20, p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = seq(1, 7, 2)))
  stopCluster(cl)
  stopImplicitCluster()
  print_end_date(start)
  
  save(train_knn_pca, file = dim_reduction.x1e3.train_knn_pca.file_path)
}

plot(train_knn_pca)
train_knn_pca

start <- put_start_date()
y_hat_knn_pca <- stats::predict(train_knn_pca, x1e3.test, type = "raw")
print_end_date(start)

mean(y_hat_knn_pca == y1e3.test)
#> [1] 0.8226923

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment --------------------------------------------------------
rm(x1e3)
rm(x1e3.train)
rm(x1e3.test)

rm(y1e3)
rm(y1e3.train)
rm(y1e3.test)

### Open log: Random Forest -------------------------------------
open_logfile(".model.random-forest")
### Random Forest --------------------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

#### Research ----------------------

research.randoms_forest.x1e3.fit_rf.nzv.mtry9.file_path <- file.path(models.random_forest.research.path, 
                                                                 "random-forest.x1e3.fit_rf.nzv.mtry9.RData")

if (file.exists(research.randoms_forest.x1e3.fit_rf.nzv.mtry9.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", research.randoms_forest.x1e3.fit_rf.nzv.mtry9.file_path)
  
  start <- put_start_date()
  load(research.randoms_forest.x1e3.fit_rf.nzv.mtry9.file_path)
  put_log("Train Data list has been loaded from cache.")
  
} else {
  
  start <- put_start_date()
  nzv <- nearZeroVar(x1e3.train)
  put_end_date(start)
  
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  system.time({fit_rf.nzv.mtry9 <- randomForest(x1e3.train[, -nzv], y1e3.train,  mtry = 9)})
 #    user  system elapsed 
 # 958.22    2.45  963.12 

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  #> Time difference of 16.08939 mins

  plot(fit_rf.nzv.mtry9)
  
  save(fit_rf.nzv.mtry9, 
       file = research.randoms_forest.x1e3.fit_rf.nzv.mtry9.file_path)
}

#### Random Forest (implementation) --------------------------------------------

randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path <- file.path(models.random_forest.path, 
                                                                 "x1e3.train_rf.nzv.mtry5_15.RData")

if (file.exists(randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path)
  
  start <- put_start_date()
  load(randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train.x1e3_rf.cv5.ntree200 <- caret::train(x1e3.train, y1e3.train, method = "rf", 
                           preProcess = "nzv",
                           trControl = trainControl(method = "cv", number = 5, p = .8),
                           ntree = 200,
                           tuneGrid = data.frame(mtry = seq(5, 15)))
  
  stopCluster(cl)
  stopImplicitCluster()
  print_end_date(start)
# [1] "Sun Apr  5 08:29:02 2026"
# Time difference of 57.23111 mins

put_log1("Saving Train fit result to the cache file:
%1...", randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path)

  start <- put_start_date()
  save(train.x1e3_rf.cv5.ntree200, 
       file = randoms_forest.x1e3.train_rf.nzv.mtry5_15.file_path)
  put_log("The Train fit result has been cached on the local File System.")
  put_end_date(start)
}

plot(train.x1e3_rf.cv5.ntree200)
train.x1e3_rf.cv5.ntree200

put_log("Predicting values on the Test Set")
start <- put_start_date()
y_hat_rf <- stats::predict(train.x1e3_rf.cv5.ntree200, x1e3.test, type = "raw")
print_end_date(start)

mean(y_hat_rf == y1e3.test)
#> [1] 0.8473077

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment --------------------------------------------------------
rm(x1e3)
rm(x1e3.train)
rm(x1e3.test)

rm(y1e3)
rm(y1e3.train)
rm(y1e3.test)


rm(train_knn_pca)
rm(fit_rf.nzv.mtry9)
rm(train_rf)


# ---------------------------
# Reference:
#
# 

start <- put_start_date()
put_end_date(start)






























