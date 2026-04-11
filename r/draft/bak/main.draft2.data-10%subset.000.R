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


### Open log: Load Train Data --------------------------------------------------
open_logfile(".load-train-data")
### Load Train Data ------------------------------------------------------------
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
  put_log1("Creating Train Data list from raw data files from root directory:
%1", img.train.root_path)
  start <- put_start_date()
  img.train.dat <- hwChar_data.load(img.train.root_path)
  
  put_log1("Train Data list structure:
%1", capture.output(str(img.train.dat)))
  
  put_log("Train Data list has been created from raw data files.")
  put_end_date(start)
  
  img.train.files <- img.train.dat$img.files
  train.labels <- img.train.dat$label.list
  img.train.list <- img.train.dat$img.list
  my_emnist.train <- img.train.dat$my_emnist
  
  
  put_log1("Saving Train Data to the cache file: 
%1", ds.train.list.file_path)
  start <- put_start_date()
  save(img.train.files,
       train.labels,
       img.train.list,
       my_emnist.train, 
       file = ds.train.list.file_path)
  put_log("Train Data list has been cached to the File System.")
  put_end_date(start)
  
  rm(img.train.dat)
}

put_log1("Train image file list structure:
%1", capture.output(str(img.train.files)))

put_log1("Train dataset labels:
%1", train.labels, .sep = " ")

put_log1("`img.train.list` data structure:
%1", capture.output(str(img.train.list)))

put_log1("My Extended MNIST-like dataset (matrix) dimensions: 
%1", dim(my_emnist.train), .sep = " ")

# Visualize the first char:
char.image(my_emnist.train[1,])

#### Clean up Environment -----------------------
rm(img.train.files)
rm(img.train.list)

#### Init `x` & `y` variables -------------------
x <- my_emnist.train
rm(my_emnist.train)

dim(x)
class(x)
str(x)
mean(rowMeans(x))

y <- as.factor(rownames(x))
str(y)
length(y)
mean(y)
mean(is.na(y))
max(is.na(y))

##### Clean up Environment -----------------------
rm(my_emnist.train)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset -------------
open_logfile(".split.train_set-10%train_subset")
#### Split Train Dataset to extract a train sample of 10% size -----------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x <- dim(x)
dim.x
dim.x[1]
dim.x[2]

test_ratio <- 0.9 
x.sample.seed <- dim.x[1]
x.sample.seed
test.sample.seed <- as.integer(x.sample.seed*test_ratio)
test.sample.seed

ds.0.1train.subset.file_path <- file.path(ds.subsets.path, "train-data.10%subset.RData")
ds.0.1train.subset.file_path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(ds.0.1train.subset.file_path)) {
  put_log1("Loading Split Train Data from cache file: 
%1", ds.0.1train.subset.file_path)
  
  start <- put_start_date()
  load(ds.0.1train.subset.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  start <- put_start_date()
  train.dataset.list <- sample_train_test_sets.mx(x, 
                                                  x.sample.seed,
                                                  test.ratio = test_ratio,
                                                  shuffle.test_rows = TRUE,
                                                  shuffle.seed = test.sample.seed)
  put_end_date(start)
  str(train.dataset.list)
  
  x.train <- train.dataset.list$train_set
  y.train <- as.factor(rownames(x.train))
  
  x.test <- train.dataset.list$test_set
  y.test <- as.factor(rownames(x.test))
  
  start <- put_start_date()
  put_log1("Caching data in the file
%1 ...", ds.0.1train.subset.file_path)
  
  save(x.train,
       y.train,
       x.test,
       y.test,
       file = ds.0.1train.subset.file_path)
  
  put_log1("The Train Data Subset objects has been cached in file:
`%1`", ds.0.1train.subset.file_path)
  put_end_date(start)
  
  rm(train.dataset.list)
}

stopCluster(cl)
stopImplicitCluster()


  dim(x.train)
  #> [1] 
  
  str(y.train)
  length(y.train)
  
  dim(x.test)
  #> [1] 166823    784

  str(y.test)
  length(y.test)

### Close Log ------------------------------------------------------------------
log_close()











#### Init `x4e3` & `y4e3` variables (Max 4e3 items per char class) -------------------
ch.labels <- train.labels4e3
x4e3 <- my_emnist.train.subset4e3
dim(x4e3)
class(x4e3)
str(x4e3)

y4e3 <- as.factor(rownames(x4e3))
str(y4e3)
length(y4e3)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset (Max 4e3 files per char class) -------------
open_logfile(".split-train-data-subset4e3")
### Split Train Dataset --------------------------------------------------------
dim.x4e3 <- dim(x4e3)
dim.x4e3
dim.x4e3[1]
dim.x4e3[2]

x4e3.sample.seed <- dim.x4e3[1]
x4e3.sample.seed
test.sample.seed <- as.integer(x4e3.sample.seed*0.2)
test.sample.seed

train.dataset.list <- sample_train_test_sets.mx(x4e3, 
                                                x4e3.sample.seed,
                                                shuffle.test_rows = TRUE,
                                                shuffle.seed = test.sample.seed)
str(train.dataset.list)

x4e3.train <- train.dataset.list$train_set
dim(x4e3.train)

y4e3.train <- as.factor(rownames(x4e3.train))
str(y4e3.train)
length(y4e3.train)

x4e3.test <- train.dataset.list$test_set
dim(x4e3.test)
#> [1] 166823    784

y4e3.test <- as.factor(rownames(x4e3.test))
str(y4e3.test)
length(y4e3.test)

### Close Log ------------------------------------------------------------------
log_close()

## Model Building --------------------------------------------------------------
### Open log: Train Model kNN+PCA ----------------------------------------------
open_logfile(".train-model.knn+pca")
### Training Model using the following methods: kNN, PCA -----------------------
# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path)) {
  dir.create(knn_pca.path)
}

k_values <- seq(1, 7, 2)
# k_values <- c(4, 5, 6)

dim_reduction.x4e3.train_knn_pca.file_path <-
  file.path(knn_pca.path, "dim-reduction.x4e3.train.k1-7.2nn+pca.RData")
 
# dim_reduction.x4e3.train_knn_pca.file_path <- 
#   file.path(knn_pca.path, "dim-reduction.x4e3.train.k4-5-6nn+pca.RData")

if (file.exists(dim_reduction.x4e3.train_knn_pca.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", dim_reduction.x4e3.train_knn_pca.file_path)
  
  start <- put_start_date()
  load(dim_reduction.x4e3.train_knn_pca.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  put_log("Training Model `kNN+PCA` on the dataset subset: `x4e3.train`..." )
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train_knn_pca <- caret::train(x4e3.train, y4e3.train, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = 5, p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = k_values))
  print_end_date(start)
  # Time difference of 10.72675 mins
  # Time difference of 8.459877 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x4e3.train`")
  
train_knn_pca.k1_7.2 <- train_knn_pca
# train_knn_pca.k4.5.6 <- train_knn_pca


  put_log("Saving Model in the cache file: `kNN+PCA`...")
  start <- put_start_date()
  save(train_knn_pca.k1_7.2, file = dim_reduction.x4e3.train_knn_pca.file_path)
  # save(train_knn_pca.k4.5.6, file = dim_reduction.x4e3.train_knn_pca.file_path)
  stopCluster(cl)
  stopImplicitCluster()
  print_end_date(start)
  rm(train_knn_pca)
  
  put_log1("The Model `kNN+PCA` trained on the dataset subset `x4e3.train` has been cached in file:
`%1`", dim_reduction.x4e3.train_knn_pca.file_path)
}

plot(train_knn_pca.k1_7.2)
train_knn_pca.k1_7.2
# k  Accuracy   Kappa    
# 1  0.7934615  0.7880263
# 3  0.8058974  0.8007895
# 5  0.8114744  0.8065132
# 7  0.8110256  0.8060526

# plot(train_knn_pca.k4.5.6)
# train_knn_pca.k4.5.6
# k  Accuracy   Kappa    
# 4  0.8061538  0.8010526
# 5  0.8093590  0.8043421
# 6  0.8092628  0.8042434


put_log("Predicting on `x4e3.test`")
start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)
y4e3_hat_knn_pca <- stats::predict(train_knn_pca.k1_7.2, x4e3.test, type = "raw")
# y4e3_hat_knn_pca <- stats::predict(train_knn_pca.k4.5.6, x4e3.test, type = "raw")
print_end_date(start)
# Time difference of 32.93572 secs

xy4e3.accuracy <- mean(y4e3_hat_knn_pca == y4e3.test)
xy4e3.accuracy
# train_knn_pca.k1_7.2
#> [1] 0.8226923
# train_knn_pca.k4.5.6
#> [1] 0.8238462

put_log2("Accuracy of `x4e3` model (trained and tested on subset of size %1 items):
%2", 4e3, xy4e3.accuracy)

put_log("Predicting on `x.test`")
start <- put_start_date()
y4e3_hat_knn_pca.x.test <- stats::predict(train_knn_pca.k1_7.2, x.test, type = "raw")
# y4e3_hat_knn_pca.x.test <- stats::predict(train_knn_pca.k4.5.6, x.test, type = "raw")
print_end_date(start)
# Time difference of 11.01372 mins

mean(y4e3_hat_knn_pca.x.test == y.test)
#> [1] 0.8410531

put_log2("Accuracy of `x4e3` model (trained on subset of size %1 items) tested on full-size test set:
%2", 4e3, xy4e3.accuracy)

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment (x4e3, y4e3) -------------------------------------------
# rm(x4e3)
# rm(x4e3.train)
# rm(x4e3.test)
# 
# rm(y4e3)
# rm(y4e3.train)
# rm(y4e3.test)

### Open log: Random Forest -------------------------------------
open_logfile(".model.random-forest")
### Random Forest --------------------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

#### Research ----------------------

research.randoms_forest.x4e3.fit_rf.nzv.mtry9.file_path <- file.path(models.random_forest.research.path, 
                                                                 "random-forest.x4e3.fit_rf.nzv.mtry9.RData")

if (file.exists(research.randoms_forest.x4e3.fit_rf.nzv.mtry9.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", research.randoms_forest.x4e3.fit_rf.nzv.mtry9.file_path)
  
  start <- put_start_date()
  load(research.randoms_forest.x4e3.fit_rf.nzv.mtry9.file_path)
  put_log("Train Data list has been loaded from cache.")
  
} else {
  
  start <- put_start_date()
  nzv <- nearZeroVar(x4e3.train)
  put_end_date(start)
  
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  system.time({fit_rf.nzv.mtry9 <- randomForest(x4e3.train[, -nzv], y4e3.train,  mtry = 9)})
 #    user  system elapsed 
 # 958.22    2.45  963.12 

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  #> Time difference of 16.08939 mins

  plot(fit_rf.nzv.mtry9)
  
  save(fit_rf.nzv.mtry9, 
       file = research.randoms_forest.x4e3.fit_rf.nzv.mtry9.file_path)
}

#### Random Forest (implementation) --------------------------------------------

randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path <- file.path(models.random_forest.path, 
                                                                 "x4e3.train_rf.nzv.mtry5_15.RData")

if (file.exists(randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path)
  
  start <- put_start_date()
  load(randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train.x4e3_rf.cv5.ntree200 <- caret::train(x4e3.train, y4e3.train, method = "rf", 
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
%1...", randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path)

  start <- put_start_date()
  save(train.x4e3_rf.cv5.ntree200, 
       file = randoms_forest.x4e3.train_rf.nzv.mtry5_15.file_path)
  put_log("The Train fit result has been cached on the local File System.")
  put_end_date(start)
}

plot(train.x4e3_rf.cv5.ntree200)
train.x4e3_rf.cv5.ntree200

put_log("Predicting values on the Test Set")
start <- put_start_date()
y_hat_rf <- stats::predict(train.x4e3_rf.cv5.ntree200, x4e3.test, type = "raw")
print_end_date(start)

mean(y_hat_rf == y4e3.test)
#> [1] 0.8473077

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment --------------------------------------------------------
rm(x4e3)
rm(x4e3.train)
rm(x4e3.test)

rm(y4e3)
rm(y4e3.train)
rm(y4e3.test)


rm(train_knn_pca)
rm(fit_rf.nzv.mtry9)
rm(train_rf)


# ---------------------------
# Reference:
#
# 

start <- put_start_date()
put_end_date(start)






























