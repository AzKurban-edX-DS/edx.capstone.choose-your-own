# Main Script (Data Subset) ------------------------------------------------------------------

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Logging Helper Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-helper.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.path, "data-helper.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


### Open log: Load Train Data Subset (Max 2e3 files per char class) -------------
open_logfile(".load-train-data.subset2e3-rows")
### Load Train Data Subset (Max 2e3 files per char class) -----------------------
char_files.max2e3 <- 2e3 
char_files.max2e3

ds.train.subset2e3.file_path <- file.path(ds.subsets.path, "train-data-subset2e3.RData")
ds.train.subset2e3.file_path

if (file.exists(ds.train.subset2e3.file_path)) {
  put_log1("Loading Train Data subset (Max 2e3 files per char class) from cache file: 
%1", ds.train.subset2e3.file_path)
  
  start <- put_start_date()
  load(ds.train.subset2e3.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  put_log1("Creating Train Data subset (Max 2e3 files per char class) list 
from raw data files from root directory:
%1", img.train.root_path)
  
  start <- put_start_date()
  train.dat.subset2e3 <- hwChar_data.load(img.train.root_path, 
                                          char_files.max = char_files.max2e3,
                                          char_files.seed = char_files.max2e3)
  
  put_log1("Train Data subset (Max 2e3 files per char class) list structure:
%1", capture.output(str(train.dat.subset2e3)))
  
  put_log("Train Data subset (Max 2e3 files per char class) list 
has been created from raw data files.")
  put_end_date(start)
  
  train.files.subset2e3 <- train.dat.subset2e3$img.files
  train.labels2e3 <- train.dat.subset2e3$label.list
  train.images.subset2e3 <- train.dat.subset2e3$img.list
  my_emnist.train.subset2e3 <- train.dat.subset2e3$my_emnist
  
  put_log1("Saving Train Data subset (Max 2e3 files per char class) to the cache file: 
%1", ds.train.subset2e3.file_path)
  start <- put_start_date()
  save(train.files.subset2e3,
       train.labels2e3,
       train.images.subset2e3,
       my_emnist.train.subset2e3, 
       file = ds.train.subset2e3.file_path)
  put_log("Train Data subset (Max 2e3 files per char class) list has been cached to the File System.")
  put_end_date(start)
  
  rm(train.dat.subset2e3)
}

put_log1("Train image file list subset (Max 2e3 files per char class) structure:
%1", capture.output(str(train.files.subset2e3)))

put_log1("Train dataset labels:
%1", train.labels2e3, .sep = " ")

put_log1("`train.images.subset2e3` data structure:
%1", capture.output(str(train.images.subset2e3)))

put_log1("`my_emnist.train.subset2e3` dataset matrix dimensions: 
%1", dim(my_emnist.train.subset2e3), .sep = " ")

# Visualize the first char:
char.image(my_emnist.train.subset2e3[1,])

# rm(train.files.subset2e3)
# rm(train.images.subset2e3)
# rm(my_emnist.train.subset2e3)

#### Init `x2e3` & `y2e3` variables (Max 2e3 items per char class) -------------------
ch.labels <- train.labels2e3
x2e3 <- my_emnist.train.subset2e3
dim(x2e3)
class(x2e3)
str(x2e3)

y2e3 <- as.factor(rownames(x2e3))
str(y2e3)
length(y2e3)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset (Max 2e3 files per char class) -------------
open_logfile(".split-train-data-subset2e3")
### Split Train Dataset --------------------------------------------------------
dim.x2e3 <- dim(x2e3)
dim.x2e3
dim.x2e3[1]
dim.x2e3[2]

x2e3.sample.seed <- dim.x2e3[1]
x2e3.sample.seed
test.sample.seed <- as.integer(x2e3.sample.seed*0.2)
test.sample.seed

train.dataset.list <- sample_train_test_sets.mx(x2e3, 
                                                x2e3.sample.seed,
                                                shuffle.test_rows = TRUE,
                                                shuffle.seed = test.sample.seed)
str(train.dataset.list)

x2e3.train <- train.dataset.list$train_set
dim(x2e3.train)

y2e3.train <- as.factor(rownames(x2e3.train))
str(y2e3.train)
length(y2e3.train)

x2e3.test <- train.dataset.list$test_set
dim(x2e3.test)
#> [1] 166823    784

y2e3.test <- as.factor(rownames(x2e3.test))
str(y2e3.test)
length(y2e3.test)

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

dim_reduction.x1e3.train_knn_pca.file_path <-
  file.path(knn_pca.path, "dim-reduction.x2e3.train.k1-7.2nn+pca.RData")
 
# dim_reduction.x2e3.train_knn_pca.file_path <- 
#   file.path(knn_pca.path, "dim-reduction.x2e3.train.k4-5-6nn+pca.RData")

if (file.exists(dim_reduction.x2e3.train_knn_pca.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", dim_reduction.x2e3.train_knn_pca.file_path)
  
  start <- put_start_date()
  load(dim_reduction.x2e3.train_knn_pca.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  put_log("Training Model `kNN+PCA` on the dataset subset: `x2e3.train`..." )
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train_knn_pca <- caret::train(x2e3.train, y2e3.train, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = 5, p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = k_values))
  print_end_date(start)
  # Time difference of 10.72675 mins
  # Time difference of 8.459877 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x2e3.train`")
  
train_knn_pca.k1_7.2 <- train_knn_pca
# train_knn_pca.k4.5.6 <- train_knn_pca


  put_log("Saving Model in the cache file: `kNN+PCA`...")
  start <- put_start_date()
  save(train_knn_pca.k1_7.2, file = dim_reduction.x2e3.train_knn_pca.file_path)
  # save(train_knn_pca.k4.5.6, file = dim_reduction.x2e3.train_knn_pca.file_path)
  stopCluster(cl)
  stopImplicitCluster()
  print_end_date(start)
  rm(train_knn_pca)
  
  put_log1("The Model `kNN+PCA` trained on the dataset subset `x2e3.train` has been cached in file:
`%1`", dim_reduction.x2e3.train_knn_pca.file_path)
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


put_log("Predicting on `x2e3.test`")
start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)
y2e3_hat_knn_pca <- stats::predict(train_knn_pca.k1_7.2, x2e3.test, type = "raw")
# y2e3_hat_knn_pca <- stats::predict(train_knn_pca.k4.5.6, x2e3.test, type = "raw")
print_end_date(start)
# Time difference of 32.93572 secs

xy2e3.accuracy <- mean(y2e3_hat_knn_pca == y2e3.test)
xy2e3.accuracy
# train_knn_pca.k1_7.2
#> [1] 0.8226923
# train_knn_pca.k4.5.6
#> [1] 0.8238462

put_log2("Accuracy of `x2e3` model (trained and tested on subset of size %1 items):
%2", 2e3, xy2e3.accuracy)

put_log("Predicting on `x.test`")
start <- put_start_date()
y2e3_hat_knn_pca.x.test <- stats::predict(train_knn_pca.k1_7.2, x.test, type = "raw")
# y2e3_hat_knn_pca.x.test <- stats::predict(train_knn_pca.k4.5.6, x.test, type = "raw")
print_end_date(start)
# Time difference of 11.01372 mins

mean(y2e3_hat_knn_pca.x.test == y.test)
#> [1] 0.8410531

put_log2("Accuracy of `x2e3` model (trained on subset of size %1 items) tested on full-size test set:
%2", 2e3, xy2e3.accuracy)

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment (x2e3, y2e3) -------------------------------------------
# rm(x2e3)
# rm(x2e3.train)
# rm(x2e3.test)
# 
# rm(y2e3)
# rm(y2e3.train)
# rm(y2e3.test)

### Open log: Random Forest -------------------------------------
open_logfile(".model.random-forest")
### Random Forest --------------------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

#### Research ----------------------

research.randoms_forest.x2e3.fit_rf.nzv.mtry9.file_path <- file.path(models.random_forest.research.path, 
                                                                 "random-forest.x2e3.fit_rf.nzv.mtry9.RData")

if (file.exists(research.randoms_forest.x2e3.fit_rf.nzv.mtry9.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", research.randoms_forest.x2e3.fit_rf.nzv.mtry9.file_path)
  
  start <- put_start_date()
  load(research.randoms_forest.x2e3.fit_rf.nzv.mtry9.file_path)
  put_log("Train Data list has been loaded from cache.")
  
} else {
  
  start <- put_start_date()
  nzv <- nearZeroVar(x2e3.train)
  put_end_date(start)
  
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  system.time({fit_rf.nzv.mtry9 <- randomForest(x2e3.train[, -nzv], y2e3.train,  mtry = 9)})
 #    user  system elapsed 
 # 958.22    2.45  963.12 

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  #> Time difference of 16.08939 mins

  plot(fit_rf.nzv.mtry9)
  
  save(fit_rf.nzv.mtry9, 
       file = research.randoms_forest.x2e3.fit_rf.nzv.mtry9.file_path)
}

#### Random Forest (implementation) --------------------------------------------

randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path <- file.path(models.random_forest.path, 
                                                                 "x2e3.train_rf.nzv.mtry5_15.RData")

if (file.exists(randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path)
  
  start <- put_start_date()
  load(randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train.x2e3_rf.cv5.ntree200 <- caret::train(x2e3.train, y2e3.train, method = "rf", 
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
%1...", randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path)

  start <- put_start_date()
  save(train.x2e3_rf.cv5.ntree200, 
       file = randoms_forest.x2e3.train_rf.nzv.mtry5_15.file_path)
  put_log("The Train fit result has been cached on the local File System.")
  put_end_date(start)
}

plot(train.x2e3_rf.cv5.ntree200)
train.x2e3_rf.cv5.ntree200

put_log("Predicting values on the Test Set")
start <- put_start_date()
y_hat_rf <- stats::predict(train.x2e3_rf.cv5.ntree200, x2e3.test, type = "raw")
print_end_date(start)

mean(y_hat_rf == y2e3.test)
#> [1] 0.8473077

### Close Log ------------------------------------------------------------------
log_close()



## Clean Up Environment --------------------------------------------------------
rm(x2e3)
rm(x2e3.train)
rm(x2e3.test)

rm(y2e3)
rm(y2e3.train)
rm(y2e3.test)


rm(train_knn_pca)
rm(fit_rf.nzv.mtry9)
rm(train_rf)


# ---------------------------
# Reference:
#
# 

start <- put_start_date()
put_end_date(start)






























