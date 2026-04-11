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

### Open log: Split Train Dataset (x) -------------
open_logfile(".split.train_set-10%train_subset")
#### Split Train Dataset to extract a train sample of 10% size -----------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x <- dim(x)
dim.x
dim.x[1]
dim.x[2]

test_ratio <- 0.9 
x0.1.subset.sample.seed <- dim.x[1]
x0.1.subset.sample.seed
test.sample.seed <- as.integer(x0.1.subset.sample.seed*test_ratio)
test.sample.seed

cache_file.path <- file.path(ds.subsets.path, "train-data.10%subset.RData")
cache_file.path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Split Train Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
} else {
  train.dataset.list <- sample_train_test_sets.mx(x, 
                                                  x0.1.subset.sample.seed,
                                                  test.ratio = test_ratio,
                                                  shuffle.test_rows = TRUE,
                                                  shuffle.seed = test.sample.seed)
  str(train.dataset.list)
  
  x0.1.subset <- train.dataset.list$train_set
  y0.1.subset <- as.factor(rownames(x0.1.subset))
  
  x0.9.test.list <- splitDataset(train.dataset.list$test_set, 9)
  put_log1("Test dataset list structure:
%1", capture.output(str(x0.9.test.list)))

  put_log1("Caching data in the file
%1 ...", cache_file.path)
  
  save(x0.1.subset,
       y0.1.subset,
       x0.9.test.list,
       file = cache_file.path)
  
  put_log1("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  
  rm(train.dataset.list)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x0.1.subset)
#> [1] 

str(y0.1.subset)
length(y0.1.subset)

str(x0.9.test.list)


##### Clean up Environment -----------------------
rm(x)
rm(y)

### Close Log ------------------------------------------------------------------
log_close()

### Open log: Split Train Data Subset (x0.1.subset) -------------
open_logfile(".split.x0.1.subset")
#### Split Train Data Subset (x0.1.subset) -----------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x0.1.subset <- dim(x0.1.subset)
dim.x0.1.subset
dim.x0.1.subset[1]
dim.x0.1.subset[2]

test_ratio <- 0.1 
x0.1.subset.sample.seed <- dim.x0.1.subset[1]
x0.1.subset.sample.seed
test.sample.seed <- as.integer(x0.1.subset.sample.seed*test_ratio)
test.sample.seed

ds.x0.1.subset.file_path <- file.path(ds.subsets.path, "split-data.x0.1.subset.RData")
ds.x0.1.subset.file_path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(ds.x0.1.subset.file_path)) {
  put_log1("Loading Split Train Data from cache file: 
%1", ds.x0.1.subset.file_path)
  
  load(ds.x0.1.subset.file_path)
  put_log("Train Data list has been loaded from cache.")
} else {
  train.dataset.list <- sample_train_test_sets.mx(x0.1.subset, 
                                                  x0.1.subset.sample.seed,
                                                  test.ratio = test_ratio,
                                                  shuffle.test_rows = TRUE,
                                                  shuffle.seed = test.sample.seed)
  str(train.dataset.list)
  
  x0.1.train <- train.dataset.list$train_set
  y0.1.train <- as.factor(rownames(x0.1.train))
  
  x0.1.test <- train.dataset.list$test_set
  y0.1.test <- as.factor(rownames(x0.1.test))
  
  start <- put_start_date()
  put_log1("Caching data in the file
%1 ...", ds.x0.1.subset.file_path)
  
  save(x0.1.train,
       y0.1.train,
       x0.1.test,
       y0.1.test,
       file = ds.x0.1.subset.file_path)
  
  put_log1("The Train Data Subset objects have been cached in file:
`%1`", ds.x0.1.subset.file_path)
  put_end_date(start)
  
  rm(train.dataset.list)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x0.1.train)
#> [1] 

str(y0.1.train)
length(y0.1.train)

dim(x0.1.test)
#> [1] 166823    784

str(y0.1.test)
length(y0.1.test)

##### Clean up Environment -----------------------
rm(x0.1.subset)
rm(y0.1.subset)

### Close Log ------------------------------------------------------------------
log_close()

## Model Building --------------------------------------------------------------
### Training Model using the following methods: kNN, PCA -----------------------
# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path)) {
  dir.create(knn_pca.path)
}

#### Open log: Pre-train kNN+PCA Model ---------------------------------
open_logfile(".pre-train-model.knn+pca")
#### Pre-tuning k for kNN -------------------------------
k.values <- seq(1, 7, 2)

cache_file.path <-
  file.path(knn_pca.path, "dim-reduction.x0.1.train.k1-7.2nn+pca.RData")
 
start <- put_start_date()
# Thu Apr 9 09:14:47 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1...", cache_file.path)
  
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Model Fit Data have been loaded from cache.")
} else {
  put_log("Training Model `kNN+PCA` on the dataset subset: `x0.1.train`..." )
  
  train_knn_pca.k1_7.2 <- caret::train(x0.1.train, y0.1.train, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = 5, p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = k.values))
  put_end_date(start)
  # Time difference of 40.88067 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x0.1.train`")

  put_log("Saving Model in the cache file: `kNN+PCA`...")
  start <- put_start_date()
  save(train_knn_pca.k1_7.2, file = cache_file.path)
  put_end_date(start)
  # Time difference of mins

  put_log1("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", cache_file.path)

}

stopCluster(cl)
stopImplicitCluster()

plot(train_knn_pca.k1_7.2)
put_log1("kNN+PCA Model trained result:
%1",capture.output(train_knn_pca.k1_7.2))

# 
# k-Nearest Neighbors 

# 75032 samples
# 784 predictor
# 39 classes: '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
# 
# Pre-processing: principal component signal extraction (743), centered (743), scaled (743), remove (41) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 60022, 60023, 60030, 60025, 60028 
# Resampling results across tuning parameters:

# k  Accuracy   Kappa    
# 1  0.8520766  0.8461487
# 3  0.8610591  0.8554102
# 5  0.8632982  0.8576989
# 7  0.8625919  0.8569392

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 5.

### Close Log ------------------------------------------------------------------
log_close()
##### Open log: Predictions for kNN+PCA Pre-trained Model on `x0.1.test` dataset ----
open_logfile(".predict.knn+pca.pre-trained.x0.1.test")
##### Construct Predictions for kNN+PCA Pre-trained Model ------------
cache_file.path <-
  file.path(knn_pca.path, "x0.1.test.k1-7.2nn+pca.predictions.RData")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Predicted Data from cache file: 
%1...", cache_file.path)
  
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Predicted Data have been loaded from cache.")
} else {
  put_log("Predicting on `x0.1.test`")
  
  y0.1_hat_knn_pca.k1_7.2 <- stats::predict(train_knn_pca.k1_7.2, x0.1.test, type = "raw")
  put_end_date(start)
  # Time difference of 2.74791 mins
  
  put_log3("Predicted data Summary of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k 1-7 by step 2,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, capture.output(summary(y0.1_hat_knn_pca.k1_7.2)))

  start <- put_start_date()
  xy0.1.knn_pca.k1_7.2.accuracy <- mean(y0.1_hat_knn_pca.k1_7.2 == y0.1.test)
  put_end_date(start)
  # Time difference of ??? mins
  
  xy0.1.knn_pca.k1_7.2.accuracy
  #> [1] 0.8693882
  
  put_log3("Accuracy of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k 1-7 by step 2,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, xy0.1.knn_pca.k1_7.2.accuracy)
  #> 0.868550221477314
  
  save(y0.1_hat_knn_pca.k1_7.2,
       xy0.1.knn_pca.k1_7.2.accuracy,
       file = cache_file.path)
}
  
stopCluster(cl)
stopImplicitCluster()
#> [1] 

##### Close Log ------------------------------------------------------------------
log_close()

##### Open log: Predictions for kNN+PCA Pre-trained Model on `x0.9.test.list` ----
open_logfile(".predict.knn+pca.pre-trained.x0.9.test.list")
##### Predict for kNN+PCA Pre-trained Model on `x0.9.test.list` ----------------
cache_file.path <-
  file.path(knn_pca.path, "x0.9.test.list.k1-7.2nn+pca.predictions.RData")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Predicted Data from cache file: 
%1...", cache_file.path)
  
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Predicted Data have been loaded from cache.")
} else {

  y_hat_knn_pca.k1_7.2.x0.9.test <- lapply(seq_len(length(x0.9.test.list$x)), 
                                                   function(i){
    x.test <- x0.9.test.list$x[[i]]
    y_hat <- x0.9.test.list$y[[i]]
    put_log("Predicting on `x0.9.test.list`")
    start <- put_start_date()
    y_hat <- stats::predict(train_knn_pca.k1_7.2, x.test, type = "raw")
    put_end_date(start)

    plot(y_hat)
    
    put_log("Summary of predicted data for the `x0.1 kNN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 1 to 7 with a step of 2,
and tested on the %1 10% subset of the remaining 90% of the `Train Set`:
%2", n.to_ordinal(i), capture.output(summary(y_hat)))
    
    y_hat
  }) |> unlist()
  
  put_log("Summary of predicted data for the `x0.1 kNN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 1 to 7 with a step of 2,
and tested on the remaining 90% of the `Train Set`:
%1", capture.output(summary(y_hat_knn_pca.k1_7.2.x0.9.test)))
  
  plot(y_hat_knn_pca.k1_7.2.x0.9.test)

  put_log("Cashing prediction results in the file system...")
  save(y_hat_knn_pca.k1_7.2.x0.9.test,
       file = cache_file.path)
  
  put_log("The prediction results have been saved to the file::
%1", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()

##### Accuracy of the kNN+PCA Pre-trained Model Predictions on `x0.9.test.list` ----
put_log("Validating predictions for the pre-tuned `x0.1 kNN+PCA` model...")

xy0.9.knn_pca.k1_7.2.accuracy <- 
  mean(y_hat_knn_pca.k1_7.2.x0.9.test == unlist(x0.9.test.list$y))
xy0.9.knn_pca.k1_7.2.accuracy
#> 0.8677351

put_log("Accuracy of the predicted data for the `x0.1 kNN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 1 to 7 with a step of 2,
and tested on the remaining 90% of the `Train Set`:
%1", xy0.9.knn_pca.k1_7.2.accuracy)
#> [1] 0.867735081163533

##### Close Log ------------------------------------------------------------------
log_close()


## Clean Up Environment (x4e3, y4e3) -------------------------------------------
# rm(x4e3)
# rm(x4e3.train)
# rm(x4e3.test)
# 
# rm(y4e3)
# rm(y4e3.train)
# rm(y4e3.test)

#### Open log: Fine-tune kNN+PCA Model ---------------------------------
open_logfile(".fine-tune-model.knn+pca")
#### Fine-tuning k for kNN -------------------------------
k_values <- c(4, 5, 6)

cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.fine-tune.k4-6NN+PCa.RData")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log("Loading `kNN+PCA` model Fit Data from the cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("The `kNN+PCA` model Fit Data has been loaded from the cache file.")

} else {
  put_log("Fine-tuning `kNN+PCA` model on the dataset subset: `x0.1.train`..." )
  
  train_knn_pca.k4_6 <- caret::train(x0.1.train, y0.1.train, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = 5, p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = k.values))
  put_end_date(start)
  # Time difference of 8.459877 mins
  put_log("The Model `kNN+PCA` has been fine-tuned on the dataset subset: `x0.1.train`")

  put_log("Saving fine-tuned `kNN+PCA` Model in the cache file...")
  start <- put_start_date()
  save(train_knn_pca.k4_6, file = cache_file.path)
  put_end_date(start)
  
  put_log("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", cache_file.path)
  # [1] "Thu Apr  9 09:55:40 2026"
  # Time difference of 40.88067 mins
}

stopCluster(cl)
stopImplicitCluster()
plot(train_knn_pca.k4_6)
put_log1("kNN+PCA Model trained result:
%1",capture.output(train_knn_pca.k4_6))

# k  Accuracy   Kappa    
# 4  0.8061538  0.8010526
# 5  0.8093590  0.8043421
# 6  0.8092628  0.8042434

### Close Log ------------------------------------------------------------------
log_close()
##### Open log: Construct Predictions for kNN+PCA Fine-tuned Model ------------
open_logfile(".predict.knn+pca.pre-trained-model")
##### Construct Predictions for kNN+PCA Fine-tuned Model ------------
cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.k4_6.2nn+pca.predictions.RData")

start <- put_start_date()
# Fri Apr 10 04:24:08 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Predicted Data from cache file: 
%1...", cache_file.path)
  
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Predicted Data have been loaded from cache.")
} else {
  put_log("Predicting on `x0.1.test`")
  
  y0.1_hat_knn_pca.k4_6 <- stats::predict(train_knn_pca.k4_6, x0.1.test, type = "raw")
  put_end_date(start)
  # Time difference of 2.730859 mins
  
  put_log3("Predicted data Summary of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k: 4,5,6,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, capture.output(summary(y0.1_hat_knn_pca.k4_6)))
  
  xy0.1.knn_pca.k4_6.accuracy <- mean(y0.1_hat_knn_pca.k4_6 == y0.1.test)
  xy0.1.knn_pca.k4_6.accuracy
  #> [1] 0.8455645
  
  put_log3("Accuracy of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
fine-tuned for sequence of k: 4,5,6,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, xy0.1.knn_pca.k4_6.accuracy)
  #> 0.84556446785586
  
  put_log("Predicting on `x0.9.test`")
  start <- put_start_date()
  # Fri Apr 10 05:02:33 2026
  
  y_hat_knn_pca.k4_6.x0.9.test <- stats::predict(train_knn_pca.k4_6, x0.9.test, type = "raw")
  put_end_date(start)
  # Time difference of ??? mins
  
  put_log3("Predicted data Summary of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k: 4,5,6,
and tested on %2 size proportion test set:
%3", 0.1, 0.9, capture.output(summary(y_hat_knn_pca.k4_6.x0.9.test)))
  
  start <- put_start_date()
  xy0.9.knn_pca.k4_6.accuracy <- mean(y_hat_knn_pca.k4_6.x0.9.test == y0.9.test)
  put_end_date(start)
  # Time difference of 11.01372 mins
  
  xy0.9.knn_pca.k4_6.accuracy
  #> 0.8676296
  
  put_log3("Accuracy of `x0.1` kNN+PCA model trained on subset of %1 size proportion,
tuned for sequence of k: 4,5,6,
and tested on %2 size proportion test set:
%3", 0.1, 0.9, xy0.9.knn_pca.k4_6.accuracy)
  #> [1] 0.867629564204937
  
  save(y0.1_hat_knn_pca.k4_6,
       xy0.1.knn_pca.k4_6.accuracy,
       y_hat_knn_pca.k4_6.x0.9.test,
       xy0.9.knn_pca.k4_6.accuracy,
       file = cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()
#> [1] 

##### Close Log ------------------------------------------------------------------
log_close()

## Clean Up Environment (x4e3, y4e3) -------------------------------------------
# rm(y0.1_hat_knn_pca.k4-6)
# rm()
# rm()
# 
# rm()
# rm()
# rm()

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
  put_end_date(start)
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
put_end_date(start)

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






























