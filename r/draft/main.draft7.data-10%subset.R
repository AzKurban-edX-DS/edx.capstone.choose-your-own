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

#### Open log: Pre-training kNN+PCA Model Log ----------------------------------
open_logfile(".pre-train-model.k1-7.2nn+pca")
#### Tuning k1_7.2NN+PCA model by *k* parameter ranging from 1 to 7 by step 2 -------
k.values <- seq(1, 7, 2)

cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.k1-7.2nn+pca.RData")
 
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
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9)),
                                tuneGrid = data.frame(k = k.values))
  put_end_date(start)
  # Time difference of 40.88067 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x0.1.train`")

  put_log("Saving Model in the cache file: `kNN+PCA`...")
  start <- put_start_date()
  save(train_knn_pca.k1_7.2, file = cache_file.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log1("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", cache_file.path)

}

stopCluster(cl)
stopImplicitCluster()

put_log("kNN+PCA Model trained result:
%1", train_knn_pca.k1_7.2,
  capture_output = 1)

acc.max.idx <- which.max(train_knn_pca.k1_7.2$results$Accuracy)
acc.max.idx

k.1to7step2.max_accuracy <- train_knn_pca.k1_7.2$results$Accuracy[acc.max.idx]
k.1to7step2.max_accuracy

k.1to7step2.best <- train_knn_pca.k1_7.2$results$k[acc.max.idx]
k.1to7step2.best

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
#### Open log: Fine-tuning kNN+PCA Model Log ---------------------------------
open_logfile(".fine-tune-model.k4-6nn+pca")
#### Fine-tuning kNN+PCA Model by *k* parameter (4 to 6) of kNN ----------------
k.values <- c(4, 5, 6)

cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.fine-tune.k4-6NN+PCA.RData")

start <- put_start_date()
# w.pc_cores <- as.integer(N_pcCores / 2 + 1) 
# w.pc_cores  

cl <- makeCluster(N_pcCores, type='PSOCK', outfile="")
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
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9),
                                                         verboseIter = TRUE),
                                tuneGrid = data.frame(k = k.values))
  put_end_date(start)
  # Time difference of 39.72623 mins
  put_log("The Model `kNN+PCA` has been fine-tuned on the dataset subset: `x0.1.train`")

  put_log("Saving fine-tuned `kNN+PCA` Model in the cache file...")
  start <- put_start_date()
  save(train_knn_pca.k4_6, file = cache_file.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", cache_file.path)
  # [1] "Thu Apr  9 09:55:40 2026"
  # Time difference of 40.88067 mins
}

stopCluster(cl)
stopImplicitCluster()
plot(train_knn_pca.k4_6)
put_log("kNN+PCA Model trained result:
%1",train_knn_pca.k4_6,
         capture_output = 1)

str(train_knn_pca.k4_6)

acc.max.idx <- which.max(train_knn_pca.k4_6$results$Accuracy)
k.4to6.max_accuracy <- train_knn_pca.k4_6$results$Accuracy[acc.max.idx]
k.4to6.max_accuracy

k.4to6.best <- train_knn_pca.k4_6$results$k[acc.max.idx]
k.4to6.best

k.4to6.best == k.1to7step2.best
#> TRUE 

# kNN+PCA Model trained result:
#   k-Nearest Neighbors 
# 
# 75032 samples
# 784 predictor
# 39 classes: '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
# 
# Pre-processing: principal component signal extraction (743), centered (743), scaled (743), remove (41) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 60027, 60026, 60028, 60021, 60026 
# Resampling results across tuning parameters:
#   
#   k  Accuracy   Kappa    
# 1  0.8509833  0.8450118
# 3  0.8602059  0.8545233
# 5  0.8635644  0.8579822
# 7  0.8624317  0.8567740
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 5.

### Close Log ------------------------------------------------------------------
log_close()
##### Open log: Predictions on `k5NN+PCA` (fine-tuned) Model for `x0.1.test` dataset ----
open_logfile(".x0.1.test.predict.k5nn+pca")
##### Constructing Predictions on k5NN+PCA (fine-tuned) Model for `x0.1.test` dataset ----
cache_file.path <-
  file.path(knn_pca.path, "x0.1.test.k5NN+PCA.predictions.RData")

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
  put_log("Constructing predictions for the `x0.1.test` dataset 
using the fine-tuned K5NN+PCA model...")
  
  y0.1_hat_knn_pca.k4_6 <- stats::predict(train_knn_pca.k4_6, x0.1.test, type = "raw")
  put_end_date(start)
  # Time difference of 2.74791 mins
  put_log("The `k5NN+PCA` Model: Generating predictions have been completed `x0.1.test` dataset.")
  
  put_log("Validating accuracy of the k5NN+PCA (fine-tuned) Model predictions 
made for the `x0.1.test` dataset...")
  
  xy0.1.knn_pca.k4_6.accuracy <- mean(y0.1_hat_knn_pca.k4_6 == y0.1.test)

  put_log("The accuracy value is %1", xy0.1.knn_pca.k4_6.accuracy)
  #> [1] 0.8693882
  
  save(y0.1_hat_knn_pca.k4_6,
       xy0.1.knn_pca.k4_6.accuracy,
       file = cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()
#> [1] 

put_log("Summary of predicted data made using the fine-tuned `k5NN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 4 to 6,
and tested on the 10% sample from the remaining 90% data of the `Train Set`:
%1", summary(y0.1_hat_knn_pca.k4_6))


put_log("Accuracy of the predicted data for the `k5NN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 4 to 6,
and tested on the 10% sample from the remaining data of the `Train Set`:
%1", xy0.1.knn_pca.k4_6.accuracy)
#> [1] 0.868550221477314


##### Close Log ------------------------------------------------------------------
log_close()

##### Open log: Predictions on `k5NN+PCA` (fine-tuned) Model for `x0.9.test` dataset ----
open_logfile(".x0.9.test.list.predict.k5nn+pca")
##### Constructing Predictions on k5NN+PCA (fine-tuned) Model for `x0.9.test` dataset ----
cache_file.path <-
  file.path(knn_pca.path, "x0.9.test.list.k5NN+PCA.predictions.RData")

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

  y_hat.k4_6nn.pca.x0.9.test <- lapply(seq_len(length(x0.9.test.list$x)), 
                                                   function(i){
    x.test <- x0.9.test.list$x[[i]]
    y_hat <- x0.9.test.list$y[[i]]
    put_log("Predicting on `x0.9.test.list`")
    start <- put_start_date()
    y_hat <- stats::predict(train_knn_pca.k4_6, x.test, type = "raw")
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
%1", capture.output(summary(y_hat.k4_6nn.pca.x0.9.test)))
  
  plot(y_hat.k4_6nn.pca.x0.9.test)

  put_log("Cashing prediction results in the file system...")
  save(y_hat.k4_6nn.pca.x0.9.test,
       file = cache_file.path)
  
  put_log("The prediction results have been saved to the file::
%1", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()

##### Accuracy of the k5NN+PCA (fine-tuned) Model Predictions on `x0.9.test` dataset ----
put_log("Validating predictions for the pre-tuned `x0.1 kNN+PCA` model...")

xy0.9.k4_6nn.pca.accuracy <- 
  mean(y_hat.k4_6nn.pca.x0.9.test == unlist(x0.9.test.list$y))
xy0.9.k4_6nn.pca.accuracy
#> 0.8677351

put_log("Accuracy of the predicted data for the `x0.1 kNN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 1 to 7 with a step of 2,
and tested on the remaining 90% of the `Train Set`:
%1", xy0.9.k4_6nn.pca.accuracy)
#> [1] 0.867735081163533

##### Close Log ------------------------------------------------------------------
log_close()


## Clean Up Environment (x4e3, y4e3) -------------------------------------------
# rm(x4e3)
# rm(x0.1.train)
# rm(x4e3.test)
# 
# rm(y4e3)
# rm(y4e3.train)
# rm(y4e3.test)

### Random Forest --------------------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

### Open log: Random Forest: Research ------------------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry9")
#### Research and estimate performance of the `Random Forest` method -----------

x0.1.train.fit_rf.nzv.mtry9.file_path <- 
  file.path(models.random_forest.research.path, 
            "x0.1.train.fit_rf.nzv.mtry9.RData")

if (file.exists(x0.1.train.fit_rf.nzv.mtry9.file_path)) {
  put_log("Loading Model Fit Data from cache file: 
%1", x0.1.train.fit_rf.nzv.mtry9.file_path)
  
  start <- put_start_date()
  load(x0.1.train.fit_rf.nzv.mtry9.file_path)
  put_log("Train Data list has been loaded from cache.")
  
} else {
  
  start <- put_start_date()
  nzv <- nearZeroVar(x0.1.train)
  put_end_date(start)
  # Time difference of 56.0386 secs
  
  x0.1.train_nzv <- x0.1.train[, -nzv]
  
  put_log("Pre-training `Random Forest` model on the following 10% sample 
from the `Train set`: `x0.1.train`..." )
  
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  

  fit_rf.nzv.mtry9 <- randomForest(x0.1.train_nzv, y0.1.train,  mtry = 9)

  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("The `Random Forest` model has been pre-trained on the dataset: `x0.1.train`." )
  put_end_date(start)
  #> Time difference of 44.35714 mins

  plot(fit_rf.nzv.mtry9)
  
  save(fit_rf.nzv.mtry9, 
       file = x0.1.train.fit_rf.nzv.mtry9.file_path)
}


put_log("Summary of the fit data produced by pre-training the `Random Forest` model:
%1", summary(fit_rf.nzv.mtry9),
        capture_output = 1)

##### Close Log ------------------------------------------------------------------
log_close()
##### Open log: Predictions on `RF` Model for `x0.1.test` dataset ----
open_logfile(".x0.1.test.predict.rf.mtry9")
##### Constructing Predictions on `RF.mtry9` Model for `x0.1.test` dataset ----
cache_file.path <-
  file.path(models.random_forest.research.path, "x0.1.test.rf.mtry9.predictions.RData")

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
  put_log("Predicting `RF.mtry9` model on `x0.1.test`...")
  
  y0.1_hat_rf.mtry9 <- stats::predict(fit_rf.nzv.mtry9, x0.1.test, type = "response")
  put_end_date(start)
  # Time difference of 2.74791 mins
  
  put_log("The `RF.mtry9` Model: Generating predictions have been completed `x0.1.test` dataset.")
  
  
  put_log("Validating accuracy of the k5NN+PCA (fine-tuned) Model predictions 
made for the `x0.1.test` dataset...")

  xy0.rf.mtry9.accuracy <- mean(y0.1_hat_rf.mtry9 == y0.1.test)
  # Time difference of ??? mins
  put_log("The accuracy value is %1", xy0.rf.mtry9.accuracy)
#> [1] 0.876092421884353
  
  save(y0.1_hat_rf.mtry9,
       xy0.rf.mtry9.accuracy,
       file = cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()

put_log("Summary of predicted data made using the `RF.mtry9` model,
trained on a 10% sample of the`Train Set` dataset for `mtry = 9`,
and tested on the 10% sample from the remaining 90% data of the `Train Set`:
%1", summary(y0.1_hat_rf.mtry9))


put_log("Accuracy of the predicted data for the `k5NN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 4 to 6,
and tested on the 10% sample from the remaining data of the `Train Set`:
%1", xy0.rf.mtry9.accuracy)
#> 0.876092421884353

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Random Fores: Training -------------------------------------
open_logfile(".x0.1.train.fit_rf.nzv.mtry5_15")
#### Random Forest: Training --------------------------------------------

x0.1.train.rf.nzv.mtry5_15.file_path <- file.path(models.random_forest.path, 
                                                                 "x0.1.train_rf.nzv.mtry5_15.RData")
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(x0.1.train.rf.nzv.mtry5_15.file_path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1", x0.1.train.rf.nzv.mtry5_15.file_path)
  
  start <- put_start_date()
  load(x0.1.train.rf.nzv.mtry5_15.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  start <- put_start_date()
  
  put_log("Training `Random Forest` model on the following 10% sample 
from the `Train Set`: `x0.1.train`..." )
  x0.1.train.rf.cv5.ntree200.fit <- caret::train(x0.1.train, y0.1.train, method = "rf", 
                           preProcess = "nzv",
                           trControl = trainControl(method = "cv", number = 5, p = .8),
                           ntree = 200,
                           tuneGrid = data.frame(mtry = seq(5, 15, 5)))
  
  put_log("The `Random Forest` model has been trained on the dataset: `x0.1.train`." )
  
  put_end_date(start)
# [1] "Sun Apr  5 08:29:02 2026"
# Time difference of 57.23111 mins

put_log("Saving Train fit result...", )

  start <- put_start_date()
  save(x0.1.train.rf.cv5.ntree200.fit, 
       file = x0.1.train.rf.nzv.mtry5_15.file_path)
  put_log("The Train fit result has been saved to the cache file:
%1.", x0.1.train.rf.nzv.mtry5_15.file_path)
  put_end_date(start)
}

stopCluster(cl)
stopImplicitCluster()

plot(x0.1.train.rf.cv5.ntree200.fit)

put_log("Summary of the fit data produced by training the `Random Forest` model:
%1", summary(x0.1.train.rf.cv5.ntree200.fit),
        capture_output = 1)

# put_log("Predicting values on the Test Set")
# start <- put_start_date()
# y_hat_rf <- stats::predict(x0.1.train.rf.cv5.ntree200.fit, x4e3.test, type = "raw")
# put_end_date(start)
# 
# mean(y_hat_rf == y4e3.test)
# #> [1] 0.8473077

### Close Log ------------------------------------------------------------------
log_close()
##### Open log: Predictions on `RF` Model for `x0.1.test` dataset ----
open_logfile(".x0.1.test.predict.rf")
##### Constructing Predictions on `RF` Model for `x0.1.test` dataset ----
cache_file.path <-
  file.path(knn_pca.path, "x0.1.test.k4-6nn+pca.predictions.RData")

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
  put_log("Predicting (fine-tuned) K5NN+PCA model on `x0.1.test`...")
  
  y0.1_hat_knn_pca.k4_6 <- stats::predict(train_knn_pca.k4_6, x0.1.test, type = "raw")
  put_end_date(start)
  # Time difference of 2.74791 mins
  
  put_log3("Predicted data Summary of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k 1-7 by step 2,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, capture.output(summary(y0.1_hat_knn_pca.k4_6)))
  
  start <- put_start_date()
  xy0.1.knn_pca.k4_6.accuracy <- mean(y0.1_hat_knn_pca.k4_6 == y0.1.test)
  put_end_date(start)
  # Time difference of ??? mins
  
  xy0.1.knn_pca.k4_6.accuracy
  #> [1] 0.8693882
  
  put_log3("Accuracy of `x0.1 kNN+PCA` model trained on subset of %1 size proportion,
tuned for sequence of k 1-7 by step 2,
and tested on %2 size proportion test set:
%3", 0.1, 0.1, xy0.1.knn_pca.k4_6.accuracy)
  #> 0.868550221477314
  
  save(y0.1_hat_knn_pca.k4_6,
       xy0.1.knn_pca.k4_6.accuracy,
       file = cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()
#> [1] 

##### Close Log ------------------------------------------------------------------
log_close()

##### Open log: Predictions on `RF` Model for `x0.9.test` dataset ----
open_logfile(".predict.rfx0.9.test.list")
##### Constructing Predictions for k5NN+PCA (fine-tuned) Model for `x0.9.test` dataset ----
cache_file.path <-
  file.path(knn_pca.path, "x0.9.test.list.k4-6nn+pca.predictions.RData")

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
  
  y_hat.k4_6nn.pca.x0.9.test <- lapply(seq_len(length(x0.9.test.list$x)), 
                                       function(i){
                                         x.test <- x0.9.test.list$x[[i]]
                                         y_hat <- x0.9.test.list$y[[i]]
                                         put_log("Predicting on `x0.9.test.list`")
                                         start <- put_start_date()
                                         y_hat <- stats::predict(train_knn_pca.k4_6, x.test, type = "raw")
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
%1", capture.output(summary(y_hat.k4_6nn.pca.x0.9.test)))
  
  plot(y_hat.k4_6nn.pca.x0.9.test)
  
  put_log("Cashing prediction results in the file system...")
  save(y_hat.k4_6nn.pca.x0.9.test,
       file = cache_file.path)
  
  put_log("The prediction results have been saved to the file::
%1", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()

##### Accuracy of the kNN+PCA Pre-trained Model Predictions on `x0.9.test.list` ----
put_log("Validating predictions for the pre-tuned `x0.1 kNN+PCA` model...")

xy0.9.k4_6nn.pca.accuracy <- 
  mean(y_hat.k4_6nn.pca.x0.9.test == unlist(x0.9.test.list$y))
xy0.9.k4_6nn.pca.accuracy
#> 0.8677351

put_log("Accuracy of the predicted data for the `x0.1 kNN+PCA` model,
trained on a 10% sample of the`Train Set` dataset,
optimized for a sequence of *k* values ranging from 1 to 7 with a step of 2,
and tested on the remaining 90% of the `Train Set`:
%1", xy0.9.k4_6nn.pca.accuracy)
#> [1] 0.867735081163533

##### Close Log ------------------------------------------------------------------
log_close()
## Clean Up Environment --------------------------------------------------------
rm(x4e3)
rm(x0.1.train)
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






























