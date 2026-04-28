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

## Load Common Helper Functions --------------------------------------------------
common_helper.funcs.file_path <- file.path(support_functions.path, "common-helper.R")


source(common_helper.funcs.file_path, 
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


## Load Model Helper Functions --------------------------------------------------
model_helper.funcs.file_path <- file.path(support_functions.path, "models-helper.R")


source(model_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


### Open log: Load Train Data --------------------------------------------------
open_logfile(".load-train-data")
### Load Train Data ------------------------------------------------------------
train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.RData")
train.img28x28bin.list.file_path

my_mnist.file_path <- file.path(train.data.path, "my_emnist.RData")
my_mnist.file_path

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

if(!file.exists(my_mnist.file_path)){
  put_log1("LoadingBinary Image 28x28 list from cache file: 
%1", train.img28x28bin.list.file_path)

  put_log("Building flatten (`EMNIST`-like) dataset...")
  my_emnist <- img28x28.list2flatten.mx(img28xc28bin.list$img.list)
  put_log("The flatten dataset have been created.")
  str(my_emnist)

  put_log1("Saving flatten training dataset to the cache file: 
%1", my_mnist.file_path)
  start <- put_start_date()
  save(my_emnist,
       file = my_mnist.file_path)
  put_log("The flatten training dataset has been saved to the cache file:
%1", my_mnist.file_path)
  put_end_date(start)
} else {
  start <- put_start_date()
  put_log("Loading flatten training dataset from cache.")
  load(my_mnist.file_path)
  put_log("The flatten training dataset has been loaded from cache.")
  put_end_date(start)
}

put_log("Binary Image 28x28 list structure:
%1", capture.output(str(img28x28bin.list)))

y.labels <- img28x28bin.list$label.list

put_log("Train dataset labels:
%1", y.labels, .sep = " ")


#### Init `x` & `y` variables -------------------
# Short name for current working dataset
x <- my_emnist
str(x)
dim(x)
class(x)

# class identifies
y.groups <- ds.get_classIDs.grouped(x)
y <- y.groups$classID
str(y)
length(y)
# 834032

y.int <- as.integer(y)
y.chars <- y.groups$groupByClass
str(y.chars)

max(y.chars$n)
# 65504
y.chars$classID[which.max(y.chars$n)]
#> [1] 0
#> Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

min(y.chars$n)
# 4261
y.chars$classID[which.min(y.chars$n)]
#> [1] J
#> Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

print(y.chars, n = length(y.chars$classID))
#----
# A tibble: 39 × 2
# classID     n
# <fct> <int>
# 1 #     15600
# 2 $     16199
# 3 &     13000
# 4 @     38009
# 5 0     65504 # max(n)
# 6 1     43773
# 7 2     39351
# 8 3     39996
# 9 4     38112
# 10 5     32317
# 11 6     38879
# 12 7     41080
# 13 8     38795
# 14 9     38319
# 15 A     17205
# 16 B      8666
# 17 C     13560
# 18 D     15509
# 19 E     32627
# 20 F     11635
# 21 G      5443
# 22 H     12133
# 23 I     13873
# 24 J      4261 # min(n)
# 25 K      4334
# 26 L     21648
# 27 M     12089
# 28 N     21421
# 29 P     11095
# 30 Q      4707
# 31 R     20498
# 32 S     25910
# 33 T     30853
# 34 U     16385
# 35 V      7246
# 36 W      7266
# 37 X      5106
# 38 Y      6762
# 39 Z      4866

##### Clean up Environment -----------------------
# rm(my_emnist.train)

# Remove intermediate data for free the computer memory 
rm(my_emnist)
rm(img28x28bin.list)

  
##### Clean up Environment -----------------------
rm(img28x28bin.list)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Split Train Dataset (x) -------------
open_logfile(".split.10%train.balanced_subset")
#### Split Train Dataset  (10% for Train set) ----------------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x <- dim(x)
dim.x
dim.x[1]
dim.x[2]

test_ratio <- 0.9 
sample_seed <- dim.x[1]
sample_seed
shuffle_seed <- as.integer(sample_seed*test_ratio)
shuffle_seed

cache_file.path <- file.path(ds.subsets.path, "x.1bl.train(balanced).RData")
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
  split.list <- sample_train_test_sets.mx(x, 
                                          sample_seed,
                                          test.ratio = test_ratio,
                                          shuffle.seed = shuffle_seed)
  str(split.list)
  
  x.1bl.train <- split.list$train_set
  y.1bl.train <- as.factor(rownames(x.1bl.train))
  
  x.9.test <- split.list$test_set
  y.9.test <- as.factor(rownames(x.9.test))
  
  
#   x0.9.test.list <- splitDataset(split.list$test_set, 9)
#   put_log1("Test dataset list structure:
# %1", capture.output(str(x0.9.test.list)))

  put_log1("Caching data in the file
%1 ...", cache_file.path)
  
  save(x.1bl.train,
       y.1bl.train,
       x.9.test,
       y.9.test,
       file = cache_file.path)
  
  put_log1("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  
  rm(split.list)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x.1bl.train)
#> [1] 16653   784
str(x.1bl.train)

str(y.1bl.train)
length(y.1bl.train)

str(x.9.test)
str(y.9.test)
length(y.9.test)
#> [1] 817379


### Close Log ------------------------------------------------------------------
log_close()

## Model Building --------------------------------------------------------------
### Training Model using methods: kNN, PCA -----------------------
# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path)) {
  dir.create(knn_pca.path)
}

#### Open log: Pre-training kNN+PCA Model Log ----------------------------------
open_logfile(".pre-train-model.k4nn+pca")
#### Training k4NN+PCA model by *k* = 4 parameter -------

cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.k4nn+pca.RData")
 
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1...", cache_file.path)
  
  start <- put_start_date()
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Model Fit Data have been loaded from cache.")
} else {
  put_log("Training Model `kNN+PCA` on the dataset subset: `x0.1.train`..." )
  
  start <- put_start_date()
  train_knn_pca.k4 <- caret::train(x.1bl.train, y.1bl.train, method = "knn", 
                                preProcess = "pca",
                                # trControl = trainControl(),
                                tuneGrid = data.frame(k = 4))
  put_end_date(start)
  # Time difference of 40.88067 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x0.1.train`")

  put_log("Saving Model in the cache file: `kNN+PCA`...")
  start <- put_start_date()
  save(train_knn_pca.k4, file = cache_file.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log1("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", cache_file.path)

}

stopCluster(cl)
stopImplicitCluster()

put_log("kNN+PCA Model trained result:
%1", capture.output(str(train_knn_pca.k4)))

acc.max.idx <- which.max(train_knn_pca.k4$results$Accuracy)
acc.max.idx

k4nn.max_accuracy <- train_knn_pca.k4$results$Accuracy[acc.max.idx]
k4nn.max_accuracy

k4nn.best <- train_knn_pca.k4$results$k[acc.max.idx]
k4nn.best

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
#### Open log: Pre-training kNN+PCA Model Log ----------------------------------
open_logfile(".pre-train-model.k1-7.2nn+pca")
#### Tuning k1_7.2NN+PCA model by *k* parameter ranging from 1 to 7 by step 2 -------
k.values <- seq(1, 7, 2)

cache_file.path <-
  file.path(knn_pca.path, "x0.1.train.k1-7.2nn+pca.RData")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Model Fit Data from cache file: 
%1...", cache_file.path)
  
  start <- put_start_date()
  load(cache_file.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Model Fit Data have been loaded from cache.")
} else {
  put_log("Training Model `kNN+PCA` on the dataset subset: `x0.1.train`..." )
  
  start <- put_start_date()
  train_knn_pca.k1_7.2 <- caret::train(x.1bl.train, y.1bl.train, method = "knn", 
                                preProcess = "pca",
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
%1", capture.output(train_knn_pca.k1_7.2))
str(train_knn_pca.k1_7.2)

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
%1",capture.output(train_knn_pca.k4_6))

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

#### Research and estimate performance of the `Random Forest` method -----------
# nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,
# trying increase its value to 20 in this section.

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
  put_log("Predicted Data have been loaded from cache.")
  put_end_date(start)
  # Time difference of 
  
} else {
  put_log("Predicting `RF.mtry9` model on `x0.1.test`...")
  
  y0.1_hat_rf.mtry9 <- stats::predict(fit_rf.nzv.mtry9, x0.1.test, type = "response")
  
  put_log("The `RF.mtry9` Model: Generating predictions have been completed `x0.1.test` dataset.")
  put_end_date(start)
  # Time difference of 2.74791 mins
  
  
  put_log("Validating accuracy of the `RF.mtry9` Model predictions 
made for the `x0.1.test` dataset...")

  xy0.rf.mtry9.accuracy <- mean(y0.1_hat_rf.mtry9 == y0.1.test)
  # Time difference of ??? mins
  put_log("The accuracy value is %1", xy0.rf.mtry9.accuracy)
#> [1] 0.876092421884353
  
  put_log("Saving predicted results...")

    save(y0.1_hat_rf.mtry9,
       xy0.rf.mtry9.accuracy,
       file = cache_file.path)
    
    put_log("The Train fit result has been saved to the cache file:
%1.", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()

put_log("Summary of predicted data made using the `RF.mtry9` model,
trained on a 10% sample of the`Train Set` dataset for `mtry = 9`,
and tested on the 10% sample from the remaining 90% data of the `Train Set`:
%1", summary(y0.1_hat_rf.mtry9),
        capture_output = 1)


put_log("Accuracy of the predicted data for the `RF.mtry9` model,
trained on a 10% sample of the`Train Set` dataset for `mtry = 9`,
and tested on the 10% sample from the remaining 90% data of the `Train Set`:
%1", xy0.rf.mtry9.accuracy)
#> 0.876092421884353

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Preprocessing datasets -------------------------------------------
open_logfile("preprocess.datasets")
### Preprocessing datasets -------------------------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "preprocessed-datasets.RData")

if (file.exists(cache_file.path)) {
  put_log("Loading Preprocessed Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Preprocessed Data has been loaded from cache.")
  
} else {
  start <- put_start_date()
  nzv <- nearZeroVar(x0.1.train)
  nzv.test <- nearZeroVar(x0.1.test)
  put_end_date(start)
  # Time difference of 56.0386 secs
  
  x0.1.train_nzv <- x0.1.train[, -nzv]
  dim(x0.1.train_nzv)
  # [1] 75032 743
  
  x0.1.test_nzv <- x0.1.test[, -nzv]
  dim(x0.1.test_nzv)
  # [1] 8353 743
  
  put_log("Pre-training `RF.mtry9` model on a 10% sample from the `Train set`,
          pre-processed by NZV method (`x0.1.train_nzv`)..." )
  
  start <- put_start_date()
  
  put_log("Preprocessing transformation using the `Train Set` (`x0.1.train` object)...")
  start <- put_start_date()
  pp0.1 <- preProcess(x0.1.train, method = c("nzv", "center", "scale")) 
  
  put_log("Preprocessing transformation has been completed")
  put_end_date(start)
  
  start <- put_start_date()
  put_log("Applying preprocess transformation on the datasets...")
  x0.1.train.preprocessed <- stats::predict(pp0.1, x0.1.train)
  put_log("Preprocess transformation has been applied on the `x0.1.train` object:
  %1", capture.output(str(x0.1.train.preprocessed)))
  
  x0.1.test.preprocessed <- stats::predict(pp0.1, x0.1.test)
  put_log("Preprocess transformation has been applied on the `x0.1.test` object:
  %1", capture.output(str(x0.1.test.preprocessed)))
  put_end_date(start)
  
  start <- put_start_date()
  put_log("Binarizing the datasets...")
  x0.1.train.binarized <- x.binarize(x0.1.train)
  put_log("`x0.1.train` object has been binarized:
  %1", capture.output(str(x0.1.train.binarized)))
  
  x0.1.test.binarized <- x.binarize(x0.1.test)
  put_log("Preprocess transformation has been applied on the `x0.1.test` object:
  %1", capture.output(str(x0.1.test.binarized)))
  put_end_date(start)
  
  put_log("Caching Preprocessed Data in the file system...")
  start <- put_start_date()
  save(x0.1.train_nzv,
       x0.1.train.preprocessed,
       x0.1.train.binarized,
       x0.1.test.preprocessed,
       x0.1.test.binarized,
       file = cache_file.path)
  
  put_log("The Preprocessed data has been saved to the cache file:
  %1.", cache_file.path)
  put_end_date(start)
}

##### Close Log ------------------------------------------------------------------
log_close()
open_logfile(".research.x0.1.train.preprocessed.fit_rf.mtry18")
### Open log: `NZV` model for the default mtry  (NA) & ntree = 200 -------
open_logfile("x0.1.train.nzv.fit_rf.mtry_default")
##### RF: Default value of `mtry (NA)` & ntree = 200 ---------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry_default.accuracy.RData")

start <- put_start_date()
fit_rf.mtry_default.tuned_result <- tune.rf(x0.1.train_nzv, 
                                        y0.1.train,
                                        x0.1.test,
                                        y0.1.test,
                                        cache_file = cache_file.path)
# Time difference of the last iteration 19.8342 mins

put_log("Structure of results of tuning the model for the default value of parameter `mtry (NA)`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry_default.tuned_result)))
put_end_date(start)
# Time difference of 6.260901 hours

fit_rf.mtry_default.accuracy <- 
  sapply(fit_rf.mtry_default.tuned_result, 
         function(result) result$accuracy)

# plot(mtry_default, fit_rf.mtry_default.accuracy)

max.idx <- which.max(fit_rf.mtry_default.accuracy)

max_accuracy <- max(fit_rf.mtry_default.accuracy)
max_accuracy
# [1] 0.8833952

best_mtry <- mtry_default[[max.idx]]
best_mtry
# [1] 25



##### Close Log ----------------------------------------------------------------
log_close()

### Open log: Optimizing for mtry = 9 ------------------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry9")
##### Optimizing for mtry = 9 --------------------------------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry9.RData")
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

start <- put_start_date()

if (file.exists(cache_file.path)) {
  put_log("Loading Model Fit Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
  
} else {
  fit_rf.nzv.mtry9 <- randomForest(x0.1.train_nzv, y0.1.train,  mtry = 9)

  put_log("The `RF` model has been pre-trained on the dataset: `x0.1.train`
with parameter value: `.mtry = 9`." )
  put_end_date(start)
  #> Time difference of 44.35714 mins

  put_log("Saving Pre-train fit result...")
  
  save(fit_rf.nzv.mtry9,
       file = cache_file.path)

  put_log("The Pre-train fit result has been saved to the cache file:
%1.", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()
  
plot(fit_rf.nzv.mtry9)

put_log("Summary of fitting results obtained during the preliminary training of the `RFs` model:
%1", summary(fit_rf.nzv.mtry9),
        capture_output = 1)
# str(fit_rf.nzv.mtry9)
mean(fit_rf.nzv.mtry9$err.rate)

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing pre-processed model for mtry = 18 & ntree = 200 -------
open_logfile("x0.1.train.preprocessed.fit_rf.mtry18")
##### Optimizing pre-processed model for mtry = 18 & ntree = 200 ---------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.preprocessed.fit_rf.mtry18.accuracy.RData")

mtry18 <- 18
start <- put_start_date()
fit_rf.pp.mtry18.tuned_result <- tune.rf(x0.1.train.preprocessed, 
                                      y0.1.train,
                                      x0.1.test.preprocessed,
                                      y0.1.test,
                                      mtry = mtry18,
                                      cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 18`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.pp.mtry18.tuned_result)))
put_end_date(start)
# Time difference of 17.51424 mins

fit_rf.mtry18.accuracy <- 
  sapply(fit_rf.pp.mtry18.tuned_result, 
         function(result) result$accuracy)

plot(mtry18, fit_rf.mtry18.accuracy)

max.idx <- which.max(fit_rf.mtry18.accuracy)

max_accuracy <- max(fit_rf.mtry18.accuracy)
max_accuracy
# [1] 0.88136

best_mtry <- mtry18[[max.idx]]
best_mtry
# [1] 18

##### Close Log ----------------------------------------------------------------
log_close()

### Open log: Optimizing pre-processed model for mtry = 25 & ntree = 200 -------
open_logfile("x0.1.train.preprocessed.fit_rf.mtry25")
##### Optimizing pre-processed model for mtry = 25 & ntree = 200 ---------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.preprocessed.fit_rf.mtry25.accuracy.RData")

mtry25 <- 25
start <- put_start_date()
fit_rf.pp.mtry25.tuned_result <- tune.rf(x0.1.train.preprocessed, 
                                      y0.1.train,
                                      x0.1.test.preprocessed,
                                      y0.1.test,
                                      mtry = mtry25,
                                      cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 25`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.pp.mtry25.tuned_result)))
put_end_date(start)
# Time difference of 17.72548 mins

fit_rf.mtry25.accuracy <- 
  sapply(fit_rf.pp.mtry25.tuned_result, 
         function(result) result$accuracy)

plot(mtry25, fit_rf.mtry25.accuracy)

max.idx <- which.max(fit_rf.mtry25.accuracy)

max_accuracy <- max(fit_rf.mtry25.accuracy)
max_accuracy
# [1] 0.8825572

best_mtry <- mtry25[[max.idx]]
best_mtry
# [1] 25

##### Close Log ----------------------------------------------------------------
log_close()

### Open log: Optimizing binarized model for mtry = 25 & ntree = 200 -------
open_logfile("x0.1.train.binarized.fit_rf.mtry25")
##### Optimizing binarized model for mtry = 25 & ntree = 200 ---------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.binarized.fit_rf.bin.mtry25.accuracy.RData")

mtry25 <- 25
start <- put_start_date()
fit_rf.bin.mtry25.tuned_result <- tune.rf(x0.1.train.binarized, 
                                      y0.1.train,
                                      x0.1.test.binarized,
                                      y0.1.test,
                                      mtry = mtry25,
                                      cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 25`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.bin.mtry25.tuned_result)))
put_end_date(start)

fit_rf.bin.mtry25.accuracy <- 
  sapply(fit_rf.bin.mtry25.tuned_result, 
         function(result) result$accuracy)

plot(mtry25, fit_rf.bin.mtry25.accuracy)

max.idx <- which.max(fit_rf.bin.mtry25.accuracy)

max_accuracy <- max(fit_rf.bin.mtry25.accuracy)
max_accuracy
# [1] 0.8807614

best_mtry <- mtry25[[max.idx]]
best_mtry
# [1] 25

##### Close Log ----------------------------------------------------------------
log_close()

### Open log: Optimizing for mtry = c(5, 10, 15) -------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry5,10,15")
##### Optimizing for mtry = c(5, 10, 15) & ntree = 200 -------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry5,10,15.accuracy.RData")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

start <- put_start_date()

if (file.exists(cache_file.path)) {
  put_log("Loading Model Fit Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
  
} else {
  
  fit_rf.nzv.mtry5_10_15 = c(5, 10, 15)
  
  # Time difference of 56.0386 secs
  
  
  fit_rf.nzv.mtry5_10_15.tuned_result <- lapply( fit_rf.nzv.mtry5_10_15, function(mtry.val){
    put_log("Tuning `RF` model for `mtry = %1`...", mtry.val)
    start <- put_start_date()
    
    fit <-randomForest(x0.1.train_nzv, 
                       y0.1.train,  
                       mtry = mtry.val, 
                       ntree = 200)
    
    plot(fit)

    put_log("The `RF` model has been pre-trained on the dataset: `x0.1.train`
with parameter value: `.mtry = %1`.", mtry.val)
    put_end_date(start)
    
    put_log("Predicting `RF` model on `x0.1.test` for `mtry = %1`...", mtry.val)
    start <- put_start_date()
    
    y_hat <- stats::predict(fit, x0.1.test, type = "response")
    
    put_log("The `RF` Model: Generating predictions task has been completed.")

    
    put_log("Validating accuracy of the `RF.mtry9` Model predictions 
made for the `x0.1.test` dataset...")
    
    acc <- mean(y_hat == y0.1.test)
    put_log("The accuracy value is %1", acc)
    put_end_date(start)
    # Time difference of ??? mins
    
    c(mtry=mtry.val, 
      predictions = y_hat,
      err.rate = fit$err.rate,
      accuracy = acc)
  }) 

  put_end_date(start)
  #> Time difference of 44.35714 mins

  put_log("Saving the model tuning result...")
  
  save(fit_rf.nzv.mtry5_10_15.tuned_result,
       fit_rf.nzv.mtry5_10_15,
       file = cache_file.path)

  put_log("The Pre-train fit result has been saved to the cache file:
%1.", cache_file.path)
}

stopCluster(cl)
stopImplicitCluster()
put_log("Summary of tuned results for the `RF` model:
%1", summary(fit_rf.nzv.mtry5_10_15.tuned_result),
        capture_output = 1)

str(fit_rf.nzv.mtry5_10_15.tuned_result)

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = c(11, 12, 13, 14) -------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry11-14")
##### Optimizing for mtry = c(11, 12, 13, 14) & ntree = 200 -------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry11-14.accuracy.RData")

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = c(16, 17, 20) ------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry16,17,20")
##### Optimizing for mtry = c(16, 17, 20) & ntree = 200 -------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry16,17,20.accuracy.RData")
mtry16.17.20 = c(16, 17, 20)

start <- put_start_date()
fit_rf.nzv.mtry16.17.20.tuned_result <- tune.rf(x0.1.train_nzv, 
                                                y0.1.train,
                                                x0.1.test,
                                                y0.1.test,
                                                mtry = mtry16.17.20,
                                                cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.nzv.mtry16.17.20.tuned_result)))
put_end_date(start)

fit_rf.mtry16.17.20.accuracy <- 
  sapply(fit_rf.nzv.mtry16.17.20.tuned_result, 
         function(result) result$accuracy)

plot(mtry16.17.20, fit_rf.mtry16.17.20.accuracy)

max.idx <- which.max(fit_rf.mtry16.17.20.accuracy)

max_accuracy <- max(fit_rf.mtry16.17.20.accuracy)
max_accuracy
# [1] 0.8804023

best_mtry <- mtry16.17.20[max.idx]
best_mtry
# [1] 16

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = c(18, 19) ------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry18,19")
##### Optimizing for mtry = c(18, 19) & ntree = 200 -------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry18,19.accuracy.RData")

mtry18_19 <- c(18, 19)
start <- put_start_date()
fit_rf.mtry18_19.tuned_result <- tune.rf(x0.1.train_nzv, 
                                                y0.1.train,
                                                x0.1.test,
                                                y0.1.test,
                                                mtry = mtry18_19,
                                                cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry18_19.tuned_result)))
put_end_date(start)

fit_rf.mtry18_19.accuracy <- 
  sapply(fit_rf.mtry18_19.tuned_result, 
         function(result) result$accuracy)

plot(mtry18_19, fit_rf.mtry18_19.accuracy)

max.idx <- which.max(fit_rf.mtry18_19.accuracy)

max_accuracy <- max(fit_rf.mtry18_19.accuracy)
max_accuracy
# [1] 0.8817191

best_mtry <- mtry18_19[[max.idx]]
best_mtry
# [1] 18

##### Close Log ----------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = 18 ------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry18")
##### Optimizing for mtry = 18 & ntree = 200 -------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry18.accuracy.RData")

mtry18 <- 18
start <- put_start_date()
fit_rf.mtry18.tuned_result <- tune.rf(x0.1.train_nzv, 
                                                y0.1.train,
                                                x0.1.test,
                                                y0.1.test,
                                                mtry = mtry18,
                                                cache_file = cache_file.path)

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry18.tuned_result)))
put_end_date(start)

fit_rf.mtry18.accuracy <- 
  sapply(fit_rf.mtry18.tuned_result, 
         function(result) result$accuracy)

plot(mtry18, fit_rf.mtry18.accuracy)

max.idx <- which.max(fit_rf.mtry18.accuracy)

max_accuracy <- max(fit_rf.mtry18.accuracy)
max_accuracy
# [1] 0.8817191

best_mtry <- mtry18[[max.idx]]
best_mtry
# [1] 18

##### Close Log ----------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = 7:25 ---------------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry7_25")
##### RF: Optimizing for mtry = 7:25 & ntree = 200 ---------------------------------
cache_file.path <- file.path(models.random_forest.research.path, 
                             "x0.1.train.fit_rf.nzv.mtry7_25.accuracy.RData")

mtry7_25 <- seq(7,25)
start <- put_start_date()
fit_rf.mtry7_25.tuned_result <- tune.rf(x0.1.train_nzv, 
                                                y0.1.train,
                                                x0.1.test,
                                                y0.1.test,
                                                mtry = mtry7_25,
                                                cache_file = cache_file.path)
# Time difference of the last iteration 19.8342 mins

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry7_25.tuned_result)))
put_end_date(start)
# Time difference of 6.260901 hours

fit_rf.mtry7_25.accuracy <- 
  sapply(fit_rf.mtry7_25.tuned_result, 
         function(result) result$accuracy)

plot(mtry7_25, fit_rf.mtry7_25.accuracy)

max.idx <- which.max(fit_rf.mtry7_25.accuracy)

max_accuracy <- max(fit_rf.mtry7_25.accuracy)
max_accuracy
# [1] 0.8833952

best_mtry <- mtry7_25[[max.idx]]
best_mtry
# [1] 25

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = 20:30 --------------------------------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry20_30")
##### RF: Optimizing for mtry = 20:30 & ntree = 200 ---------------------------------
cache_root.path <- file.path(models.random_forest.research.path, "x0.1.train_nzv")

mtry20_30 <- seq(20,30)
start <- put_start_date()
fit_rf.mtry20_30.tuned_result <- tune.rf(x0.1.train_nzv, 
                                         y0.1.train,
                                         x0.1.test,
                                         y0.1.test,
                                         mtry = mtry20_30,
                                         cache_root = cache_root.path,
                                         cache_file = "x0.1.train.fit_rf.nzv.mtry20_30.accuracy.RData")

# Time difference of the last iteration 19.8342 mins

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry20_30.tuned_result)))
put_end_date(start)
# Time difference of 6.260901 hours

fit_rf.mtry20_30.accuracy <- 
  sapply(fit_rf.mtry20_30.tuned_result, 
         function(result) result$accuracy)

plot(mtry20_30, fit_rf.mtry20_30.accuracy)

max.idx <- which.max(fit_rf.mtry20_30.accuracy)

max_accuracy <- max(fit_rf.mtry20_30.accuracy)
max_accuracy
# [1] 0.8833952

best_mtry <- mtry20_30[[max.idx]]
best_mtry
# [1] 25

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = 27,28,29,30,45,60,90,150,180 ---------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry27,28,29,30,45,60,90,150,180")
##### RF: Optimizing for mtry = 27,28,29,30,45,60,90,150,180 & ntree = 200 -----
cache_root.path <- file.path(models.random_forest.research.path, "x0.1.train_nzv")

mtry27_30.45.60.90.150.180 <- c(27,28,29,30,45,60,90,150,180)
start <- put_start_date()
fit_rf.mtry27_30.45.60.90.150.180.tuned_result <- tune.rf(x0.1.train_nzv, 
                                         y0.1.train,
                                         x0.1.test,
                                         y0.1.test,
                                         mtry = mtry27_30.45.60.90.150.180,
                                         cache_root = cache_root.path,
                                         cache_file = "x0.1.train.fit_rf.nzv.mtry27_30.45.60.90.150.180.accuracy.RData")

# Time difference of the last iteration 19.8342 mins

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry27_30.45.60.90.150.180.tuned_result)))
put_end_date(start)
# Time difference of 6.260901 hours

fit_rf.mtry27_30.45.60.90.150.180.accuracy <- 
  sapply(fit_rf.mtry27_30.45.60.90.150.180.tuned_result, 
         function(result) result$accuracy)

plot(mtry27_30.45.60.90.150.180, fit_rf.mtry27_30.45.60.90.150.180.accuracy)

max.idx <- which.max(fit_rf.mtry27_30.45.60.90.150.180.accuracy)

max_accuracy <- max(fit_rf.mtry27_30.45.60.90.150.180.accuracy)
max_accuracy
# [1] 0.8833952

best_mtry <- mtry27_30.45.60.90.150.180[[max.idx]]
best_mtry
# [1] 25

##### Close Log ------------------------------------------------------------------
log_close()
### Open log: Optimizing for mtry = 45,49,53,57,60,64,68,72,76,80,85,90 ---------------
open_logfile(".research.x0.1.train.fit_rf.nzv.mtry45_60_90.4")
##### RF: Optimizing for mtry = 45,49,53,57,60,64,68,72,76,80,85,90 & ntree = 200 -----
cache_root.path <- file.path(models.random_forest.research.path, "x0.1.train_nzv")

mtry45_60_90.4 <- c(45,49,53,57,60,64,68,72,76,80,85,90)
start <- put_start_date()
fit_rf.mtry45_60_90.4.tuned_result <- tune.rf(x0.1.train_nzv, 
                                         y0.1.train,
                                         x0.1.test,
                                         y0.1.test,
                                         mtry = mtry45_60_90.4,
                                         cache_root = cache_root.path,
                                         cache_file = "x0.1.train.fit_rf.nzv.mtry45_60_90.4.accuracy.RData")

# Time difference of the last iteration 19.8342 mins

put_log("Structure of results of tuning the model for parameter `mtry = 16, 17, 20`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset,
pre-processed using `Nzv` method, and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(str(fit_rf.mtry45_60_90.4.tuned_result)))
put_end_date(start)
# Time difference of 6.260901 hours

fit_rf.mtry45_60_90.4.accuracy <- 
  sapply(fit_rf.mtry45_60_90.4.tuned_result, 
         function(result) result$accuracy)

plot(mtry45_60_90.4, fit_rf.mtry45_60_90.4.accuracy)

max.idx <- which.max(fit_rf.mtry45_60_90.4.accuracy)

max_accuracy <- max(fit_rf.mtry45_60_90.4.accuracy)
max_accuracy
# [1] 0.8833952

best_mtry <- mtry45_60_90.4[[max.idx]]
best_mtry
# [1] 25

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


## Deep Learning Methods -------------------------------------------------------
### Open log: Split Train Dataset (x) -------------
open_logfile(".split.80%train.balanced_subset")
#### Split Train Dataset  (90% for Train set) ----------------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x <- dim(x)
dim.x
dim.x[1]
dim.x[2]

sample_seed <- dim.x[1]
sample_seed
shuffle_seed <- as.integer(sample_seed*test_ratio)
shuffle_seed

cache_file.path <- file.path(ds.subsets.path, "x.9bl.train(balanced).RData")
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
  split.list <- sample_train_test_sets.mx(x, 
                                          sample_seed,
                                          test.ratio = 0.1,
                                          shuffle.seed = shuffle_seed)
  str(split.list)
  
  x.9bl.train <- split.list$train_set
  y.9bl.train <- as.factor(rownames(x.9bl.train))
  
  x.1.test <- split.list$test_set
  y.1.test <- as.factor(rownames(x.1.test))
  
  
  #   x0.1.test.list <- splitDataset(split.list$test_set, 9)
  #   put_log1("Test dataset list structure:
  # %1", capture.output(str(x0.1.test.list)))
  
  put_log1("Caching data in the file
%1 ...", cache_file.path)
  
  save(x.9bl.train,
       y.9bl.train,
       x.1.test,
       y.1.test,
       file = cache_file.path)
  
  put_log1("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  
  rm(split.list)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x.9bl.train)
#> [1] 16653   784
str(x.9bl.train)

str(y.9bl.train)
length(y.9bl.train)

str(x.1.test)
str(y.1.test)
length(y.1.test)
#> [1] 817379


### Close Log ------------------------------------------------------------------
log_close()

### Basic DL Classifier -----------------------------------------------------------
# Reference:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/

y.9bl.train.cat <- to_categorical(y.9bl.train)
colnames(y.9bl.train.cat) <- y.labels
dim(y.9bl.train.cat)
str(y.9bl.train.cat)
head(y.9bl.train.cat)
# max(y.9bl.train.cat)

y.1.test.cat <- to_categorical(y.1.test)
colnames(y.1.test.cat) <- y.labels
dim(y.1.test.cat)
str(y.1.test.cat)
head(y.1.test.cat)

#### Open log: Building Basic DL Model -----------------------------------------
open_logfile("dl.basic-model")
#### Basic DL Model building on dataset: `dl.basic_model`: `x0.1.dl.model` ---------
dl.keras3.path <- file.path(models.path, "dl.keras3")

if(!dir.exists(dl.keras3.path))
  dir.create(dl.keras3.path)

cache_file.path <- file.path(dl.keras3.path, 
                             "dl.basic.x.9bl.train.model.RData")

if (file.exists(cache_file.path)) {
  put_log("Loading `DL Keras3` model from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("`DL Keras3` model has been loaded from cache.")
  
} else {
  n.input_shape <- ncol(x.9bl.train)
  # 784
  
  n.output <- length(y.labels)
  # 39
  
  n.hl.units <- ceiling(n.input_shape*2/3+n.output)
  # 562
  
  dl.basic_model <- keras_model_sequential() |>
  layer_dense(units = n.hl.units, activation = "relu", 
              input_shape = c(n.input_shape)) |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = n.output, activation = "softmax")

  summary(dl.basic_model)

  dl.basic_model |> compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  put_log("Training the Basic DL Model...")
  start <- put_start_date()
  
  dl.basic.x.9bl.train.history <- dl.basic_model |> 
    fit(x.9bl.train, 
        y.9bl.train.cat, 
        epochs = 100, 
        batch_size = 512, 
        validation_split = 0.15)
  
  put_log("The Basic DL Model has been trained on `x.9bl.train` dataset.")
  put_end_date(start)
  
  put_log("Saving `DL Keras3` model to the cache file...")
  start <- put_start_date()
  
  save(dl.basic_model,
       dl.basic.x.9bl.train.history,
       file = cache_file.path)
  
  put_log("The `DL Keras3` model has been saved to the cache file: 
%1", cache_file.path)
  put_log("Training Basic DL Model task has been completed on the Train Set 
(90% balanced sample of the dataset).")
  put_end_date(start)
# Time difference of 38.48235 mins
}

plot(dl.basic.x.9bl.train.history)
str(dl.basic.x.9bl.train.history)
#### DL Basic Model Evaluation ----------------------------------------------

start <- put_start_date()
dl.basic_model |> evaluate(x.1.test, y.1.test.cat)
# $accuracy
# [1] 0.9065974
# 
# $loss
# [1] 1.154121

put_end_date(start)
# Time difference of 1.30212 mins

start <- put_start_date()
preds <- dl.basic_model |>
  predict(x.1.test) 
put_end_date(start)
# Time difference of  mins

colnames(preds) <- y.labels
head(preds[,1:5])
#                 #            $            &            @            0
# [1,] 8.469580e-25 2.824278e-25 1.338040e-31 1.180656e-36 3.503761e-16
# [2,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
# [3,] 9.542136e-15 3.782388e-15 1.619873e-16 3.058267e-19 4.522308e-08
# [4,] 7.770129e-13 1.467184e-19 8.129414e-25 4.881509e-21 9.999547e-01
# [5,] 4.731567e-38 0.000000e+00 0.000000e+00 0.000000e+00 5.828601e-27
# [6,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00

dim(preds)
#> [1] 684467     39

preds.ts <- as_tensor(preds)
str(preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

predictions <- preds.ts |> op_argmax(2)
predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(predictions)
#> [1] 684467
# predictions$numpy()


# y.1.test
# as.integer(y.1.test)

mean(predictions$numpy() == as.integer(y.1.test))
# [1] 0.9065974

##### Close Log ------------------------------------------------------------------
log_close()

### Convolutional Neural Network (CNN) -----------------------------------------
# Reference:
# Deep Learning Using R with keras (CNN)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

#### Open log: Prepare CNN Datasets -------------
open_logfile(".prepare-cnn-datasets")

#### Prepare CNN Datasets ------------------------------------------------------
start <- put_start_date()
put_log("Loading Binary Image 28x28 Matrix list from cache.")
load(train.img28x28bin.list.file_path)
put_log("The Binary Image 28x28 Matrix list has been loaded from cache.")
put_end_date(start)
# Time difference of 6.805764 secs


# ----
put_log("Building flatten (`EMNIST`-like) dataset...")
x_flat <- img28x28.list2flatten.mx(img28x28bin.list$img.list)
put_log("The flatten dataset have been created.")
dim(x_flat)
# ----

img.nested_list <- img28x28bin.list$img.list
str(img.nested_list)



start <- put_start_date()
put_log("Combining nested list of images to list of arrays...")

img28x28mx.list <- lapply(img.nested_list, function(item){
  # char.image(item$img.list[[1]])
  # img.array <- simplify2array(item$img.list)
  img.array <- abind(item$img.list, rev.along = 3)
  # dim(img.array)
  # str(img.array)
  # char.image(img.array[1,,])
})

put_log("Function `img28x28.list2matrix`:
Combined image data matrix has been created with the following structure:
%1", capture.output(str(img.mx)))
put_end_date(start)
str(img28x28mx.list)

start <- put_start_date()
put_log("Combining image list to array...")
x <- abind(img28x28mx.list, along = 1)

put_log("Function `img28x28.list2matrix`:
Combined image matrix array has been created with the following dimentions:
%1", capture.output(dim(x)))
# char.image(x[2,,])
put_end_date(start)

dim(x)
dim.x <- dim(x)
dim.x
#> [1] 834032     28     28
nrow(x)
#> [1] 834032

# Input image dimensions
img_rows <- dim.x[2]
img_rows
img_cols <- dim.x[3]
img_cols

first_G.idx <- which(y == "G")[1]
#> [1] 598137

char.image(x[first_G.idx,,])
char.image(x[first_G.idx - 1,,])

# # Add channel into the dimension
# x3d <- array_reshape(x, c(dim.x[1], dim.x[2], dim.x[3], 1))
# dim(x3d)

rownames(x3d) <- as.character(y)
str(x3d)

input_shape <- c(dim.x[2], dim.x[3], 1)
input_shape
#> [1] 28 28  1
#### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Split Train Dataset (x3d) -------------
open_logfile(".split3d.10%train.balanced_subset")
#### Split Train Dataset  (10% for Train set) ----------------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

sample_seed <- 1
shuffle_seed <- 2

cache_file.path <- file.path(ds.subsets.path, "cnn1-datasets.RData")
cache_file.path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log("Loading Split Train Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
} else {
  # ----
    x_flat.split.list <- sample_train_test_sets.mx(x_flat, 
                                                  sample_seed,
                                                  test.ratio = 0.9,
                                                  shuffle.seed = shuffle_seed)
    
    x_flat.1bl.train <- x_flat.split.list$train_set
    dim(x_flat.1bl.train)
    y_flat.1bl.train <- as.factor(rownames(x_flat.1bl.train))
    sum(y_flat.1bl.train != y_cnn.1bl.train)
  # --------------

  split.list <- sample_train_test_sets.x3d(x, 
                                          sample_seed,
                                          test.ratio = 0.9,
                                          shuffle.seed = shuffle_seed)
  str(split.list)
  
  x_train <- split.list$train_set
  dim(x_train)
  #> [1] 16653    28    28
  nrow(x_train)
  #> [1] 16653
  
  y_cnn.1bl.train <- as.factor(rownames(x_train))
  
  y_cnn.1bl.train.cat <- to_categorical(y_cnn.1bl.train)
  colnames(y_cnn.1bl.train.cat) <- y.labels
  
  
  # Add channel into the dimension
  x_cnn.1bl.train <- array_reshape(x_train, 
                                   c(nrow(x_train), 
                                     img_rows, 
                                     img_cols, 
                                     1))
  x_test <- split.list$test_set
  dim(x_test)
  #> [1] 817379     28     28
  nrow(x_test)
  #> [1] 817379
  
  y_cnn.9.test <- as.factor(rownames(x_test))

  y_cnn.9.test.cat <- to_categorical(y_cnn.9.test)
  colnames(y_cnn.9.test.cat) <- y.labels
  
  # Add channel into the dimension
  x_cnn.9.test <- array_reshape(x_test, 
                                c(nrow(x_test), 
                                  img_rows, 
                                  img_cols, 
                                  1))
  start <- put_start_date()
  put_log("Caching data in the file
%1 ...", cache_file.path)
  
  save(x_cnn.1bl.train,
       y_cnn.1bl.train,
       y_cnn.1bl.train.cat,
       x_cnn.9.test,
       y_cnn.9.test,
       y_cnn.9.test.cat,
       file = cache_file.path)
  
  put_log("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  put_end_date(start)
  
  rm(split.list)
  rm(x_train)
  rm(x_test)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x_cnn.1bl.train)
#> [1] 16653    28    28     1

# str(y_cnn.1bl.train)
length(y_cnn.1bl.train)
#> [1] 16653

dim(y_cnn.1bl.train.cat)
#> [1] 16653    39
str(y_cnn.1bl.train.cat)
head(y_cnn.1bl.train.cat[,1:30])

dim(x_cnn.9.test)
#> [1] 817379     28     28      1

# str(y_cnn.9.test)
length(y_cnn.9.test)
#> [1] [1] 817379

dim(y_cnn.9.test.cat)
#> [1] 817379     39
str(y_cnn.9.test.cat)
head(y_cnn.9.test.cat[,1:30])

### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Build CNN Model -------------------------------------------------
open_logfile(".build-cnn-model")
#### Model building ------------------------------------------------------------

# Define a CNN model structure:
#> Now we define a CNN model with two 2D convolutional layers with max pooling, 
#> and the 2nd layer with additonal dropout to prevent overfitting. 
#> Then flatten the output and use two dense layers to connect to the categoires 
#> of the image. [*]

cnn_model <- keras_model_sequential() |>
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
                input_shape = input_shape) |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_dropout(rate = 0.25) |>
  layer_flatten() |>
  layer_dense(units = 128, activation = 'relu') |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = num_classes, activation = 'softmax')

summary(cnn_model)
plot(cnn_model)

# Similar to DNN model, we need to compile the defined CNN model. [*]

# Compile model
cnn_model |> compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

#> Now, we can train the model with our processed data. 
#> Each epochs's history can be saved to track the progress. 
#> Please note, as we are not using GPU, it takes a few minutes to finish. 
#> Please be patient while waiting for the results. 
#> The training time can be significantly reduced if running on GPU. [*]

# Define a few parameters to be used in the CNN model
batch_size <- 128
num_classes <- 39
epochs <- 10
vld_split <- 0.2

dim(x_train)
x_train <- x_cnn.1bl.train
y_train <- y_flat.1bl.train
y_train.cat <- y_cnn.1bl.train.cat

dim(x_flat.1bl.train)
#> [1] 16653   784

x_train.list <- asplit(x_flat.1bl.train, 1)
str(x_train.list)

x_cnn.train.list <- lapply(x_train.list, function(v) {
  matrix(v, nrow = img_rows)
})

str(x_cnn.train.list)

x_cnn.train.array <- abind(x_cnn.train.list, rev.along = 3)
dim(x_cnn.train.array)
char.image(x_cnn.train.array[2,,])

x_cnn.train <- array_reshape(x_cnn.train.array, 
                             c(nrow(x_cnn.train.array), 
                               img_rows, 
                               img_cols, 
                               1))
dim(x_cnn.train)
char.image(x_cnn.train[1,,,1])
y_train.cat[1,1]
y_train[1]
next_char.idx <- min(which(y_train != "#"))
next_char.idx
#> [1] 428

y_train[428]
# [1] $
#   39 Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N ... Z
y_train[427]
# [1] #
# 39 Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N ... Z
y_train.cat[427,]
# # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W 
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
# X Y Z 
# 0 0 0 
y_train.cat[428,]
# # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W 
# 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
# X Y Z 
# 0 0 0 

char.image(x_cnn.train.array[428,,])
char.image(x_cnn.train.array[427,,])



put_log("Training the CNN Model...")
start <- put_start_date()

# Train model
cnn.1bl.train_history <- cnn_model |> 
  fit(x_cnn.train, 
      y_train.cat,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = vld_split
)

put_log("The CNN Model has been trained on `x_cnn.1bl.train` dataset.")
put_end_date(start)

# ------------

x_train <- x_flat.1bl.train



put_log("Training the Basic DL Model...")
start <- put_start_date()

dl_history.test <- dl.basic_model |> 
  fit(x_train, 
      y_train.cat, 
      epochs = epochs, 
      batch_size = batch_size, 
      validation_split = vld_split)

put_log("The Basic DL Model has been trained on `x.9bl.train` dataset.")
put_end_date(start)

# ----













#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
### Close Log ------------------------------------------------------------------
log_close()

### Advanced CNN Model ---------------------------------------------------------
# Reference:
# TensorFlow 2 quickstart for experts
# https://tensorflow.rstudio.com/tutorials/quickstart/advanced

# To use legacy `keras` package uncomment the code snippet below:
# detach("package:keras3", unload = TRUE)
# library(keras)
# py_require_legacy_keras()
# library(tensorflow)

#### Model Class
# Build the a model using the Keras model subclassing API:

my_model <- new_model_class(
  classname = "MyModel",
  initialize = function(...) {
    super$initialize()
    self$conv1 <- layer_conv_2d(filters = 32, kernel_size = 3,
                                activation = 'relu')
    self$flatten <- layer_flatten()
    self$d1 <- layer_dense(units = 128, activation = 'relu')
    self$d2 <- layer_dense(units = 10)
  },
  call = function(inputs) {
    inputs |>
      tf$expand_dims(3L) |>
      self$conv1() |>
      self$flatten() |>
      self$d1() |>
      self$d2()
  }
)

# Create an instance of the model
model <- my_model()

# Choose an optimizer and loss function for training:
loss_object <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
optimizer <- optimizer_adam()

#> Select metrics to measure the loss and the accuracy of the model. 
#> These metrics accumulate the values over epochs and then print the overall result.

train_loss <- metric_mean(name = "train_loss")
str(train_loss)
train_accuracy <- metric_sparse_categorical_accuracy(name = "train_accuracy")

test_loss <- metric_mean(name = "test_loss")
test_accuracy <- metric_sparse_categorical_accuracy(name = "test_accuracy")


# Use tf$GradientTape() to train the model:
  
train_step <- function(images, labels) {
  with(tf$GradientTape() %as% tape, {
    # training = TRUE is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions <- model(images, training = TRUE)
    loss <- loss_object(labels, predictions)
  })
  gradients <- tape$gradient(loss, model$trainable_variables)
  optimizer$apply_gradients(zip_lists(gradients, model$trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)
}

train <- tf_function(function(train_ds) {
  for (batch in train_ds) {
    c(images, labels) %<-% batch
    train_step(images, labels)
  }
})

# Test the model:

test_step <- function(images, labels) {
  # training = FALSE is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions <- model(images, training = FALSE)
  t_loss <- loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)
}

test <- tf_function(function(test_ds) {
  for (batch in test_ds) {
    c(images, labels) %<-% batch
    test_step(images, labels)
  }
})

reset_metrics <- function() {
  for (metric in list(train_loss, train_accuracy,
                      test_loss, test_accuracy))
    metric$reset_state()
}

#### Proceed Classification ----------------------------------------------------

EPOCHS <- 1
for (epoch in seq_len(EPOCHS)) {
  # Reset the metrics at the start of the next epoch
  reset_metrics()
  train(train_ds)
  test(test_ds)
  cat(sprintf('Epoch %d', epoch), "\n")
  cat(sprintf('Loss: %f', train_loss$result()), "\n")
  cat(sprintf('Accuracy: %f', train_accuracy$result() * 100), "\n")
  cat(sprintf('Test Loss: %f', test_loss$result()), "\n")
  cat(sprintf('Test Accuracy: %f', test_accuracy$result() * 100), "\n")
}




# ---------------------------
# Reference:
#
# 
start <- put_start_date()
put_end_date(start)






























