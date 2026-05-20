#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA & Random Forest Models
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Initial Paths ---------------------------------------------------------------
stopifnot(dir.exists(models.path))
models.random_forest.path <- file.path(models.path, "random-forest")

if(!dir.exists(models.random_forest.path))
  dir.create(models.random_forest.path)

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path))
  dir.create(knn_pca.path)

### Open log: Load Split Dataset -------------
open_logfile(".split.10%train.balanced_subset")

#### Load Split Train Dataset  (10% for Train set) ----------------------------------
start <- put_start_date()


if (!exists("ds_flatten.0.1split_list")) {
  stopifnot(file.exists(my_emnist.0.1split.file_path))

  put_log1("Loading the Split Flattened Dataset from the backup file...")
  
  ds_flatten.0.1split_list <- readRDS(my_emnist.0.1split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.0.1split.file_path)
} 


x.train <- ds_flatten.0.1split_list$train_set$x.train
x.test <- ds_flatten.0.1split_list$test_set$x.test
x.test.files <- ds_flatten.0.1split_list$test_set$x.files



y.train.groups <- ds.get_classIDs.grouped(x.train)
y.train <- y.train.groups$classID

stopifnot(sum(as.character(y.train) != rownames(x.train)) == 0)

put_log("The Train Set is balanced by set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
# A tibble: 39 × 2
#    classID     n
#    <fct>   <int>
  #  1 #         425
  #  2 $         425
  #  3 &         425
  #  4 @         425
  #  5 0         425
  #  6 1         425
  #  7 2         425
  #  8 3         425
  #  9 4         425
  # 10 5         425
  # 11 6         425
  # 12 7         425
  # 13 8         425
  # 14 9         425
  # 15 A         425
  # 16 B         425
  # 17 C         425
  # 18 D         425
  # 19 E         425
  # 20 F         425
  # 21 G         425
  # 22 H         425
  # 23 I         425
  # 24 J         425
  # 25 K         425
  # 26 L         425
  # 27 M         425
  # 28 N         425
  # 29 P         425
  # 30 Q         425
  # 31 R         425
  # 32 S         425
  # 33 T         425
  # 34 U         425
  # 35 V         425
  # 36 W         425
  # 37 X         425
  # 38 Y         425
  # 39 Z         425
}

y.test.groups <- ds.get_classIDs.grouped(x.test)
y.test <- y.test.groups$classID

stopifnot(sum(as.character(y.test) != rownames(x.test)) == 0)

put_log("The Test Set is balanced by set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
# A tibble: 39 × 2
#    classID     n
#    <fct>   <int>
#  1 #         852
#  2 $         852
#  3 &         852
#  4 @         852
#  5 0         852
#  6 1         852
#  7 2         852
#  8 3         852
#  9 4         852
# 10 5         852
# 11 6         852
# 12 7         852
# 13 8         852
# 14 9         852
# 15 A         852
# 16 B         852
# 17 C         852
# 18 D         852
# 19 E         852
# 20 F         852
# 21 G         852
# 22 H         852
# 23 I         852
# 24 J         852
# 25 K         852
# 26 L         852
# 27 M         852
# 28 N         852
# 29 P         852
# 30 Q         852
# 31 R         852
# 32 S         852
# 33 T         852
# 34 U         852
# 35 V         852
# 36 W         852
# 37 X         852
# 38 Y         852
# 39 Z         852  
}

dim(x.train)
#> [1] 16653   784
str(x.train)

str(y.train)
length(y.train)

str(x.test)
str(y.test)
length(y.test)
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
open_logfile(".pre-train-model.k1-7nn+pca")
#### Tuning k1_7NN+PCA model by *k* parameter ranging from 1 to 7 --------------
# (The training takes about half an hour)
k.values <- seq_len(8)

k1_7nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k1-7nn+pca(0.1train-set).rds")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(k1_7nn_pca.model.backup.path)) {
  put_log1("Loading pre-trained `kNN+PCA` Model (tuned for `k` values ranged from 1 to 7) from backup file: 
%1...", k1_7nn_pca.model.backup.path)
  
  start <- put_start_date()
  readRDS(k1_7nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("Model Fit Data have been loaded from cache.")
} else {
  put_log("Training Model `kNN+PCA` on the !0% size Train Set..." )
  
  start <- put_start_date()
  #flush.console()
  k1_7nn_pca.model <- caret::train(x.train, y.train, method = "knn", 
                                preProcess = "pca",
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9),
                                                         verboseIter = TRUE,
                                                         verbose = TRUE),
                                tuneGrid = data.frame(k = k.values))
  put_end_date(start)
  # Time difference of 40.88067 mins
  put_log("The Model `kNN+PCA` has been trained on the dataset subset: `x0.1.train`")

  put_log("Saving Model in the cache file: `kNN+PCA`...")

    saveRDS(k1_7nn_pca.model, 
          file = k1_7nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log1("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", k1_7nn_pca.model.backup.path)

}

stopCluster(cl)
stopImplicitCluster()

#### The Tuning Results Visualization & Analysis -------------------------------

put_log("kNN+PCA Model trained result:
%1", capture.output(k1_7nn_pca.model))

# k1_7nn_pca.model$results |>
#   ggplot(mapping = aes(x = k,
#                        y = Accuracy)) +
#   geom_col(fill = "steelblue",
#            color = "black") +
#   labs(x = "`k` Parameter value",
#        y = "Accuracy",
#        title = "Tuning the `kNN+PCA` model by *k* parameter") +
#   scale_y_continuous(labels = scales::label_percent(accuracy = 1),
#                      expand = c(0, 0, 0.005, 0))

k1_7nn_pca.model$results |>
  data.plot(title = "",
xname = "k",
yname = "Accuracy",
xlabel = "k Parameter",
ylabel = "Accuracy")

acc.max.idx <- which.max(k1_7nn_pca.model$results$Accuracy)
acc.max.idx

k.1to7step2.max_accuracy <- k1_7nn_pca.model$results$Accuracy[acc.max.idx]
k.1to7step2.max_accuracy

k.1to7step2.best <- k1_7nn_pca.model$results$k[acc.max.idx]
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

k4_6nn_pca.model.backup.path <-
  file.path(knn_pca.path, "x0.1.train.fine-tune.k4-6NN+PCA.RData")

start <- put_start_date()
# w.pc_cores <- as.integer(N_pcCores / 2 + 1) 
# w.pc_cores  

cl <- makeCluster(N_pcCores, type='PSOCK', outfile="")
registerDoParallel(cl)

if (file.exists(k4_6nn_pca.model.backup.path)) {
  put_log("Loading `kNN+PCA` model Fit Data from the cache file: 
%1", k4_6nn_pca.model.backup.path)
  
  load(k4_6nn_pca.model.backup.path)
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
  save(train_knn_pca.k4_6, file = k4_6nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log("The Model `kNN+PCA` trained on the dataset subset `x0.1.train` has been cached in file:
`%1`", k4_6nn_pca.model.backup.path)
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


