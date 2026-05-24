#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA & Random Forest Models
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Open log: Load Split Dataset -------------
open_logfile(".split.10%train.balanced_subset")

#### Loading Split Dataset allocated 10% for the Train Set ---------------------
start <- put_start_date()


if (!exists("ds_flatten.0.1split_list")) {
  stopifnot(file.exists(my_emnist.0.1split.file_path))

  put_log("Loading the Split Flattened Dataset from the backup file...")
  
  ds_flatten.0.1split_list <- readRDS(my_emnist.0.1split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.0.1split.file_path)
} 


x0.1.train <- ds_flatten.0.1split_list$train_set$x.train
x0.9.test <- ds_flatten.0.1split_list$test_set$x.test
x0.9.test.files <- ds_flatten.0.1split_list$test_set$x.files



y0.1.train.groups <- ds.get_classIDs.grouped(x0.1.train)
y0.1.train <- y0.1.train.groups$classID

stopifnot(sum(as.character(y0.1.train) != rownames(x0.1.train)) == 0)

put_log("The Train Set is balanced by set of Classes:
%1", capture.output(print(y0.1.train.groups$groupByClass, n = N.classes)))
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

y0.9.test.groups <- ds.get_classIDs.grouped(x0.9.test)
y0.9.test <- y0.9.test.groups$classID

stopifnot(sum(as.character(y0.9.test) != rownames(x0.9.test)) == 0)

put_log("The Test Set is balanced by set of Classes:
%1", capture.output(print(y0.9.test.groups$groupByClass, n = N.classes)))
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

dim(x0.1.train)
#> [1] 16653   784
str(x0.1.train)

str(y0.1.train)
length(y0.1.train)

str(x0.9.test)
str(y0.9.test)
length(y0.9.test)
#> [1] 817379

### Close Log ------------------------------------------------------------------
log_close()

## `kNN+PCA` Model -------------------------------------------------------------
# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

### `kNN+PCA` Model Initial Paths -----------------------------------------------------
stopifnot(dir.exists(models.path))
knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path))
  dir.create(knn_pca.path)

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
#### Tuning k1_7NN+PCA model by *k* parameter ranging from 1 to 7 on 10% size Train Set ----
# (The training takes about half an hour)
k.values <- seq_len(8)

k1_7nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k1-7nn+pca(0.1train-set).rds")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(k1_7nn_pca.model.backup.path)) {
  put_log("Loading pre-trained `kNN+PCA` Model 
(tuned for `k` values ranged from 1 to 7) from the backup file...")
  
  k1_7nn_pca.model <- readRDS(k1_7nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The pre-trained Model has been loaded from the following file:
%1", k1_7nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 10% size Train Set..." )
  
  start <- put_start_date()
  #flush.console()
  k1_7nn_pca.model <- caret::train(x0.1.train, y0.1.train, method = "knn", 
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
  put_log("The Model `kNN+PCA` has been trained on the 10% size Train Set")

  put_log("Saving the pre-trained model in the backup file...")

    saveRDS(k1_7nn_pca.model, 
          file = k1_7nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log("The Model `kNN+PCA` pre-trained on the 10% size Train Set 
for *k* values ranged from 1 to 7 has been backed up in the following file:
`%1`", k1_7nn_pca.model.backup.path)

}

stopCluster(cl)
stopImplicitCluster()

#### The Tuning Results Visualization & Analysis -------------------------------

put_log("The pre-trained `kNN+PCA` Model trained result:
%1", capture.output(k1_7nn_pca.model))

# The Model tuning visualzation:
k1_7nn_pca.model$results |>
  data.plot(title = "`kNN+PCA` Multiclass Classifier Model Tuning Results",
xname = "k",
yname = "Accuracy",
xlabel = "k",
ylabel = "Accuracy")

acc.max.idx <- which.max(k1_7nn_pca.model$results$Accuracy)
acc.max.idx

k.1_7.max_accuracy <- k1_7nn_pca.model$results$Accuracy[acc.max.idx]
k.1_7.max_accuracy

k.best <- k1_7nn_pca.model$results$k[acc.max.idx]
k.best
# 6
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

### Open log: Load Split Dataset (Test Set of 20% size) ------------------------
open_logfile(".split.20%test.balanced_subset")
#### Loading Split Dataset allocated 20% for the Test set (default) ------------

start <- put_start_date()

if (!exists("ds_flatten.split_list")) {
  stopifnot(file.exists(my_emnist.split.file_path))
  
  put_log("Loading the Split Flattened Dataset from the backup file...")
  
  ds_flatten.split_list <- readRDS(my_emnist.split.file_path)
  
  put_log("The Split Flattened Dataset has been loaded from the following backup file:
%1", my_emnist.split.file_path)
} 


x.train <- ds_flatten.split_list$train_set$x.train
x.test <- ds_flatten.split_list$test_set$x.test
x.test.files <- ds_flatten.split_list$test_set$x.files



y.train.groups <- ds.get_classIDs.grouped(x.train)
y.train <- y.train.groups$classID

stopifnot(sum(as.character(y.train) != rownames(x.train)) == 0)

put_log("The Train Set is balanced by set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #        3407
  #  2 $        3407
  #  3 &        3407
  #  4 @        3407
  #  5 0        3407
  #  6 1        3407
  #  7 2        3407
  #  8 3        3407
  #  9 4        3407
  # 10 5        3407
  # 11 6        3407
  # 12 7        3407
  # 13 8        3407
  # 14 9        3407
  # 15 A        3407
  # 16 B        3407
  # 17 C        3407
  # 18 D        3407
  # 19 E        3407
  # 20 F        3407
  # 21 G        3407
  # 22 H        3407
  # 23 I        3407
  # 24 J        3407
  # 25 K        3407
  # 26 L        3407
  # 27 M        3407
  # 28 N        3407
  # 29 P        3407
  # 30 Q        3407
  # 31 R        3407
  # 32 S        3407
  # 33 T        3407
  # 34 U        3407
  # 35 V        3407
  # 36 W        3407
  # 37 X        3407
  # 38 Y        3407
  # 39 Z        3407
}

y.test.groups <- ds.get_classIDs.grouped(x.test)
y.test <- y.test.groups$classID

stopifnot(sum(as.character(y.test) != rownames(x.test)) == 0)

put_log("The Train Set is balanced by set of Classes:
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

#### Open log: Training kNN+PCA Model for best *k% Parameter -------------------
open_logfile(".pre-train-model.k1-7nn+pca")
#### Training kNN+PCA Model for best *k% Parameter -----------------------------
# (The training takes about half an hour)

k_best.nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k_best.nn+pca.rds")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(k_best.nn_pca.model.backup.path)) {
  put_log("Loading the `kNN+PCA` Model (trained for the best `k` value) from the backup file...")
  
  k_best.nn_pca.model <- readRDS(k_best.nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The `kNN+PCA` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 80% size Train Set..." )
  
  start <- put_start_date()
  #flush.console()
  k_best.nn_pca.model <- caret::train(x.train, y.train, method = "knn", 
                                   preProcess = "pca",
                                   trControl = trainControl("cv", 
                                                            number = 5, 
                                                            p = 0.95,
                                                            preProcOptions = list(thresh = 0.9),
                                                            verboseIter = TRUE,
                                                            verbose = TRUE),
                                   tuneGrid = data.frame(k = k.best)) # *k* = 6
  put_end_date(start)
  # Time difference of 40.88067 mins
  put_log("The Model `kNN+PCA` has been trained on the 80% size Train Set")
  
  put_log("Saving `kNN+PCAM`odel in the backup file: `...")
  
  saveRDS(k_best.nn_pca.model, 
          file = k_best.nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 12.37275 secs
  
  put_log("The Model `kNN+PCA` trained on the 80% size Train Set has been cached in file:
`%1`", k_best.nn_pca.model.backup.path)
  
}

stopCluster(cl)
stopImplicitCluster()
  
# Log Elapsed Time for training & tuning `kNN+PCA`: 03:21:28

### Close Log ------------------------------------------------------------------
log_close()

##### Open log: Predictions on `k5NN+PCA` (best *k*) Model -------------------
open_logfile(".x.test.predict.k(best)nn+pca")
##### Constructing Predictions on kNN+PCA (for best *k* Parameter value) ------
knn_pca.best.preds.backup <-
  file.path(knn_pca.path, "x.test.k(best)NN+PCA.predictions.rds")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(knn_pca.best.preds.backup)) {
  put_log("Loading Predicted Data from cache file: 
%1...", knn_pca.best.preds.backup)
  
  knn_pca.best.preds <- readRDS(knn_pca.best.preds.backup)
  k_best.nn_pca.model.predicted <- knn_pca.best.preds$predicted
  knn_pca.best.accuracy <- knn_pca.best.preds$accuracy
  rm(knn_pca.best.preds)
  put_end_date(start)
  # Time difference of 
  
  put_log("Predicted Data have been loaded from cache.")
} else {
  put_log("Constructing predictions using the `KNN+PCA` model trained for the best *k* value...")
  
  if(!exists("k_best.nn_pca.model")) {
    stopifnot(file.exists(k_best.nn_pca.model.backup.path))
    
    put_log("Loading the `kNN+PCA` Model (trained for the best `k` value) from the backup file...")
    
    k_best.nn_pca.model -> readRDS(k_best.nn_pca.model.backup.path)
    put_end_date(start)
    # Time difference of 
    
    put_log("The `kNN+PCA` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)
  }
  
  k_best.nn_pca.model.predicted <- stats::predict(k_best.nn_pca.model, x.test, type = "raw")
  put_end_date(start)
  # Time difference of 2.25995 hours
  
  put_log("The (Best *k*) `kNN+PCA` Model: Generating predictions have been completed on `x.test` dataset.")
  
  put_log("Validating accuracy of the (Best *k*) `kNN+PCA` Model predictions 
made on the `x.test` dataset...")
  
  knn_pca.best.accuracy <- mean(k_best.nn_pca.model.predicted == y.test)

  put_log("Backing up the `kNN+PCA` Model's tuning best results to file...")
  #> [1] 0.8693882
  
  saveRDS(list(predicted = k_best.nn_pca.model.predicted,
               accuracy = knn_pca.best.accuracy,
               k_best = k.best),
       file = knn_pca.best.preds.backup)
  
  put_log("The accuracy of the (Best *k*) `kNN+PCA` Model prediction is 
%1", knn_pca.best.accuracy)
}

stopCluster(cl)
stopImplicitCluster()
#> [1] 

put_log("Accuracy of the predicted data for the `kNN+PCA` model tuned by *k* parameter:
%1", knn_pca.best.accuracy)
#> [1] 0.860810160105935


##### Close Log ----------------------------------------------------------------
log_close()

## Random Forest MCC Model -----------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

### `Random Forest (RF) MCC` Model Initial Paths -----------------------------------------------------

models.random_forest.path <- file.path(models.path, "random-forest")

if(!dir.exists(models.random_forest.path))
  dir.create(models.random_forest.path)

models.rf.tune.path = file.path(models.random_forest.path, "tune")

if(!dir.exists(models.rf.tune.path))
  dir.create(models.rf.tune.path)

### Open log: `RF MCC` model for the default mtry  & ntree = 500 --------------
open_logfile("x0.1.train.fit_rf.mtry_default.ntree500")
##### Train `RF MCC` model with the default mtry value & ntree = 500 ---------------------------
fit_rf.mtry_default.backup.path <- file.path(models.rf.tune.path, 
                             "fit_rf.mtry_default.ntree500.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry_default.backup.path)) {
  put_log("Loading the `RF MCC` model trained with the default `mtry` parameter value from the backup file...")
  
  fit_rf.mtry_default <- readRDS(fit_rf.mtry_default)
  
  put_Log("The `RF MCC` model trained with the default `mtry` parameter value 
has been loaded from the following backup file:
%1", fit_rf.mtry_default.backup.path)
  put_end_date(start)
} else {
  put_log("Training the `RF MCC` model with the default `mtry` parameter value...")
  set.seed(N.classes)
  fit_rf.mtry_default <- randomForest(x0.1.train, 
                                      y0.1.train,
                                      x0.9.test,
                                      y0.9.test,
                                      keep.forest = TRUE,
                                      ntree = 500)
  
  put_log("The `RF MCC` model has been trained with the default `mtry` parameter value.")
  put_end_date(start)
  # Time difference of the last iteration 19.8342 mins
  
  put_log("Saving the `RF MCC` model trained with the default `mtry` parameter value to the backup file...")
  saveRDS(fit_rf.mtry_default,
          file = fit_rf.mtry_default.backup.path)
  put_log("The `RF MCC` model trained with the default `mtry` parameter value 
has been saved to the following backup file:
%1", fit_rf.mtry_default.backup.path)
  put_end_date(start)
}

put_log("Results summary of tuning the model for the default value of parameter `mtry`, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset 
and tested on the 10% sample from the remaining 
90% data of the `Train Set`:
%1", capture.output(summary(fit_rf.mtry_default)))
put_end_date(start)
# Time difference of 6.260901 hours

plot(fit_rf.mtry_default)

put_log("Prediction accuracy of the 'RF' MCC Model trained with the default value 
of the `mtry` parameter is as follows:
%1", mean(fit_rf.mtry_default$test$predicted == y0.9.test))
# [1] 0.8397469

rf_conf.mx <- confusion_matrix(as.character(y0.9.test),
                               as.character(fit_rf.mtry_default$test$predicted))
str(rf_conf.mx)

plot_confusion_matrix(rf_conf.mx,
                      palette = "Greens",
                      font_counts = font(size = 3.5,
                                         
                                         color = "red"),
                      add_normalized = FALSE,
                      add_col_percentages = FALSE,
                      add_row_percentages = FALSE)

##### Close Log ----------------------------------------------------------------
log_close()

### Open log: Tuning `RF MCC` model with `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) & ntree = 400 ----
open_logfile(".x0.1.train.fit_rf.tune_mtry")
##### Tune `RF MCC` model with `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) & ntree = 400 ----
fit_rf.tuned_mtry.backup.path <- file.path(models.rf.tune.path, 
                                             "fit_rf.tuned_mtry.ntree500.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.tuned_mtry.backup.path)) {
  put_log("Loading the `RF MCC` model tuned by `mtry` parameter values from the backup file...")
  
  fit_rf.tuned_mtry <- readRDS(fit_rf.tuned_mtry)
  
  put_Log("The `RF MCC` model, tuned `mtry` parameter values, has been loaded from the following backup file:
%1", fit_rf.tuned_mtry.backup.path)
  put_end_date(start)
} else {
  put_log("Tuning the `RF MCC` model by `mtry` parameter values...")
  
  #> Since p = n.img_cols * n.img_rows = n.img_cols^2 = 28^2
  #> sqrt(p) = n.img_cols = 28

  mtry.values <- seq(n.img_cols/2, 2*n.img_cols) # 14:56
  start <- put_start_date()
  set.seed(N.classes)
  fit_rf.mtry_tuned <- train(x0.1.train, 
                             y0.1.train,
                             method = "rf",
                             ntree = 400,
                             tuneGrid = data.frame(mtry = mtry.values))
  
  
  put_log("The `RF MCC` model has been tuned by `mtry` parameter values.")
  put_end_date(start)
  # Time difference of the last iteration 19.8342 mins
  
  put_log("Saving the `RF MCC` model trained with the default `mtry` parameter value to the backup file...")
  saveRDS(fit_rf.tuned_mtry,
          file = fit_rf.tuned_mtry.backup.path)
  put_log("The `RF MCC` model trained with the default `mtry` parameter value 
has been saved to the following backup file:
%1", fit_rf.tuned_mtry.backup.path)
  put_end_date(start)
}

put_log("Below are results of tuning the model by `mtry` parameter values, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset:
%1", capture.output(fit_rf.mtry_tuned))
put_end_date(start)
# Time difference of 17.51424 mins

confusionMatrix(fit_rf.mtry_tuned)
ggplot(fit_rf.mtry_tuned)

# plot(mtry14_56, fit_rf.mtry14_56.accuracy)
# 
# max.idx <- which.max(fit_rf.mtry14_56.accuracy)
# 
# max_accuracy <- max(fit_rf.mtry14_56.accuracy)
# max_accuracy
# # [1] 0.88136
# 
# best_mtry <- mtry14_56[[max.idx]]
# best_mtry
# [1] 18

##### Close Log ------------------------------------------------------------------
log_close()
