#%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA MCC & RF Models 
#%%%%%%%%%%%%%%%%%%%%%%%%%

#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 
#> Random Forest (RF) Multiclass Classifier Models

## Loading Split Dataset allocated 10% for the Train Set ---------------------
open_logfile(".split.10%train.balanced_subset")
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

log_close()

## `kNN+PCA MCC` Model Tuning --------------------------------------------------
# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

### `kNN+PCA MCC` Model Initial Paths -----------------------------------------------------
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

open_logfile(".pre-train-model.k1-7nn+pca")

#### Tuning k1_7NN+PCA model by *k* parameter ranging from 1 to 7 on 10% size Train Set ----
# (The training takes about half an hour)
k.values <- seq_len(8)

k1_7nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k1-7nn+pca(0.1train-set).rds")

if (file.exists(k1_7nn_pca.model.backup.path)) {
  put_log("Loading pre-trained `kNN+PCA MCC` Model 
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
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  k1_7nn_pca.model <- caret::train(x0.1.train, y0.1.train, method = "knn", 
                                preProcess = "pca",
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9),
                                                         verboseIter = TRUE,
                                                         verbose = TRUE),
                                tuneGrid = data.frame(k = k.values))
  stopCluster(cl)
  stopImplicitCluster()
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

#### The Tuning Results Visualization & Analysis -------------------------------

put_log("The pre-trained `kNN+PCA MCC` Model trained result:
%1", capture.output(k1_7nn_pca.model))

# The Model tuning visualzation:
trellis.par.set(caretTheme())
plot(k1_7nn_pca.model, 
     main = "`kNN+PCA` Multiclass Classifier Model Tuning Results")

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

log_close()

## Loading Split Dataset allocated 20% for the Test set (default) ------------
open_logfile(".split.20%test.balanced_subset")

start <- put_start_date()
ds_flatten <- load_flatten_datasets("ds_flatten.split_list", 
                                 my_emnist.split.file_path)
x.train <- ds_flatten$x.train
x.test <- ds_flatten$x.test
x.test.files <- ds_flatten$x.files

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

log_close()

## Training kNN+PCA Model for best *k% Parameter -----------------------------
# (The training takes about half an hour)
open_logfile(".pre-train-model.k1-7nn+pca")

k_best.nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k_best.nn+pca.rds")

if (file.exists(k_best.nn_pca.model.backup.path)) {
  put_log("Loading the `kNN+PCA MCC` Model (trained for the best `k` value) from the backup file...")
  
  k_best.nn_pca.model <- readRDS(k_best.nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The `kNN+PCA MCC` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 80% size Train Set..." )
  
  start <- put_start_date()
  
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  #flush.console()
  k_best.nn_pca.model <- caret::train(x.train, 
                                      y.train, 
                                      method = "knn", 
                                      preProcess = "pca",
                                      trControl = trainControl("cv", 
                                                               number = 5, 
                                                               p = 0.95,
                                                               preProcOptions = list(thresh = 0.9),
                                                               verboseIter = TRUE,
                                                               verbose = TRUE),
                                      tuneGrid = data.frame(k = k.best)) # *k* = 6
  stopCluster(cl)
  stopImplicitCluster()
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


log_close()
# Log Elapsed Time for training & tuning `kNN+PCA`: 03:21:28

##### Constructing Predictions on kNN+PCA (for best *k* Parameter value) ------
open_logfile(".x.test.predict.k(best)nn+pca")

knn_pca.best.preds.backup0 <-
  file.path(knn_pca.path, "x.test.k(best)NN+PCA.predictions0.rds")

knn_pca.best.preds.backup <-
  file.path(knn_pca.path, "x.test.k(best)NN+PCA.predictions.rds")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

if (file.exists(knn_pca.best.preds.backup)) {
  put_log("Loading Predicted Data of the Fine-Tuned kNN+PCA Model...") 
  
  knn_pca.best.preds <- readRDS(knn_pca.best.preds.backup)
  k_best.nn_pca.predicted <- knn_pca.best.preds$predicted
  k_best.nn_pca.probs <- knn_pca.best.preds$probs
  k_best.nn_pca.conf.mx <- knn_pca.best.preds$confusion.mx
  k_best.nn_pca.roc_curves <- knn_pca.best.preds$roc.curves
  k_best <- knn_pca.best.preds$k_best
  
  rm(knn_pca.best.preds)
  put_end_date(start)
  # Time difference of 

#   if (file.exists(knn_pca.best.preds.backup0)) {
#     put_log("Loading Predicted Data from cache file: 
# %1...", knn_pca.best.preds.backup0)
#     
#     knn_pca.best.preds <- readRDS(knn_pca.best.preds.backup0)
#     k_best.nn_pca.model.predicted <- knn_pca.best.preds$predicted
#     rm(knn_pca.best.preds)
#     put_end_date(start)
#     # Time difference of 
#   }
  
  put_log("The Predicted Data of the Fine-Tuned kNN+PCA Model has been loaded from the following file:
%1...", knn_pca.best.preds.backup)
} else {
  put_log("Constructing predictions using the `kNN+PCA MCC` Model trained for the best *k* value...")
  
  if(!exists("k_best.nn_pca.model")) {
    stopifnot(file.exists(k_best.nn_pca.model.backup.path))
    
    put_log("Loading the `kNN+PCA MCC` Model (trained for the best `k` value) from the backup file...")
    
    k_best.nn_pca.model -> readRDS(k_best.nn_pca.model.backup.path)
    put_end_date(start)
    # Time difference of 
    
    put_log("The `kNN+PCA MCC` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)
  }
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  
  set.seed(nrow(x.test))
  
  # k_best.nn_pca.model.predicted <- stats::predict(k_best.nn_pca.model,
  #                                                 x.test,
  #                                                 type = "raw")

  k_best.nn_pca.probs <- predict.train(k_best.nn_pca.model, 
                                       newdata = x.test,
                                       type = "prob",
                                       verbose = TRUE)
  
  k_best.nn_pca.predicted <- predicted_probs2classes(as.matrix(k_best.nn_pca.probs),
                                                  y.labels)
  
  #### ROC Curves
  # References:
  # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
  # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
  
  put_log("Fine-tuned `kNN+PCA MCC` Model: Calculating a ROC curve for each class...")
  k_best.nn_pca.roc_curves <- calc.roc_curves(y.test,
                                              k_best.nn_pca.probs,
                                              y.labels)
  
  put_log("Fine-tuned `kNN+PCA MCC` Model: The per-class ROC curve calculation has been completed.")
  
  put_log("Fine-tuned `kNN+PCA MCC` Model: Creating a Confusion Matrix...")
  k_best.nn_pca.conf.mx <- confusion_matrix(as.character(y.test),
                                                as.character(k_best.nn_pca.predicted))
  put_log("Fine-tuned `kNN+PCA MCC` Model: The Confusion Matrix has been created:
%1", capture.output(k_best.nn_pca.conf.mx))
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  # Time difference of 2.25995 hours
  
  # sum(k_best.nn_pca.model.predicted != k_best.nn_pca.predicted)
  # diff.idx <- which(k_best.nn_pca.model.predicted != k_best.nn_pca.predicted)
  # k_best.nn_pca.probs[diff.idx,]

  put_log("The (Best *k*) `kNN+PCA MCC` Model: Generating predictions have been completed on `x.test` dataset.")
  
  put_log("Saving the Fine-tuned `kNN+PCA MCC` Model data...")
  #> [1] 0.8693882
  
  # saveRDS(list(predicted = k_best.nn_pca.model.predicted,
  #              probs = k_best.nn_pca.probs,
  #              accuracy = knn_pca.best.accuracy,
  #              k_best = k.best),
  #      file = knn_pca.best.preds.backup0)
  
  saveRDS(list(predicted = k_best.nn_pca.predicted,
               probs = k_best.nn_pca.probs,
               confusion.mx = k_best.nn_pca.conf.mx,
               roc.curves = k_best.nn_pca.roc_curves,
               k_best = k.best),
       file = knn_pca.best.preds.backup)
  
  put_log("The data of the fine-tuned `kNN+PCA MCC` Model has been saved to the following file:
%1", knn_pca.best.preds.backup)
}

put_log("Validating accuracy of the (Best *k*) `kNN+PCA MCC` Model predictions 
made on the `x.test` dataset...")
# knn_pca.best.accuracy0 <- mean(k_best.nn_pca.model.predicted == y.test)
knn_pca.best.accuracy <- mean(k_best.nn_pca.predicted == y.test)

put_log("Accuracy of the predicted data for the `kNN+PCA MCC` Model tuned by *k* parameter:
%1", knn_pca.best.accuracy)
#> [1] 0.860810160105935

plot(k_best.nn_pca.roc_curves[[1]], 
     main = "ROC Curves for the Fine-tuned `kNN+PCA MCC` Model by the `k` Parameter")
for (class.idx in 2:N.classes) {
  lines(k_best.nn_pca.roc_curves[[class.idx]], col = class.idx)
}

# cl <- makeCluster(N_pcCores)
# registerDoParallel(cl)
# 
# dev.off()
# plot_confusion_matrix(k_best.nn_pca.conf.mx,
#                       palette = "Greens",
#                       font_counts = font(size = 3,
#                                          color = "red"),
#                       add_normalized = FALSE,
#                       add_col_percentages = FALSE,
#                       add_row_percentages = FALSE)
# 
# stopCluster(cl)
# stopImplicitCluster()

knn_pca.best.accuracy.by_class <- MCClassifier.accuracy.by_class(y.labels,
                                                                 y.test,
                                                                 k_best.nn_pca.predicted)
knn_pca.best.accuracy.by_class
{
  
#' class  accuracy
#'     # 1.0000000
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 1.0000000
#'     0 0.9753521
#'     1 0.7159624
#'     2 0.8673709
#'     3 0.9553991
#'     4 0.8955399
#'     5 0.8403756
#'     6 0.9096244
#'     7 0.9647887
#'     8 0.8497653
#'     9 0.9002347
#'     A 0.8169014
#'     B 0.8227700
#'     C 0.9436620
#'     D 0.8697183
#'     E 0.8744131
#'     F 0.8697183
#'     G 0.5551643
#'     H 0.8943662
#'     I 0.6549296
#'     J 0.9178404
#'     K 0.8650235
#'     L 0.4730047
#'     M 0.9483568
#'     N 0.9178404
#'     P 0.9366197
#'     Q 0.4929577
#'     R 0.8591549
#'     S 0.8556338
#'     T 0.8720657
#'     U 0.9154930
#'     V 0.9178404
#'     W 0.9330986
#'     X 0.8849765
#'     Y 0.7570423
#'     Z 0.8755869  
}

put_log("`kNN+PCA MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(y.labels,
                            knn_pca.best.accuracy.by_class,
                            title.prefix = "kNN+PCA-based Multiclass")
put_end_date(start)
# Time difference of  hours


log_close()

## Random Forest MCC Model -----------------------------------------------------
# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

### `Random Forest (RF) MCC` Model Initial Paths -------------------------------

models.random_forest.path <- file.path(models.path, "random-forest")

if(!dir.exists(models.random_forest.path))
  dir.create(models.random_forest.path)

models.rf.tune.path = file.path(models.random_forest.path, "tune")

if(!dir.exists(models.rf.tune.path))
  dir.create(models.rf.tune.path)

### Train `RF MCC` model with the default mtry value & ntree = 500 ---------------------------
open_logfile("x0.1.train.fit_rf.mtry_default.ntree500")

fit_rf.mtry_default.backup.path <- file.path(models.rf.tune.path, 
                             "fit_rf.mtry_default.ntree500.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry_default.backup.path)) {
  put_log("Loading data of the `RF MCC` model, 
trained with the default `mtry` parameter value, from the backup file...")
  
  fit.set <- readRDS(fit_rf.mtry_default.backup.path)
  fit_rf.mtry_default <- fit.set$fit
  rf_conf.mx.mtry_default <- fit.set$confusion.mx
  rm(fit.set)
  
  put_log("The data of the `RF MCC` Model, trained with the default `mtry` parameter value, 
has been loaded from the following backup file:
%1", fit_rf.mtry_default.backup.path)
  put_end_date(start)
} else {
  put_log("Training the `RF MCC` model with the default `mtry` parameter value...")
  set.seed(N.classes)

  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  fit_rf.mtry_default <- randomForest(x0.1.train, 
                                      y0.1.train,
                                      x0.9.test,
                                      y0.9.test,
                                      keep.forest = TRUE,
                                      ntree = 500)
  
  put_log("The `RF MCC` Model has been trained with the default `mtry` parameter value.")
  put_end_date(start)
  
  put_log("`RF MCC` Model pre-trained with the default `mtry` parameter value: Creating a Confusion Matrix...")
  rf_conf.mx.mtry_default <- confusion_matrix(as.character(y0.9.test),
                                              as.character(fit_rf.mtry_default$test$predicted))
  put_log("`RF MCC` Model pre-trained with the default `mtry` parameter value: 
The Confusion Matrix has been created:
%1", capture.output(rf_conf.mx.mtry_default))
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()
  
  # Time difference of the last iteration 19.8342 mins
  
  put_log("Saving the pre-trained `RF MCC` Model data...")
  saveRDS(list(fit = fit_rf.mtry_default,
               confusion.mx = rf_conf.mx.mtry_default),
          file = fit_rf.mtry_default.backup.path)
  put_log("The data of the pre-trained `RF MCC` Model (with the default `mtry` parameter value) 
has been saved to the following file:
%1", fit_rf.mtry_default.backup.path)
  put_end_date(start)
# Time difference of  mins
}

put_log("The results of pre-training the `RF MCC` Model 
(with the default `mtry` parameter value) on a 10% sample of the`Train Set` dataset 
and testing on the remaining 90% of the `Train Set` are as follows:
%1", capture.output(fit_rf.mtry_default))
put_end_date(start)
# Time difference of 6.260901 hours

plot(fit_rf.mtry_default, 
     main = "`RF MCC` Model Pre-trained with the Default `mtry` Parameter Value")

# cl <- makeCluster(N_pcCores)
# registerDoParallel(cl)
# 
# dev.off()
# plot_confusion_matrix(rf_conf.mx.mtry_default,
#                       plot_title = caption,
#                       palette = "Greens",
#                       font_counts = font(size = 3,
#                                          color = "red"),
#                       add_normalized = FALSE,
#                       add_col_percentages = FALSE,
#                       add_row_percentages = FALSE)
# stopCluster(cl)
# stopImplicitCluster()

put_log("Prediction accuracy of the `RF MCC` Model,
pre-trained with the default `mtry` parameter value, is as follows:
%1", mean(fit_rf.mtry_default$test$predicted == y0.9.test))
# [1] 0.8397469

fit_rf.mtry_default.accuracy.by_class <- 
  MCClassifier.accuracy.by_class(y.labels,
                                 y0.9.test,
                                 fit_rf.mtry_default$test$predicted)

put_log("The per-class prediction accuracy of the `RF MCC` Model, pre-trained 
with the default `mtry` parameter value, is as follows:
%1",capture.output(fit_rf.mtry_default.accuracy.by_class))
{
  
#' class  accuracy
#'     # 1.0000000
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 0.9994784
#'     0 0.9530516
#'     1 0.6966615
#'     2 0.7509129
#'     3 0.9259259
#'     4 0.8495044
#'     5 0.7996870
#'     6 0.8928013
#'     7 0.9405321
#'     8 0.8012520
#'     9 0.9183620
#'     A 0.7712572
#'     B 0.8252478
#'     C 0.8969744
#'     D 0.7793427
#'     E 0.8724570
#'     F 0.8557642
#'     G 0.4718310
#'     H 0.8510694
#'     I 0.6165884
#'     J 0.8763693
#'     K 0.8638498
#'     L 0.4783516
#'     M 0.9381847
#'     N 0.8774126
#'     P 0.9170579
#'     Q 0.5670318
#'     R 0.8341158
#'     S 0.8281168
#'     T 0.8286385
#'     U 0.8823683
#'     V 0.8839332
#'     W 0.9431403
#'     X 0.8821075
#'     Y 0.8004695
#'     Z 0.8802817
    }

put_log("`RF MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(y.labels,
                            fit_rf.mtry_default.accuracy.by_class,
                            title.prefix = "Random Forest-based (default `mtry`) Multiclass")
log_close()
### Tune `RF MCC` model with `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) & ntree = 200 ----
#### Step 1. Coarse Tuning: `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) by step 6 ----
open_logfile(".x0.1.train.fit_rf.tune_mtry")

fit_rf.mtry_tuned.backup.path <- file.path(models.rf.tune.path, 
                                             "fit_rf.mtry-coarse_tuned.ntree200.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry_tuned.backup.path)) {
  put_log("Loading the `RF MCC` model tuned by `mtry` parameter values from the backup file...")
  
  fit.bak <- readRDS(fit_rf.mtry_tuned.backup.path)
  fit_rf.mtry_tuned <- fit.bak$fit
  mtry.tune_values <- fit.bak$mtry
  rm(fit.bak)
  
  put_log("The `RF MCC` model, tuned `mtry` parameter values, has been loaded from the following backup file:
%1", fit_rf.mtry_tuned.backup.path)
  put_end_date(start)
} else {
  put_log("Tuning the `RF MCC` model by `mtry` parameter values...")
  
  #> Since p = n.img_cols * n.img_rows = n.img_cols^2 = 28^2
  #> sqrt(p) = n.img_cols = 28

  mtry.tune_values <- seq(n.img_cols/2, 2*n.img_cols, 6) # 14:56
  start <- put_start_date()

  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  set.seed(N.classes)
  fit_rf.mtry_tuned <- train(x0.1.train, 
                             y0.1.train,
                             method = "rf",
                             ntree = 200,
                             trControl = trainControl(
                               method = "cv",          # K-fold cross-validation
                               number = 5,             # 5 folds
                               verboseIter = TRUE      # <--- This activates the progress output
                             ),
                             tuneGrid = data.frame(mtry = mtry.tune_values))
  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("The `RF MCC` model has been tuned by `mtry` parameter values.")
  put_end_date(start)
  # Time difference of 27.74778 mins
  
  put_log("Saving the `RF MCC` model trained with the default `mtry` parameter value to the backup file...")
  saveRDS(list(fit = fit_rf.mtry_tuned,
               mtry = mtry.tune_values),
          file = fit_rf.mtry_tuned.backup.path)
  put_log("The `RF MCC` model trained with the default `mtry` parameter value 
has been saved to the following backup file:
%1", fit_rf.mtry_tuned.backup.path)
  put_end_date(start)
# Time difference of 32.83442 mins

}


put_log("Below are results of tuning the model by `mtry` parameter values, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset:
%1", capture.output(fit_rf.mtry_tuned))

{
  # 16575 samples
  #   784 predictor
  #>    39 classes: 
  #>    '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
  #>    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
  #>    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
  # 
  # No pre-processing
  # Resampling: Cross-Validated (5 fold) 
  # Summary of sample sizes: 13260, 13260, 13260, 13260, 13260 
  # Resampling results across tuning parameters:
  # 
  #   mtry  Accuracy   Kappa    
  #   14    0.8244947  0.8198762
  #   20    0.8266667  0.8221053
  #   26    0.8296833  0.8252012
  #   32    0.8304072  0.8259443
  #   38    0.8303469  0.8258824
  #   44    0.8306486  0.8261920
  #   50    0.8296229  0.8251393
  #   56    0.8291403  0.8246440
  # 
  # Accuracy was used to select the optimal model using the largest value.
  # The final value used for the model was mtry = 44.
}
put_end_date(start)

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

log_close()

#### Step 2. Fine Tuning: `mtry` ranged from 38 to 50 by step 3 ----
open_logfile(".x0.1.train.fit_rf.fine-tune_mtry")

fit_rf.mtry.fine_tuned.backup.path <- file.path(models.rf.tune.path, 
                                             "fit_rf.mtry-fine_tuned.ntree200.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry.fine_tuned.backup.path)) {
  put_log("Loading the `RF MCC` model fine-tuned with `mtry` parameter values from the backup file...")
  
  fit.bak <- readRDS(fit_rf.mtry.fine_tuned.backup.path)
  fit_rf.mtry.fine_tuned <- fit.bak$fit
  mtry.fine_tune.values <- fit.bak$mtry
  rm(fit.bak)
  
  put_log("The `RF MCC` model, fine-tuned with `mtry` parameter values, 
has been loaded from the following backup file:
%1", fit_rf.mtry.fine_tuned.backup.path)
  put_end_date(start)
} else {
  put_log("Fine-Tuning the `RF MCC` model by `mtry` parameter values...")

  acc.max.idx <- which.max(fit_rf.mtry_tuned$results$Accuracy)
  mtry.fine_tune.values <- seq(mtry.tune_values[acc.max.idx-1], 
                               mtry.tune_values[acc.max.idx+1], 
                               3) # 38:50, step = 3
  start <- put_start_date()

  # Reference:
  # The code snippet below was copied from the following resource:
  # https://www.geeksforgeeks.org/machine-learning/how-to-track-progress-while-building-model-with-the-caret-package/

  # Start of copied code snippet:
  {  
    # Define the control function for cross-validation with custom functions
    custom_control <- trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      # index = createFolds(y0.1.train, k = 5),
      savePredictions = "final",
      summaryFunction = multiClassSummary,  # Use multiClassSummary for multi-class problems
      classProbs = FALSE
    )
    
    # Custom progress functions
    startFun <- function(x) {
      cat("Starting training iteration", x, "\n")
    }
    endFun <- function(x) {
      cat("Ending training iteration", x, "\n")
    }
    
    # Assign custom functions to the control object
    custom_control$start <- startFun
    custom_control$end <- endFun
  }
  # End of copied code snippet
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)

  set.seed(N.classes)
  fit_rf.mtry.fine_tuned <- train(x0.1.train, 
                                  y0.1.train,
                                  method = "rf",
                                  ntree = 200,
                                  trControl = custom_control,
                                  tuneGrid = data.frame(mtry = mtry.fine_tune.values))
  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("The `RF MCC` model has been fine-tuned by `mtry` parameter values.")
  put_end_date(start)
  # Time difference of 27.74778 mins
  
  put_log("Saving the `RF MCC` model trained with the fine-tuned `mtry` parameter values to the backup file...")
  saveRDS(list(fit = fit_rf.mtry.fine_tuned,
               mtry = mtry.fine_tune.values),
          file = fit_rf.mtry.fine_tuned.backup.path)
  put_log("The `RF MCC` model trained with the fine-tuned `mtry` parameter values 
has been saved to the following backup file:
%1", fit_rf.mtry.fine_tuned.backup.path)
  put_end_date(start)
# Time difference of 32.83442 mins

}


put_log("Below are results of tuning the model by `mtry` parameter values, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset:
%1", capture.output(fit_rf.mtry.fine_tuned$results[,1:3]))

{
  #   mtry  Accuracy     Kappa
  # 1   38 0.8296229 0.8251393
  # 2   41 0.8300452 0.8255728
  # 3   44 0.8312519 0.8268111
  # 4   47 0.8288386 0.8243344
  # 5   50 0.8302262 0.8257585
}
put_end_date(start)

confusionMatrix(fit_rf.mtry.fine_tuned)
ggplot(fit_rf.mtry.fine_tuned)

acc.fine_tuned.max <- max(fit_rf.mtry.fine_tuned$results$Accuracy)
# 0.8312519
acc.fine_tuned.max.idx <- which.max(fit_rf.mtry.fine_tuned$results$Accuracy)
# 3
mtry.fine_tuned.best <- mtry.fine_tune.values[acc.max.idx]
# 44

log_close()

#### Step 3. Final Tuning: `mtry` ranged from 42 to 49 ------------------------
open_logfile(".x0.1.train.fit_rf.fine-tune_mtry")

fit_rf.mtry.final_tuned.backup.path <- file.path(models.rf.tune.path, 
                                             "fit_rf.mtry-final_tuned.ntree200.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry.final_tuned.backup.path)) {
  put_log("Loading the `RF MCC` model final tuned with `mtry` parameter values from the backup file...")
  
  fit.bak <- readRDS(fit_rf.mtry.final_tuned.backup.path)
  fit_rf.mtry.final_tuned <- fit.bak$fit
  mtry.final_tune.values <- fit.bak$mtry
  rm(fit.bak)
  
  put_log("The `RF MCC` model, final tuned with `mtry` parameter values, 
has been loaded from the following backup file:
%1", fit_rf.mtry.final_tuned.backup.path)
  put_end_date(start)
} else {
  put_log("Final Tuning the `RF MCC` model by `mtry` parameter values...")

  mtry.seq <- seq(mtry.fine_tune.values[acc.fine_tuned.max.idx-1] + 1, 
                  mtry.fine_tune.values[length(mtry.fine_tune.values)] - 1) 
  
  mtry.final_tune.values <- mtry.seq[mtry.seq != mtry.fine_tune.values[c(acc.fine_tuned.max.idx,
                                                                         acc.fine_tuned.max.idx + 1)]]
  rm(mtry.seq)
  start <- put_start_date()

  # Reference:
  # The code snippet below was copied from the following resource:
  # https://www.geeksforgeeks.org/machine-learning/how-to-track-progress-while-building-model-with-the-caret-package/

  # Start of copied code snippet:
  {  
    # Define the control function for cross-validation with custom functions
    custom_control <- trainControl(
      method = "cv",
      number = 5,
      verboseIter = TRUE,
      # index = createFolds(y0.1.train, k = 5),
      savePredictions = "final",
      summaryFunction = multiClassSummary,  # Use multiClassSummary for multi-class problems
      classProbs = FALSE
    )
    
    # Custom progress functions
    startFun <- function(x) {
      cat("Starting training iteration", x, "\n")
    }
    endFun <- function(x) {
      cat("Ending training iteration", x, "\n")
    }
    
    # Assign custom functions to the control object
    custom_control$start <- startFun
    custom_control$end <- endFun
  }
  # End of copied code snippet
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)

  set.seed(N.classes)
  fit_rf.mtry.final_tuned <- train(x0.1.train, 
                                  y0.1.train,
                                  method = "rf",
                                  ntree = 200,
                                  trControl = custom_control,
                                  tuneGrid = data.frame(mtry = mtry.final_tune.values))
  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("The `RF MCC` model has been tuned by `mtry` parameter values.")
  put_end_date(start)
  # Time difference of 27.74778 mins
  
  put_log("Saving the `RF MCC` model trained with the default `mtry` parameter value to the backup file...")
  saveRDS(list(fit = fit_rf.mtry.final_tuned,
               mtry = mtry.final_tune.values),
          file = fit_rf.mtry.final_tuned.backup.path)
  put_log("The `RF MCC` model trained with the default `mtry` parameter value 
has been saved to the following backup file:
%1", fit_rf.mtry.final_tuned.backup.path)
  put_end_date(start)
# Time difference of 32.83442 mins

}


put_log("Below are results of tuning the model by `mtry` parameter values, 
trained using `Random Forest` method on a 10% sample of the`Train Set` dataset:
%1", capture.output(fit_rf.mtry.final_tuned$results[,1:3]))

{
}
put_end_date(start)

confusionMatrix(fit_rf.mtry.final_tuned)
ggplot(fit_rf.mtry.final_tuned)

acc.final_tuned.max <- max(fit_rf.mtry.final_tuned$results$Accuracy)
# 0.8302262
acc.final_tuned.max.idx <- which.max(fit_rf.mtry.final_tuned$results$Accuracy)
# 2
mtry.final_tuned.best <- mtry.final_tune.values[acc.max.idx]
# 45

mtry.best <- ifelse(acc.final_tuned.max > acc.fine_tuned.max, 
                    mtry.final_tuned.best,
                    mtry.fine_tuned.best)
# 44
log_close()

### Re-Train `RF MCC` model on full-scaled database with the best mtry value & ntree = 400 ---------------------------
open_logfile("x.train.fit_rf.mtry_best.ntree400")

fit_rf.mmtry_best.backup.path <- file.path(models.rf.tune.path, 
                                             "fit_rf.mtry_best.ntree400.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mmtry_best.backup.path)) {
  put_log("Loading data of the fine-tuned `RF MCC` Model by the `mtry` parameter...")
  
  fit.set <- readRDS(fit_rf.mmtry_best.backup.path)
  fit_rf.mmtry_best <- fit.set$fit
  fit_rf.mmtry_best.conf.mx <- fit.set$confusion.mx
  fit_rf.mmtry_best.roc_curves <- fit.set$roc.curves
  rm(fit.set)
  
  put_log("The data of the fine-tuned `RF MCC` Model, 
trained with the best `mtry` parameter value, has been loaded from the following backup file:
%1", fit_rf.mmtry_best.backup.path)
  put_end_date(start)
} else {
  put_log("Training the `RF MCC` model with the best `mtry` parameter value...")
  set.seed(N.classes)

  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  set.seed(nrow(x.test))
  
  fit_rf.mmtry_best <- randomForest(x.train, 
                                    y.train,
                                    x.test,
                                    y.test,
                                    mtry = mtry.best,
                                    ntree = 400)
  
  put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
  put_end_date(start)
  # Time difference of the last iteration 19.8342 mins
  
  #### ROC Curves
  # References:
  # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
  # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
  
  put_log("Fine-tuned `RF MCC` Model: Calculating a ROC curve for each class...")
  fit_rf.mmtry_best.roc_curves <- calc.roc_curves(y.test,
                                                  fit_rf.mmtry_best$test$votes,
                                                  y.labels)
 
  put_log("Fine-tuned `RF MCC` Model: The per-class ROC curve calculation has been completed.")
  
  
  put_log("Fine-tuned `RF MCC` Model: Creating a Confusion Matrix...")
  fit_rf.mmtry_best.conf.mx <- confusion_matrix(as.character(y.test),
                                                as.character(fit_rf.mmtry_best$test$predicted))
  put_log("Fine-tuned `RF MCC` Model: The Confusion Matrix has been created:
%1", capture.output(fit_rf.mmtry_best.conf.mx))
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()

  put_log("Saving the fine-tuned `RF MCC` Model data...")
  saveRDS(list(fit = fit_rf.mmtry_best,
               confusion.mx = fit_rf.mmtry_best.conf.mx,
               roc.curves = fit_rf.mmtry_best.roc_curves),
          file = fit_rf.mmtry_best.backup.path)
  put_log("The data of the fine-tuned `RF MCC` Model has been saved to the following file:
%1", fit_rf.mmtry_best.backup.path)
  put_end_date(start)
  # Time difference of  mins
}

put_log("The results of the fine-tuning `RF MCC` Model (after being trained with the best `mtry` parameter value
on an 80% sample of the`Train Set` dataset and tested on the remaining 20% of the `Train Set`) 
are as follows:
%1", capture.output(fit_rf.mmtry_best))
put_end_date(start)
# Time difference of 6.260901 hours

plot(fit_rf.mmtry_best,
     main = "Fine-tuning Results of the `RF MCC` Model by the `mtry` Parameter")

plot(fit_rf.mmtry_best.roc_curves[[1]], 
     main = "ROC Curves for the Fine-tuned `RF MCC` Model by the `mtry` Parameter")
for (class.idx in 2:N.classes) {
  lines(fit_rf.mmtry_best.roc_curves[[class.idx]], col = class.idx)
}


# cl <- makeCluster(N_pcCores)
# registerDoParallel(cl)
#
# dev.off()
# plot_confusion_matrix(fit_rf.mmtry_best.conf.mx,
#                       palette = "Greens",
#                       font_counts = font(size = 3,
#                                          color = "red"),
#                       add_normalized = FALSE,
#                       add_col_percentages = FALSE,
#                       add_row_percentages = FALSE)
# 
# stopCluster(cl)
# stopImplicitCluster()


put_log("Prediction accuracy of the fine-tuned 'RF MCC' Model, 
trained with the best `mtry` parameter value, is as follows:
%1", mean(fit_rf.mmtry_best$test$predicted == y.test))
# [1] 0.886390995545925

fit_rf.mmtry_best.accuracy.by_class <- MCClassifier.accuracy.by_class(y.labels,
                                                                 y.test,
                                                                 fit_rf.mmtry_best$test$predicted)
put_log("The per-class prediction accuracy of the fine-tuned 'RF MCC' Model, 
trained with the best `mtry` parameter value, is as follows:
%1", capture.output(fit_rf.mmtry_best.accuracy.by_class))
{
  #' class  accuracy
  #'     # 1.0000000
  #'     $ 1.0000000
  #'     & 1.0000000
  #'     @ 1.0000000
  #'     0 0.9659624
  #'     1 0.7300469
  #'     2 0.8521127
  #'     3 0.9448357
  #'     4 0.9049296
  #'     5 0.8544601
  #'     6 0.9072770
  #'     7 0.9636150
  #'     8 0.8791080
  #'     9 0.9178404
  #'     A 0.8673709
  #'     B 0.8896714
  #'     C 0.9483568
  #'     D 0.8873239
  #'     E 0.9131455
  #'     F 0.9201878
  #'     G 0.6173709
  #'     H 0.9084507
  #'     I 0.6830986
  #'     J 0.9154930
  #'     K 0.9084507
  #'     L 0.5422535
  #'     M 0.9483568
  #'     N 0.9143192
  #'     P 0.9553991
  #'     Q 0.6713615
  #'     R 0.9002347
  #'     S 0.8638498
  #'     T 0.9178404
  #'     U 0.9237089
  #'     V 0.9178404
  #'     W 0.9636150
  #'     X 0.9248826
  #'     Y 0.8392019
  #'     Z 0.9072770
  
  
}

plot_bars.accuracy.by_class(y.labels,
                            fit_rf.mmtry_best.accuracy.by_class,
                            title.prefix = "Tuned Random Forest-based Multiclass")
log_close()






