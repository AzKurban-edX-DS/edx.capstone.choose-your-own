#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA MCC Model: Re-training with the Best `k` Parameter 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Disable the elapsed time limit for expressions
#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 

# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

## Setup -----------------------------------------------------------------------

open_logfile(".re-train-model.k5(best)nn+pca")

stopifnot(file.exists(my_emnist.split.file_path,
                      k1_8nn_pca.model.backup.path),
          exists("k_best.nn_pca.model.backup.path"))

## Prepare Train Datasets ------------------------------------------------------


start <- put_start_date()

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.split.file_path)
str(ds)

x_train <- ds$train$x

put_log("The Train set is balanced with respect to the set of classes:
%1", capture.output(print(ds$train$class_groups$groupByClass, n = N.classes)))
{
  # # A tibble: 39 × 2
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
  invisible(NULL)
}

y_train <- ds$train$class_groups$classID

stopifnot(sum(as.character(y_train) != rownames(x_train)) == 0)
stopifnot(nrow(x_train) == length(y_train))

rm(ds)

## Re-Train kNN+PCA Model with the best *k% Parameter on the full Dataset ------
# (The training takes about half an hour)

put_log("Training Model `kNN+PCA` on the 80% size Training Set..." )

start <- put_start_date()

put_log("Loading pre-trained `kNN+PCA MCC` Model 
(tuned for `k` values ranged from 1 to 8) from the backup file...")

k1_8nn_pca.model <- readRDS(k1_8nn_pca.model.backup.path)
put_end_date(start)
# Time difference of 

put_log("The pre-trained Model has been loaded from the following file:
%1", k1_8nn_pca.model.backup.path)

k1_8nn.best <- k1_8nn_pca.model$results$k[acc.max.idx]

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

#flush.console()
k_best.nn_pca.model <- caret::train(x_train, 
                                    y_train, 
                                    method = "knn", 
                                    preProcess = "pca",
                                    trControl = trainControl("cv", 
                                                             number = 5, 
                                                             p = 0.95,
                                                             preProcOptions = list(thresh = 0.9),
                                                             verboseIter = TRUE),
                                    tuneGrid = data.frame(k = k1_8nn.best)) # *k* = 5
stopCluster(cl)
stopImplicitCluster()
rm(cl)

# Aggregating results
# Fitting final model on full training set
# Warning: The following pre-processing methods were eliminated: 'pca', 'center', 'scale'

put_end_date(start)
# Sun Jul 5 08:18:14 2026 
# Time difference of 1.875516 hours

put_log("The Model `kNN+PCA` has been trained on the 80% size Training Set")

put_log("Saving `kNN+PCAM`odel in the backup file: `...")

saveRDS(k_best.nn_pca.model, 
        file = k_best.nn_pca.model.backup.path)
put_end_date(start)
# Time difference of 1.880165 hours

put_log("The Model `kNN+PCA` trained on the 80% size Training Set has been cached in file:
`%1`", k_best.nn_pca.model.backup.path)

## Finalizing ------------------------------------------------------------------

rm(x_train,
   y_train)

log_close()
# =========================================================================
# Log End Time: 2026-09-02 10:56:18.180704
# Log Elapsed Time: 0 01:51:43
# =========================================================================

