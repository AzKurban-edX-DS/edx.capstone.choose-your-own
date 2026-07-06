#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA Multiclass Classifier (MCC) Model 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 
#> Random Forest (RF) Multiclass Classifier Models

# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

## Creating & Tuning k1_8nn+PCA using `caret` package ------------------------

open_logfile(".pre-train-model.k1-8nn+pca")

stopifnot(exists("x0.1.train.flatten"),
          exists("y0.1.train.flatten"),
          exists("x0.9.test.flatten"),
          exists("y0.9.test.flatten"),
          exists("x.train.flatten"),
          exists("y.train.flatten"),
          exists("x.test.flatten"),
          exists("y.test.flatten"))

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path))
  dir.create(knn_pca.path)


k.values <- seq_len(8)

k1_8nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k1-8nn+pca(0.1train-set).rds")

if (file.exists(k1_8nn_pca.model.backup.path)) {
  put_log("Loading pre-trained `kNN+PCA MCC` Model 
(tuned for `k` values ranged from 1 to 8) from the backup file...")
  
  k1_8nn_pca.model <- readRDS(k1_8nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The pre-trained Model has been loaded from the following file:
%1", k1_8nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 10% size Train Set..." )
  
  start <- put_start_date()
  #flush.console()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
# The model will be tuned by *k* parameter ranging from 1 to 8 on 10% size sample of the Train Set.

# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca

  k1_8nn_pca.model <- caret::train(x0.1.train.flatten, y0.1.train.flatten, method = "knn", 
                                preProcess = "pca",
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9),
                                                         verboseIter = TRUE),
                                tuneGrid = data.frame(k = k.values))
  stopCluster(cl)
  stopImplicitCluster()
  
  # Aggregating results
  # Selecting tuning parameters
  # Fitting k = 5 on full training set
  # Warning in pre_process_options(method, column_types) :
  #   The following pre-processing methods were eliminated: 'pca', 'center', 'scale'  

    put_end_date(start)
  # Time difference of 27.84693 mins
  
  put_log("The Model `kNN+PCA` has been trained on the 10% size Train Set")

  put_log("Saving the pre-trained model in the backup file...")

    saveRDS(k1_8nn_pca.model, 
          file = k1_8nn_pca.model.backup.path)
  
  put_log("The Model `kNN+PCA` pre-trained on the 10% size Train Set 
for *k* values ranged from 1 to 8 has been backed up in the following file:
`%1`", k1_8nn_pca.model.backup.path)

  put_end_date(start)
  # Time difference of 27.88649 mins
}

### The Tuning Results Visualization & Analysis -------------------------------

put_log("The pre-trained `kNN+PCA MCC` Model tuned result:
%1", capture.output(k1_8nn_pca.model))

# k-Nearest Neighbors 

# 16575 samples
# 784 predictor
# 39 classes: '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
# 
# Pre-processing: ignore (784) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 13260, 13260, 13260, 13260, 13260 
# Resampling results across tuning parameters:
#   
#   k  Accuracy   Kappa    
# 1  0.7838914  0.7782043
# 2  0.7677828  0.7616718
# 3  0.7923379  0.7868731
# 4  0.7930015  0.7875542
# 5  0.7955958  0.7902167
# 6  0.7931222  0.7876780
# 7  0.7937255  0.7882972
# 8  0.7924585  0.7869969
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 5.


# The Model tuning visualzation:
trellis.par.set(caretTheme())
plot(k1_8nn_pca.model, 
     main = "`kNN+PCA` Multiclass Classifier Model Tuning Results")

acc.max.idx <- which.max(k1_8nn_pca.model$results$Accuracy)
acc.max.idx
# 5

k1_8nn_pca.max_accuracy <- k1_8nn_pca.model$results$Accuracy[acc.max.idx]
k1_8nn_pca.max_accuracy
# 0.7955958

k1_8nn.best <- k1_8nn_pca.model$results$k[acc.max.idx]
k1_8nn.best
# 5
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

## Re-Training kNN+PCA Model with the best *k% Parameter on the full Dataset ---------
# (The training takes about half an hour)
open_logfile(".pre-train-model.k1-8nn+pca")

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
  k_best.nn_pca.model <- caret::train(x.train.flatten, 
                                      y.train.flatten, 
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
  # Aggregating results
  # Fitting final model on full training set
  # Warning: The following pre-processing methods were eliminated: 'pca', 'center', 'scale'
  
  put_end_date(start)
  # Sun Jul 5 08:18:14 2026 
  # Time difference of 1.875516 hours
 
  put_log("The Model `kNN+PCA` has been trained on the 80% size Train Set")
  
  put_log("Saving `kNN+PCAM`odel in the backup file: `...")
  
  saveRDS(k_best.nn_pca.model, 
          file = k_best.nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 1.880165 hours
  
  put_log("The Model `kNN+PCA` trained on the 80% size Train Set has been cached in file:
`%1`", k_best.nn_pca.model.backup.path)
  
}


log_close()
# Log Elapsed Time for training & tuning `kNN+PCA`: 03:21:28

## Constructing Predictions on kNN+PCA (for best *k* Parameter value) --------
open_logfile(".x.test.flatten.predict.k(best)nn+pca")

knn_pca.best.preds.backup <-
  file.path(knn_pca.path, "k_best.nn_pca.probs.rds")

start <- put_start_date()
# Thu Apr 9 09:14:47 2026

if (file.exists(knn_pca.best.preds.backup)) {
  put_log("Loading Predicted Data of the Fine-Tuned kNN+PCA Model...") 
  
  k_best.nn_pca.probs <- readRDS(knn_pca.best.preds.backup)
  put_end_date(start)
  # Time difference of 

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

  k_best.nn_pca.probs <- caret::predict.train(k_best.nn_pca.model, 
                                              newdata = x.test.flatten,
                                              type = "prob",
                                              verbose = TRUE)
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  # Time difference of 2.354987 hours
  
  put_log("Saving the Tuned `kNN+PCA MCC` Model predicted data...")
  #> [1] 0.8693882
  
  saveRDS(k_best.nn_pca.probs,
          file = knn_pca.best.preds.backup)
  
  put_log("The predicted data of the Tuned `kNN+PCA MCC` Model has been saved to the following file:
%1", knn_pca.best.preds.backup)
  
}

k_best.nn_pca.predicted <- predicted_probs2classes(as.matrix(k_best.nn_pca.probs),
                                                   Y.Labels)
put_end_date(start)

# put_log("Fine-tuned `kNN+PCA MCC` Model: The per-class ROC curve calculation has been completed.")
# 
# put_log("Fine-tuned `kNN+PCA MCC` Model: Creating a Confusion Matrix...")
# k_best.nn_pca.conf.mx <- confusion_matrix(as.character(y.test.flatten),
#                                           as.character(k_best.nn_pca.predicted))
# put_log("Fine-tuned `kNN+PCA MCC` Model: The Confusion Matrix has been created:
# %1", capture.output(k_best.nn_pca.conf.mx))
# put_end_date(start)
# 
# put_log("The (Best *k*) `kNN+PCA MCC` Model: Generating predictions have been completed on `x.test.flatten` dataset.")
# 
# put_log("Validating accuracy of the (Best *k*) `kNN+PCA MCC` Model predictions 
# made on the `x.test.flatten` dataset...")
# knn_pca.best.accuracy0 <- mean(k_best.nn_pca.model.predicted == y.test.flatten)

knn_pca.best.accuracy <- mean(k_best.nn_pca.predicted == y.test.flatten)

put_log("Accuracy of the predicted data for the `kNN+PCA MCC` Model tuned by *k* parameter:
%1", knn_pca.best.accuracy)
#> [1] 0.862555675935958

## Visualization & Analysis ----------------------------------------------------

# put_log("Fine-tuned `kNN+PCA MCC` Model: The per-class ROC curve calculation has been completed.")
# 
# put_log("Fine-tuned `kNN+PCA MCC` Model: Creating a Confusion Matrix...")
# k_best.nn_pca.conf.mx <- confusion_matrix(as.character(y.test.flatten),
#                                           as.character(k_best.nn_pca.predicted))
# put_log("Fine-tuned `kNN+PCA MCC` Model: The Confusion Matrix has been created:
# %1", capture.output(k_best.nn_pca.conf.mx))
# put_end_date(start)
# 
# put_log("The (Best *k*) `kNN+PCA MCC` Model: Generating predictions have been completed on `x.test.flatten` dataset.")
# 
# put_log("Validating accuracy of the (Best *k*) `kNN+PCA MCC` Model predictions 
# made on the `x.test.flatten` dataset...")
# knn_pca.best.accuracy0 <- mean(k_best.nn_pca.model.predicted == y.test.flatten)

# ROC curves
# k_best.nn_pca.roc_curves <- roc_curves

# knn_pca.best.accuracy.by_class <- MCClassifier.accuracy.by_class(Y.Labels,
#                                                                  y.test.flatten,
#                                                                  k_best.nn_pca.predicted)
# knn_pca.best.accuracy.by_class
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

# put_log("`kNN+PCA MCC` Model: Plotting bar chart of per-class accuracy...")
# plot_bars.accuracy.by_class(Y.Labels,
#                             knn_pca.best.accuracy.by_class,
#                             title.prefix = "kNN+PCA-based Multiclass")
put_end_date(start)
# Time difference of  hours


log_close()

