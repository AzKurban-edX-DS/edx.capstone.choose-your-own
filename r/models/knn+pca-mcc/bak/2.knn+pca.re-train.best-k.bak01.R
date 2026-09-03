#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA Multiclass Classifier (MCC) Model 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Disable the elapsed time limit for expressions
#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 

# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

## Prepare Input Datasets ------------------------------------------------------

stopifnot(file.exists(my_emnist.split.file_path))

open_logfile(".split.80%train.balanced_subset")
start <- put_start_date()

### Loading Split Flattened Dataset allocated 10% for the Training Set ------------

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

x_test <- ds$test$x

put_log("The Test set is balanced with respect to the set of classes:
%1", capture.output(print(ds$test$class_groups$groupByClass, n = N.classes)))
{
  # # A tibble: 39 × 2
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
  invisible(NULL)
}

y_test <- ds$test$class_groups$classID

stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0)
stopifnot(nrow(x_test) == length(y_test))

rm(ds)
log_close()




## Re-Train kNN+PCA Model with the best *k% Parameter on the full Dataset ---------
# (The training takes about half an hour)
open_logfile(".re-train-model.k5(best)nn+pca")

k_best.nn_pca.model.backup.path <-
  file.path(knn_pca.data.dir, "k_best.nn+pca.rds")

if (file.exists(k_best.nn_pca.model.backup.path)) {
  put_log("Loading the `kNN+PCA MCC` Model (trained for the best `k` value) from the backup file...")
  
  k_best.nn_pca.model <- readRDS(k_best.nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The `kNN+PCA MCC` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 80% size Training Set..." )
  
  start <- put_start_date()

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
}

rm(x_train,
   y_train)

log_close()
# Log Elapsed Time: 0 01:51:26

## Constructing Predictions on kNN+PCA (for best *k* Parameter value) --------
open_logfile(".x.test.flatten.predict.k(best)nn+pca")

knn_pca.best.preds.backup <-
  file.path(knn_pca.data.dir, "k_best.nn_pca.probs.rds")


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
                                              newdata = x_test,
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

  rm(x_test,
     k_best.nn_pca.model)



k_best.nn_pca.predicted <- predicted_probs2classes(as.matrix(k_best.nn_pca.probs),
                                                   Y.Labels)
put_end_date(start)

knn_pca.best.accuracy <- mean(k_best.nn_pca.predicted == y_test)

put_log("Accuracy of the predicted data for the `kNN+PCA MCC` Model tuned by *k* parameter:
%1", knn_pca.best.accuracy)
#> [1] 0.862555675935958

log_close()
# Log Elapsed Time: 0 02:15:36

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path))

knn_pca.eval.conf.mx.img_file <- file.path(knn_pca.data.plots.dat.dir,
                                           "knn+pca-tuned.eval.confusion-matrix.png")

model.eval.plots_dat.file <- file.path(knn_pca.data.plots.dat.dir,
                            "knn+pca-tuned.eval.plots_dat.rds")

#' (Re-)Create the `plots.args` object in the `GlobalEnv` environment 
#' of the `modelEvalResultPlotArgs` class, containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
init.plots_args(targets = y_test,
                predicted.probabilities = k_best.nn_pca.probs,
                predicted.values = k_best.nn_pca.predicted,
                alg_name = "kNN+PCA",
                plots_dat.file = model.eval.plots_dat.file,
                cm.export.img_file = knn_pca.eval.conf.mx.img_file,
                cm.print.image = T)

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()
