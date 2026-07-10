#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA Multiclass Classifier (MCC) Model 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Disable the elapsed time limit for expressions
#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 

# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

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

knn_pca.best.accuracy <- mean(k_best.nn_pca.predicted == y.test.flatten)

put_log("Accuracy of the predicted data for the `kNN+PCA MCC` Model tuned by *k* parameter:
%1", knn_pca.best.accuracy)
#> [1] 0.862555675935958
log_close()

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

knn_pca.plot_img.dir <- file.path(knn_pca.dir, "plot.img")

if(!dir.exists(knn_pca.plot_img.dir))
  dir.create(knn_pca.plot_img.dir)

knn_pca.eval.conf.mx.img_file <- file.path(knn_pca.plot_img.dir,
                                           "knn_pca.eval.confusion-matrix.png")

knn_pca.eval.conf.mx.obj_file <- file.path(knn_pca.plot_img.dir,
                                           "knn_pca.eval.confusion-matrix.rds")
start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

put_log("Plotting ROC curves the Model Evaluation Results...")
Cknn_pca.eval.roc_curves <- plot.ROC.curves(y.test.flatten,
                                            k_best.nn_pca.probs)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
k_best.nn_pca.acc_by_class <- plot.per_class.accuracy.bars(y.test.flatten,
                                                           k_best.nn_pca.predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(k_best.nn_pca.acc_by_class))
{
  #' class  accuracy
  #'     # 1.0000000
  #'     $ 1.0000000
  #'     & 1.0000000
  #'     @ 1.0000000
  #'     0 0.9718310
  #'     1 0.7042254
  #'     2 0.8615023
  #'     3 0.9553991
  #'     4 0.8955399
  #'     5 0.8356808
  #'     6 0.8990610
  #'     7 0.9636150
  #'     8 0.8521127
  #'     9 0.8920188
  #'     A 0.8239437
  #'     B 0.8251174
  #'     C 0.9448357
  #'     D 0.8744131
  #'     E 0.8779343
  #'     F 0.8685446
  #'     G 0.5669014
  #'     H 0.9014085
  #'     I 0.6467136
  #'     J 0.9107981
  #'     K 0.8744131
  #'     L 0.4671362
  #'     M 0.9483568
  #'     N 0.9237089
  #'     P 0.9366197
  #'     Q 0.5117371
  #'     R 0.8673709
  #'     S 0.8650235
  #'     T 0.8802817
  #'     U 0.9178404
  #'     V 0.9272300
  #'     W 0.9377934
  #'     X 0.8802817
  #'     Y 0.7570423
  #'     Z 0.8732394
  invisible(NULL)
}

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

k_best.nn_pca.conf_mx.set <- 
  plot.confusion_matrix(y.test.flatten,
                        k_best.nn_pca.predicted,
                        # print.plot_object = T,
                        export.img_file = knn_pca.eval.conf.mx.img_file,
                        backup.file = knn_pca.eval.conf.mx.obj_file)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(k_best.nn_pca.conf_mx.set)))

# knn_pca.eval.conf.mx.img <- magick::image_read(knn_pca.eval.conf.mx.img_file)
# plot(knn_pca.eval.conf.mx.img)

plot_image(knn_pca.eval.conf.mx.img_file)

# print_confusioin_matrix(k_best.nn_pca.conf_mx.set$cm.chart)

put_end_date(start)
log_close()

