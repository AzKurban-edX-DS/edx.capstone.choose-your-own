### Re-Train `RF MCC` model on full-scaled database with the best mtry value & ntree = 400 ---------------------------
open_logfile("x.train.flatten.fit_rf.mtry_best.ntree400")

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
  
  set.seed(nrow(x.test.flatten))
  
  fit_rf.mmtry_best <- randomForest(x.train.flatten, 
                                    y.train.flatten,
                                    x.test.flatten,
                                    y.test.flatten,
                                    mtry = mtry.best,
                                    ntree = 400)
  
  put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
  put_end_date(start)
  # Time difference of the last iteration 19.8342 mins
  
  ### ROC Curves
  # References:
  # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
  # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
  
  put_log("Fine-tuned `RF MCC` Model: Calculating a ROC curve for each class...")
  fit_rf.mmtry_best.roc_curves <- calc.roc_curves(y.test.flatten,
                                                  fit_rf.mmtry_best$test$votes,
                                                  Y.Labels)
  
  put_log("Fine-tuned `RF MCC` Model: The per-class ROC curve calculation has been completed.")
  
  
  put_log("Fine-tuned `RF MCC` Model: Creating a Confusion Matrix...")
  fit_rf.mmtry_best.conf.mx <- confusion_matrix(as.character(y.test.flatten),
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
%1", mean(fit_rf.mmtry_best$test$predicted == y.test.flatten))
# [1] 0.886390995545925

fit_rf.mmtry_best.accuracy.by_class <- MCClassifier.accuracy.by_class(Y.Labels,
                                                                      y.test.flatten,
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

plot_bars.accuracy.by_class(Y.Labels,
                            fit_rf.mmtry_best.accuracy.by_class,
                            title.prefix = "Tuned Random Forest-based Multiclass")
log_close()








