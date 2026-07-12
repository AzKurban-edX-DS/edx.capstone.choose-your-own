#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RF MCC Model: Re-Train with the Best `mtry` Parameter Value
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

# Disable the elapsed time limit for expressions
# options(timeout = max(1000, getOption("timeout")))
# options(expressions = 50000) # Increases nesting limit if needed


## Prepare Input Datasets ------------------------------------------------------

open_logfile(".split.80%train.balanced_subset")

stopifnot(exists("fit_rf.mtry.best"),
          is.numeric(fit_rf.mtry.best),
          file.exists(my_emnist.split.file_path),
          dir.exists(data.models.random_forest.dir))

start <- put_start_date()

### Loading Split Flattened Dataset allocated 10% for the Train Set ------------

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

## Re-Train `RF MCC` model on full-scaled database with the best `mtry` value & ntree = 400 ----

open_logfile(".fit_rf.re-train.mtry_best.ntree400")

fit_rf.mmtry_best.backup.path <- file.path(data.models.random_forest.dir, 
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
  
  set.seed(nrow(x_test))
  
  fit_rf.mmtry_best <- randomForest(x_train, 
                                    y_train,
                                    x_test,
                                    y_test,
                                    mtry = fit_rf.mtry.best,
                                    ntree = 400)
  
  put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
  put_end_date(start)
  # Time difference of the last iteration 19.8342 mins

#   ### ROC Curves
#   # References:
#   # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
#   # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
#   
  # put_log("Fine-tuned `RF MCC` Model: Calculating a ROC curve for each class...")
  # fit_rf.mmtry_best.roc_curves <- calc.roc_curves(y_test,
  #                                                 fit_rf.mmtry_best$test$votes,
  #                                                 Y.Labels)

#   put_log("Fine-tuned `RF MCC` Model: The per-class ROC curve calculation has been completed.")
#   
#   
#   put_log("Fine-tuned `RF MCC` Model: Creating a Confusion Matrix...")
#   fit_rf.mmtry_best.conf.mx <- confusion_matrix(as.character(y_test),
#                                                 as.character(fit_rf.mmtry_best$test$predicted))
#   put_log("Fine-tuned `RF MCC` Model: The Confusion Matrix has been created:
# %1", capture.output(fit_rf.mmtry_best.conf.mx))
#   put_end_date(start)
  
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

# plot(fit_rf.mmtry_best.roc_curves[[1]], 
#      main = "ROC Curves for the Fine-tuned `RF MCC` Model by the `mtry` Parameter")
# for (class.idx in 2:N.classes) {
#   lines(fit_rf.mmtry_best.roc_curves[[class.idx]], col = class.idx)
# }
# 

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
%1", mean(fit_rf.mmtry_best$test$predicted == y_test))
# [1] 0.886390995545925

# fit_rf.mmtry_best.accuracy.by_class <- MCClassifier.accuracy.by_class(Y.Labels,
#                                                                       y_test,
#                                                                       fit_rf.mmtry_best$test$predicted)
# put_log("The per-class prediction accuracy of the fine-tuned 'RF MCC' Model, 
# trained with the best `mtry` parameter value, is as follows:
# %1", capture.output(fit_rf.mmtry_best.accuracy.by_class))
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

# plot_bars.accuracy.by_class(Y.Labels,
#                             fit_rf.mmtry_best.accuracy.by_class,
#                             title.prefix = "Tuned Random Forest-based Multiclass")
log_close()








