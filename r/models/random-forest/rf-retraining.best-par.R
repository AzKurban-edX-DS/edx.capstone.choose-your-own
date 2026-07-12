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

fit_rf.mtry_best.backup.path <- file.path(data.models.random_forest.dir, 
                                           "fit_rf.mtry_best.ntree400.back.rds")

start <- put_start_date()

if(file.exists(fit_rf.mtry_best.backup.path)) {
  # No longer needed since the pre-trained model will be loaded from the backup file
  rm(x_train, 
     y_train,
     x_test)
  
  put_log("Loading data of the fine-tuned `RF MCC` Model by the `mtry` parameter...")
  
  fit_rf.mtry_best <- readRDS(fit_rf.mtry_best.backup.path)

  put_log("The data of the fine-tuned `RF MCC` Model, 
trained with the best `mtry` parameter value, has been loaded from the following backup file:
%1", fit_rf.mtry_best.backup.path)
  put_end_date(start)
} else {
  put_log("Training the `RF MCC` model with the best `mtry` parameter value...")
  set.seed(N.classes)
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  set.seed(nrow(x_test))
  
  fit_rf.mtry_best <- randomForest(x_train, 
                                   y_train,
                                   x_test,
                                   y_test,
                                   mtry = fit_rf.mtry.best,
                                   ntree = 400)
  rm(x_train, 
     y_train,
     x_test)
  
  put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("Saving the fine-tuned `RF MCC` Model data...")
  
  saveRDS(fit_rf.mtry_best,
          file = fit_rf.mtry_best.backup.path)
  
  put_log("The data of the fine-tuned `RF MCC` Model has been saved to the following file:
%1", fit_rf.mtry_best.backup.path)
  put_end_date(start)
  # Time difference of  mins
}

put_log("The results of the fine-tuning `RF MCC` Model (after being trained with the best `mtry` parameter value
on an 80% sample of the`Train Set` dataset and tested on the remaining 20% of the `Train Set`) 
are as follows:
%1", capture.output(fit_rf.mtry_best))
put_end_date(start)
# Time difference of 6.260901 hours

plot(fit_rf.mtry_best,
     main = "Fine-tuning Results of the `RF MCC` Model by the `mtry` Parameter")

put_log("Prediction accuracy of the fine-tuned 'RF MCC' Model, 
trained with the best `mtry` parameter value, is as follows:
%1", mean(fit_rf.mtry_best$test$predicted == y_test))
# 0.886029854339713

log_close()
# Log Elapsed Time: 0 01:37:25

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

data.models.plot_img.dir <- file.path(data.models.random_forest.dir, "plot.img")

if(!dir.exists(data.models.plot_img.dir))
  dir.create(data.models.plot_img.dir)

rf_best.eval_plots.file <- file.path(data.dir,
                                     "rf_best.eval_plots.rds")

rf_best.eval.conf.mx.obj_file <- file.path(data.dir,
                                           "rf-final.eval.confusion-matrix.rds")

rf_best.eval.conf.mx.img_file <- file.path(data.models.plot_img.dir,
                                           "rf-final.eval.confusion-matrix.png")

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

### ROC Curves
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

put_log("Plotting ROC curves the Model Evaluation Results...")
eval.roc_curves <- plot.ROC.curves(y_test,
                                   fit_rf.mtry_best$test$votes)
eval.roc_curves <- plot_ROC(y_test,
                            fit_rf.mtry_best$test$votes)
plot_ROC(eval.roc_curves)

Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")

fit_rf.mtry_best.accuracy.by_class <- 
  plot.per_class.accuracy.bars(y_test,
                               fit_rf.mtry_best$test$predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(fit_rf.mtry_best.accuracy.by_class))
{
  #' class  accuracy
  #'     # 1.0000000
  #'     $ 1.0000000
  #'     & 1.0000000
  #'     @ 1.0000000
  #'     0 0.9636150
  #'     1 0.7347418
  #'     2 0.8556338
  #'     3 0.9436620
  #'     4 0.9049296
  #'     5 0.8556338
  #'     6 0.9002347
  #'     7 0.9624413
  #'     8 0.8755869
  #'     9 0.9225352
  #'     A 0.8650235
  #'     B 0.8943662
  #'     C 0.9448357
  #'     D 0.8814554
  #'     E 0.9143192
  #'     F 0.9260563
  #'     G 0.6068075
  #'     H 0.9143192
  #'     I 0.6819249
  #'     J 0.9154930
  #'     K 0.9131455
  #'     L 0.5375587
  #'     M 0.9518779
  #'     N 0.9190141
  #'     P 0.9483568
  #'     Q 0.6572770
  #'     R 0.9072770
  #'     S 0.8685446
  #'     T 0.9154930
  #'     U 0.9260563
  #'     V 0.9166667
  #'     W 0.9647887
  #'     X 0.9260563
  #'     Y 0.8380282
  #'     Z 0.9014085
  invisible(NULL)
}

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

rf_best.eval.conf_mx.set <- 
  plot.confusion_matrix(y_test,
                        fit_rf.mtry_best$test$predicted,
                        # print.plot_object = T,
                        export.img_file = rf_best.eval.conf.mx.img_file,
                        backup.file = rf_best.eval.conf.mx.obj_file)
rm(y_test)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(rf_best.eval.conf_mx.set)))

# rf_best.eval.conf.mx.img <- magick::image_read(rf_best.eval.conf.mx.img_file)
# plot(rf_best.eval.conf.mx.img)

plot_image(rf_best.eval.conf.mx.img_file)

# print_confusioin_matrix(rf_best.eval.conf_mx.set$cm.chart)

put_end_date(start)
log_close()









