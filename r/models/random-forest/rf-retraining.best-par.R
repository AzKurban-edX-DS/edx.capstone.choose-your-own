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

  put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
  put_end_date(start)

  stopCluster(cl)
  stopImplicitCluster()
  
  fit_rf.mtry_best$test$accuracy <-  mean(fit_rf.mtry_best$test$predicted == y_test)
  
  fit_rf.mtry_best$test$targets <- y_test
  
  put_log("Saving the fine-tuned `RF MCC` Model data...")
  
  saveRDS(fit_rf.mtry_best,
          file = fit_rf.mtry_best.backup.path)
  
  put_log("The data of the fine-tuned `RF MCC` Model has been saved to the following file:
%1", fit_rf.mtry_best.backup.path)
  put_end_date(start)
  # Time difference of  mins
}



rm(x_train, 
   y_train,
   x_test,
   y_test)

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
%1", fit_rf.mtry_best$test$accuracy)
# 0.886029854339713

log_close()
# Log Elapsed Time: 0 01:37:25

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".rf-tuned.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path))

rf_tuned.eval.conf.mx.img_file <- file.path(data.models.rf.plots.dat.dir,
                                           "rf-tuned.eval.confusion-matrix.png")

rf_tuned.eval.plots_dat.file <- file.path(data.models.rf.plots.dat.dir,
                                         "rf-tuned.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(rf_tuned.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(rf_tuned.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", rf_tuned.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = fit_rf.mtry_best$test$targets,
                                predicted.probabilities = fit_rf.mtry_best$test$votes,
                                predicted.values = fit_rf.mtry_best$test$predicted,
                                alg_name = "Random Forest",
                                plots_dat.file = rf_tuned.eval.plots_dat.file,
                                cm.export.img_file = rf_tuned.eval.conf.mx.img_file,
                                cm.print.image = T)
}

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args,
   fit_rf.mtry_best)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

put_log("Saving the model-related plots input data object to file...")

saveRDS(plots.dat,
        file = rf_tuned.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", rf_tuned.eval.plots_dat.file)

rm(plots.dat)
log_close()


