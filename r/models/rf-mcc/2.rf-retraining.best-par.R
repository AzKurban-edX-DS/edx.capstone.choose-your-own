#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RF MCC Model: Re-Train with the Best `mtry` Parameter Value
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

# Disable the elapsed time limit for expressions
# options(timeout = max(1000, getOption("timeout")))
# options(expressions = 50000) # Increases nesting limit if needed

open_logfile(".fit_rf.re-train.mtry_best.ntree400")

stopifnot(file.exists(my_emnist.split.file_path,
                      fit_rf.fine_tuned.backup.path),
          dir.exists(data.models.rf.dir))

start <- put_start_date()

## Load Fine-Tuned RF MCC Model ------------------------------------------------

put_log("Loading the Fine-Tuned RF MCC Model, tuned with `mtry` parameter values from the backup file...")

fit.bak <- readRDS(fit_rf.fine_tuned.backup.path)
fit_rf.fine_tuned <- fit.bak$fit
mtry.final_tune.values <- fit.bak$mtry
rm(fit.bak)

put_log("The `RF MCC` model, final tuned with `mtry` parameter values, 
has been loaded from the following backup file:
%1", fit_rf.fine_tuned.backup.path)
put_end_date(start)

put_log("Below are results of tuning the model by `mtry` parameter values, 
trained using `Random Forest` method on a 10% sample of the`Training Set` dataset:
%1", capture.output(fit_rf.fine_tuned$results[,1:3]))

{
  #   mtry  Accuracy     Kappa
  # 1   42 0.8299246 0.8254489
  # 2   43 0.8302262 0.8257585
  # 3   45 0.8301056 0.8256347
  # 4   46 0.8293816 0.8248916
  # 5   48 0.8299849 0.8255108
  # 6   49 0.8298643 0.8253870
  invisible(NULL)
}
put_end_date(start)

put_log("Confusion matrix obtained from the evaluation of the finally tuned model:
%1", capture.output(confusionMatrix(fit_rf.fine_tuned)))

ggplot(fit_rf.fine_tuned)

acc.final_tuned.max <- max(fit_rf.fine_tuned$results$Accuracy)

put_log("The best accuracy obtained from the evaluation of the fine-tuned model:
%1", capture.output(acc.final_tuned.max))
# 0.8302262

acc.final_tuned.max.idx <- which.max(fit_rf.fine_tuned$results$Accuracy)
# 2
mtry.final_tuned.best <- mtry.final_tune.values[acc.final_tuned.max.idx]
# 43

stopifnot(mtry.final_tuned.best == fit_rf.fine_tuned$bestTune)

fit_rf.mtry.best <- ifelse(acc.final_tuned.max > acc.fine_tuned.max, 
                           mtry.final_tuned.best,
                           mtry.fine_tuned.best)

put_log("The best parameter value obtained as a result of the RF Model tuning:
%1", capture.output(fit_rf.mtry.best))
# 44

## Prepare Input Datasets ------------------------------------------------------

put_log("Loading the Split Flattened Dataset allocated 20% for the Text Set from the backup file...")

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

## Re-Train `RF MCC` model on full-scaled database with the best `mtry` value & ntree = 400 ----

put_log("Training the `RF MCC` model with the best `mtry` parameter value...")
set.seed(N.classes)

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

set.seed(nrow(x_test))

fit_rf.final <- randomForest(x_train, 
                             y_train,
                             x_test,
                             y_test,
                             mtry = fit_rf.mtry.best,
                             ntree = 400)

put_log("The `RF MCC` Model has been trained with the best `mtry` parameter value.")
put_end_date(start)

stopCluster(cl)
stopImplicitCluster()

fit_rf.final$test$accuracy <-  mean(fit_rf.final$test$predicted == y_test)

fit_rf.final$test$targets <- y_test

put_log("Saving the fine-tuned `RF MCC` Model data...")

saveRDS(fit_rf.final,
        file = fit_rf.final.backup.path)

put_log("The data of the fine-tuned `RF MCC` Model has been saved to the following file:
%1", fit_rf.final.backup.path)
put_end_date(start)
# Time difference of  mins

rm(x_train, 
   y_train,
   x_test,
   y_test)

log_close()
# Log Elapsed Time: 0 01:37:25

