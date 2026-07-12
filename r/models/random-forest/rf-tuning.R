#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Random Forest Multiclass Classifier (RF MCC) Model 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Reference:
# 3.6 Random Forest
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#random-forest

# library(randomForest)

# Disable the elapsed time limit for expressions
# options(timeout = max(1000, getOption("timeout")))
# options(expressions = 50000) # Increases nesting limit if needed

## Prepare Input Datasets ------------------------------------------------------

stopifnot(file.exists(my_emnist.0.1split.file_path))

open_logfile(".rf.load-split.10%train.balanced_sample")
start <- put_start_date()

### Loading Split Flattened Dataset allocated 10% for the Train Set ------------

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.0.1split.file_path)

put_log("The Split Flattened Dataset has been loaded from the folowing file:
%1", my_emnist.0.1split.file_path)

put_log("The Split Flattened Dataset has the follwoing structure:
%1", capture.output(str(ds)))

x_train <- ds$train$x

put_log("The Train set is balanced with respect to the set of classes:
%1", capture.output(print(ds$train$class_groups$groupByClass, n = N.classes)))
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
  #  1 #        3834
  #  2 $        3834
  #  3 &        3834
  #  4 @        3834
  #  5 0        3834
  #  6 1        3834
  #  7 2        3834
  #  8 3        3834
  #  9 4        3834
  # 10 5        3834
  # 11 6        3834
  # 12 7        3834
  # 13 8        3834
  # 14 9        3834
  # 15 A        3834
  # 16 B        3834
  # 17 C        3834
  # 18 D        3834
  # 19 E        3834
  # 20 F        3834
  # 21 G        3834
  # 22 H        3834
  # 23 I        3834
  # 24 J        3834
  # 25 K        3834
  # 26 L        3834
  # 27 M        3834
  # 28 N        3834
  # 29 P        3834
  # 30 Q        3834
  # 31 R        3834
  # 32 S        3834
  # 33 T        3834
  # 34 U        3834
  # 35 V        3834
  # 36 W        3834
  # 37 X        3834
  # 38 Y        3834
  # 39 Z        3834
  invisible(NULL)
}

y_test <- ds$test$class_groups$classID

stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0)
stopifnot(nrow(x_test) == length(y_test))
rm(ds)

log_close()

## Model Building & Tuning -----------------------------------------------------
open_logfile("x0.1.train.flatten.fit_rf.mtry_default.ntree500")

#if(!is.null(dev.list())) dev.off()
graphics.off()


models.random_forest.dir <- file.path(models_data.dir, "random-forest")

if(!dir.exists(models.random_forest.dir))
  dir.create(models.random_forest.dir)

models.rf.tune.path = file.path(models.random_forest.dir, "tune")

if(!dir.exists(models.rf.tune.path))
  dir.create(models.rf.tune.path)

fit_rf.mtry_default.backup.path <- file.path(models.random_forest.dir, 
                                             "fit_rf.mtry_default.ntree500.back.rds")

### Pre-Train RF MC` model with the default mtry value & ntree = 500 ---------------

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
  
  fit_rf.mtry_default <- randomForest(x_train, 
                                      y_train,
                                      x_test,
                                      y_test,
                                      keep.forest = TRUE,
                                      ntree = 500)
  
  put_log("The `RF MCC` Model has been trained with the default `mtry` parameter value.")
  put_end_date(start)
  
  put_log("`RF MCC` Model pre-trained with the default `mtry` parameter value: Creating a Confusion Matrix...")
  rf_conf.mx.mtry_default <- confusion_matrix(as.character(y_test),
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

plot(fit_rf.mtry_default, 
     main = "`RF MCC` Model Pre-trained with the Default `mtry` Parameter Value")

put_log("Prediction accuracy of the `RF MCC` Model,
pre-trained with the default `mtry` parameter value, is as follows:
%1", mean(fit_rf.mtry_default$test$predicted == y_test))
# [1] 0.839746933643647

log_close()
# Log Elapsed Time: 0 00:10:43

### Tune `RF MCC` model with `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) & ntree = 200 ----
### Step 1. Coarse Tuning: `mtry` ranged from sqrt(p)/2 to 2*sqrt(p) by step 6 ----
open_logfile(".x0.1.train.flatten.fit_rf.tune_mtry")

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
  fit_rf.mtry_tuned <- train(x_train, 
                             y_train,
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


put_log("Results of the coarse tuning of the model by `mtry` parameter values, 
trained using the `Random Forest` method on a 10% sample of the`Train Set` dataset:
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
  invisible(NULL)
}
put_end_date(start)

put_log("Confusion matrix obtained from the pre-trained model evaluation following coarse tuning:
%1", capture.output(confusionMatrix(fit_rf.mtry_tuned)))

ggplot(fit_rf.mtry_tuned)

log_close()

### Step 2. Fine Tuning: `mtry` ranged from 38 to 50 by step 3 ----
open_logfile(".x0.1.train.flatten.fit_rf.fine-tune_mtry")

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
      # index = createFolds(y_train, k = 5),
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
  fit_rf.mtry.fine_tuned <- train(x_train, 
                                  y_train,
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


put_log("Results of the fine-tuning of the model by `mtry` parameter values, 
trained using the `Random Forest` method on a 10% sample of the`Train Set` dataset:
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

put_log("Confusion matrix obtained from the evaluation of the fine-tuned model:
%1", capture.output(confusionMatrix(fit_rf.mtry.fine_tuned)))

ggplot(fit_rf.mtry.fine_tuned)

acc.fine_tuned.max <- max(fit_rf.mtry.fine_tuned$results$Accuracy)
# 0.8312519
acc.fine_tuned.max.idx <- which.max(fit_rf.mtry.fine_tuned$results$Accuracy)
# 3
mtry.fine_tuned.best <- mtry.fine_tune.values[acc.max.idx]
# 44

log_close()

### Step 3. Final Tuning: `mtry` ranged from 42 to 49 ------------------------
open_logfile(".x0.1.train.flatten.fit_rf.fine-tune_mtry")

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
      # index = createFolds(y_train, k = 5),
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
  fit_rf.mtry.final_tuned <- train(x_train, 
                                   y_train,
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

put_log("Confusion matrix obtained from the evaluation of the finally tuned model:
%1", capture.output(confusionMatrix(fit_rf.mtry.final_tuned)))

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

