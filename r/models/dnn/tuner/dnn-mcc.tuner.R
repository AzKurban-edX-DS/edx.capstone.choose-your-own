#%%%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model Tuning

# References:
# R interface to Keras Tuner
# https://eagerai.github.io/kerastuneR/#r-interface-to-keras-tuner

# Automating neural network configuration with Keras Tuner
# June 9, 2020 by Chris
# https://machinecurve.com/index.php/2020/06/09/automating-neural-network-configuration-with-keras-tuner

# Disable the elapsed time limit for expressions
options(timeout = max(1000, getOption("timeout")))
options(expressions = 50000) # Increases nesting limit if needed

## Prepare a Training Set for the Model Tuning ------------------------------------
open_logfile(".prepare-train-set-for-dl.model-tuning")
start <- put_start_date()
stopifnot(file.exists(ds28x28.split.train_0.1.backup.file))

put_log("Loading the Training Set of 28x28-size image data...")
train_set <- load.train_set(ds28x28.split.train_0.1.backup.file)

put_log("The Training Set of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.1.backup.file)


put_log("The Training Set object structure is as follows:
%1", capture.output(str(train_set)))

x_train <- train_set$x
# storage.mode(x_train) <- "integer"

# x_train <- x_train[seq(1e4),,]
str(x_train)
dim(x_train)

y.train.groups <- train_set$class_groups
rm(train_set)

stopifnot(sum(as.character(y.train.groups$classID) != rownames(x_train)) == 0)


y_train <- as.array(as.integer(y.train.groups$classID) - 1)
str(y_train)
dim(y_train)

stopifnot(min(y_train) == 0,
          max(y_train) == 38,
          dim(y_train) == nrow(x_train))

### Size of the Training Set by Class ------------------------------------------

put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
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

rm(y.train.groups)
log_close()

## Tuning the  BDL MCC Model ---------------------------------------------------
open_logfile(paste0(".dl-basic-model.tuning_", 
                    paste0("x_train(",
                           str_flatten(shape(x_train), 
                                       collapse = ","),")")))
### Init the Model Tuner Paths --------------------------------------------

dnn_mcc.final.file_path <- file.path(dnn_mcc.tuner.dir, 
                                      "dnn-mcc.final-model.keras")

dnnb.final_model.train_history.file_path <- file.path(dnn_mcc.tuner.dir, 
                                                    "dnn_mcc.final_model.train_history.rds")

dnn_mcc.tune.plot_img.dir <- file.path(dnn_mcc.tuner.dir, "plot.img")

if(!dir.exists(dnn_mcc.tune.plot_img.dir))
  dir.create(dnn_mcc.tune.plot_img.dir)

dnn_mcc.best_model.plot_img.file <- file.path(dnn_mcc.tune.plot_img.dir, 
                                               "tuner.best-model.png")
dnn_mcc.final_model.plot_img.file <- file.path(dnn_mcc.tune.plot_img.dir, 
                                               "tuner.final-model.png")

dnn_mcc.final.conf_mx.bak <- file.path(dnn_mcc.tune.plot_img.dir, 
                                               "tuner-final.confusion-matrix.rds")

dnn_mcc.final.conf_mx.img <- file.path(dnn_mcc.tune.plot_img.dir, 
                                               "tuner-final.confusion-matrix.png")


dnn_mcc.tuner.checkpoints.dir <- file.path(dnn_mcc.tuner.dir,
                                            "checkpoints")
if(!dir.exists(dnn_mcc.tuner.checkpoints.dir))
  dir.create(dnn_mcc.tuner.checkpoints.dir)

dnn_mcc.tuner.checkpoint.file_path <- 
  file.path(dnn_mcc.tuner.checkpoints.dir, 
            "dnn_mcc.tuner.{epoch:02d}-{val_loss:.2f}.keras")

dnn_mcc.best.checkpoints.dir <- file.path(dnn_mcc.tuner.dir,
                                            "checkpoints.best")
if(!dir.exists(dnn_mcc.best.checkpoints.dir))
  dir.create(dnn_mcc.best.checkpoints.dir)

dnn_mcc.best.checkpoint.file_path <- 
  file.path(dnn_mcc.best.checkpoints.dir, 
            "dnn_mcc.best.{epoch:02d}-{val_loss:.2f}.keras")

### Process the Tuning ---------------------------------------------------------
if(!is.null(dev.list())) dev.off()

put_log("Tuning the DL Basic Model on the full Train Dataset (`x_train`) of shape: %1...",
        paste0("(", str_flatten(shape(x_train), 
                                collapse = ","),")"))
start <- put_start_date()
# Log Start Time: 2026-06-29 09:22:51.209273

dnn_mcc.tuner <- dl.tune.hwr_model(dnn_mcc.tunable_model,
                                    x_train,
                                    y_train,
                                    dnn_mcc.tuner.dir,
                                    dnn_mcc.tuner.checkpoint.file_path,
                                    project_name = "DL.Basic.Tuner")
rm(x_train,
   y_train)

put_log("The DL Basic Model Tuning has been completed.")
put_end_date(start)

# Best val_accuracy So Far: 0.8961053490638733
# Total elapsed time: 03h 38m 12s

put_log("The DL Basic Tuning Results Summary:
%1", capture.output(dnn_mcc.tuner$results_summary()))

# <keras_tuner.src.tuners.randomsearch.RandomSearch object at 0x000001F3BD376D90>

class(dnn_mcc.tuner)

# This prints a summary of the search space and lists the top trial results
dnnb.model_tuner.result <- kerastuneR::plot_tuner(dnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The DL Basic Tuning Results:
%1", capture.output(dnnb.model_tuner.result))
{
  # [[1]]
  # 
  # [[2]]
  #   message learning_rate num_layers units_1 units_2 units_3 units_4 units_5 units_6
  # 1      NA         1e-04          9     455     775     551     295     583     199
  # 2      NA         1e-02          5     359     615     615     359     743     487
  # 3      NA         1e-03          9     551     679     231     295     391     327
  # 4      NA         1e-03         14     263     391     647     647     551      71
  # 5      NA         1e-02          8     199     679     423     583     199     615
  #   units_7 units_8 units_9 units_10 units_11 units_12 units_13 units_14 sunits_15
  # 1     743     647     391      103      743      455      583      647      679
  # 2     679     487     711      327      135      359      679      167      487
  # 3     391     231      39      391      519      199      775       39      679
  # 4     391     231     327      359      487      391      199       71      199
  # 5     519     295     679      199      647      167      583      551      583
  #   units_16 units_17 units_18 units_19 units_20 best_step     score
  # 1      359      455      359      263       39        45 0.8933208
  # 2      391      359      359      295      519        14 0.8550518
  # 3      519      487       39      743      519        19 0.8961053
  # 4      103      423      551      359      455        31 0.8862841
  # 5      231      199      583      263       71        19 0.8117780
}

rm(dnnb.model_tuner.result)

# This prints the top trials, their hyperparameters, and execution details
dnn_mcc.tuner |> results_summary(num_trials = 1L)
{
  # Results summary
  # Results in data/models/dl.keras3/dnn_mcc/tuning/keras-tuner\DL.Basic.Tuner
  # Showing 1 best trials
  # Objective(name="val_accuracy", direction="max")
  # 
  # Trial 2 summary
  # Hyperparameters:
  #   learning_rate: 0.001
  # num_layers: 9
  # units_1: 551
  # units_2: 679
  # units_3: 231
  # units_4: 295
  # units_5: 391
  # units_6: 327
  # units_7: 391
  # units_8: 231
  # units_9: 39
  # units_10: 391
  # units_11: 519
  # units_12: 199
  # units_13: 775
  # units_14: 39
  # units_15: 679
  # units_16: 519
  # units_17: 487
  # units_18: 39
  # units_19: 743
  # units_20: 519
  # Score: 0.8961053490638733
}

# Retrieve the best model from the search
dnn_mcc.best_models <- kerastuneR::get_best_models(tuner = dnn_mcc.tuner, num_models = 1L)
dnn_mcc.best_model <- dnn_mcc.best_models[[1]]
rm(dnn_mcc.best_models)

dnn_mcc.best_model$summary()
# View completed epochs of this best model
# If restore_best_weights = TRUE, this tells you the optimal epoch
# best_epoch <- dnn_mcc.best_model$history$params$epochs

dnn_mcc.best_model |> plot_keras_model(to_file = dnn_mcc.best_model.plot_img.file,
                                        show_shapes = TRUE)

dnn_mcc.tuner.best_trials <- dnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
dnn_mcc.best_trial <- dnn_mcc.tuner.best_trials[[1]]
dnn_mcc.best_trial$summary()
dnn_mcc.best_trial$best_step

dnn_mcc.best_trial$metrics$get_history('val_accuracy')

rm(dnn_mcc.tuner.best_trials,
   dnn_mcc.best_trial)

log_close()
# Log Elapsed Time: 0 00:10:44

## Prepare a Training Set for Re-training the Final Model -----------------------
open_logfile(".train-set.prepare.retraining.final-dl.model")
start <- put_start_date()
stopifnot(file.exists(ds28x28.split.train_0.8.backup.file))

put_log("Loading the Training Set of 28x28-size image data...")
train_set <- load.train_set(ds28x28.split.train_0.8.backup.file)

put_log("The Training Set of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.1.backup.file)


put_log("The Training Set object structure is as follows:
%1", capture.output(str(train_set)))

x_train <- train_set$x
# storage.mode(x_train) <- "integer"

# x_train <- x_train[seq(1e4),,]
str(x_train)
dim(x_train)

y.train.groups <- train_set$class_groups
rm(train_set)

stopifnot(sum(as.character(y.train.groups$classID) != rownames(x_train)) == 0)


y_train <- as.array(as.integer(y.train.groups$classID) - 1)
str(y_train)
dim(y_train)

stopifnot(min(y_train) == 0,
          max(y_train) == 38,
          dim(y_train) == nrow(x_train))

### Size of the Training Set by Class ------------------------------------------

put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
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

rm(y.train.groups)
log_close()

## Re-training the Final Model --------------------------------------------------

open_logfile("re-training.best.dnn_mcc-model")


if(file.exists(dnn_mcc.final.file_path)) {
  put_log("Loading pre-trained BDL MCC Model...")
  
  dnn_mcc.final_model <- keras3::load_model(dnn_mcc.final.file_path)
  
  put_log("The BDL MCC Model has been loaded from the backup file:
%1", dnn_mcc.final.file_path)
  
  if(file.exists(dnnb.final_model.train_history.file_path)){
    put_log("Loading the BDL MCC Model Train History...")
    
    dnnb.final_model.train_history <- readRDS(dnnb.final_model.train_history.file_path)
    
    put_log("The BDL MCC Model has been loaded from the backup file:
%1", dnnb.final_model.train_history.file_path)
  } else {
    warning("The BDL MCC Model backup does not exist:
", dnnb.final_model.train_history.file_path)
  }
} else {
  dnn_mcc.tuner.best_trial.ls <- dnn_mcc.tuner$oracle$get_best_trials(num_trials = 10L)
  str(dnn_mcc.tuner.best_trial.ls)
  
  dnn_mcc.tuner.best_trial.last_epochs <- sapply(dnn_mcc.tuner.best_trial.ls, 
                                                  function(trial){
                                                    trial$best_step
                                                  })
  
  dnn_mcc.retrain_epochs <- max(dnn_mcc.tuner.best_trial.last_epochs)
  
  
  dnn_mcc.tuner.best_hp.ls <- dnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)
  # str(dnn_mcc.tuner.best_hp.ls)
  
  dnn_mcc.tuner.best_hp <- dnn_mcc.tuner.best_hp.ls[[1]]
  
  put_log("The best Hyperparameters configuration:
%1", capture.output(dnn_mcc.tuner.best_hp$get_config()))
  
  class(dnn_mcc.tuner.best_hp)
  # [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
  # [2] "python.builtin.object"        
  
  dnn_mcc.tuner.best_hp
  # <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>
  
  put_log("The best Hyperparameters values:
%1", capture.output(dnn_mcc.tuner.best_hp$values))
  
  # 1. Re-build a clean model structure using the winning hyperparams
  dnn_mcc.final_model <- dnn_mcc.tuner$hypermodel$build(dnn_mcc.tuner.best_hp)
  # print(dnn_mcc.final_model)
  # dnn_mcc.final_model$summary()
  
  put_log("The Final tuned BDL Model Summary: 
%1", capture.output(dnn_mcc.final_model))
  
  dnn_mcc.final_model |> plot_keras_model(to_file = dnn_mcc.final_model.plot_img.file,
                                           show_shapes = T)
  
  #best_models <- tuner |> get_best_models(num_models = 1L)
  # best_5_models[[1]] %>% plot_keras_model()
  
  dnn_mcc.best.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = dnn_mcc.best.checkpoint.file_path,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  put_log("Training the BDL MCC Model...")
  start <- put_start_date()
  
  dnnb.final_model.train_history <- dnn_mcc.final_model |> 
    fit(x_train, 
        y_train, 
        epochs = dnn_mcc.retrain_epochs, 
        # batch_size = 128, 
        callbacks = dnn_mcc.best.callbacks,
        validation_split = 0.2
    )
  
  put_log("Saving re-trained final BDL MCC Model...")
  keras3::save_model(dnn_mcc.final_model,
                     filepath = dnn_mcc.final.file_path,
                     overwrite = TRUE)
  
  put_log("The re-trained final BDL MCC Model has been trained 
and saved in the following file:
  %1", dnn_mcc.final.file_path)
  
  put_log("Saving the BDL MCC Model History...")
  saveRDS(dnnb.final_model.train_history,
          file = dnnb.final_model.train_history.file_path)
  
  put_log("The re-trained final BDL MCC Model History has been trained 
and saved in the following file:
  %1", dnnb.final_model.train_history.file_path)
  put_end_date(start)
  # Time difference of 38.48235 mins
}

rm(x_train,
   y_train,
   dnn_mcc.tuner)

put_log("The re-trained `BDL MCC` Model has been trained with the following results
%1", dnn_mcc.final_model)

plot(dnnb.final_model.train_history)
str(dnnb.final_model.train_history)

rm(dnnb.final_model.train_history)

log_close()
# Log Elapsed Time: 0 00:13:05

## Prepare a Test Set for the Final Model Evaluation ---------------------------
open_logfile(".test-set.eval.final-dl.model")
start <- put_start_date()

put_log("Loading the Test Set of 28x28-size image data...")
test_set <- load.test_set(ds28x28.split.train_0.8.backup.file)

put_log("The Test Set of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)


put_log("The Test Set object structure is as follows:
%1", capture.output(str(test_set)))

x_test <- test_set$x
# storage.mode(x_test) <- "integer"

# x_test <- x_test[seq(1e4),,]
str(x_test)
dim(x_test)

y.test.groups <- test_set$class_groups
rm(test_set)

stopifnot(sum(as.character(y.test.groups$classID) != rownames(x_test)) == 0)

y_test <- as.array(as.integer(y.test.groups$classID) - 1)
str(y_test)
dim(y_test)

stopifnot(min(y_test) == 0,
          max(y_test) == 38,
          dim(y_test) == nrow(x_test))

### Size of the Test Set by Class ------------------------------------------

put_log("The Test Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
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

rm(y.test.groups)
log_close()

## The Final BDL MCC Model Evaluation ------------------------------------------

if(file.exists(dnn_mcc.final.eval.result.file)) {
  put_log("Loading the BDL MCC Final Model Evaluation Result object...")
  dnnb_final.eval.result <- readRDS(dnn_mcc.final.eval.result.file)
  
  put_log("The BDL MCC Final Model Evaluation Result object has been loaded 
from the following file:
%1", dnn_mcc.final.eval.result.file)
} else {
  put_log("Evaluating DL Model...")
  dnnb_final.eval.result <- dnn_mcc.final_model |> evaluate(x_test, y_test)
  put_log("BDL MCC Bset Model evaluation result:
%1", capture.output(str(dnnb_final.eval.result)))
  # List of 2
  # $ accuracy: num 0.9
  # $ loss    : num 0.374
  
  
  put_end_date(start)
  # Time difference of 1.668308 mins
  
  dnnb_final.preds <- dnn_mcc.final_model |> predict(x_test)
  str(dnnb_final.preds)
  # put_end_date(start)
  # Time difference of  mins
  
  colnames(dnnb_final.preds) <- Y.Labels
  head(dnnb_final.preds[,1:5])
  #                 #            $            &            @            0
  # [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
  # [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
  # [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
  # [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
  # [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
  # [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
  dim(dnnb_final.preds)
  #> [1] 33228    39
  
  dnnb_final.eval.result$predicted.probs <- dnnb_final.preds
  str(dnnb_final.eval.result)
  
  dnnb_final.preds.ts <- as_tensor(dnnb_final.preds)
  str(dnnb_final.preds.ts)
  #> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>
  
  dnnb_final.predictions <- dnnb_final.preds.ts |> op_argmax(2)
  dnnb_final.predictions
  #> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
  dim(dnnb_final.predictions)
  #> [1] 33228
  # dnnb_final.predictions$numpy()
  
  
  # y_test
  # as.integer(y_test)
  
  dnnb_final.pred.values.idx <- dnnb_final.predictions$numpy()
  head(dnnb_final.pred.values.idx)
  min(dnnb_final.pred.values.idx)
  max(dnnb_final.pred.values.idx)
  
  dnnb_final.pred.values <- Y.Labels[dnnb_final.pred.values.idx]
  head(dnnb_final.pred.values)
  
  dnnb_final.eval.result$predicted.values <- dnnb_final.pred.values
  str(dnnb_final.eval.result)
  
  
  y_test.idx <- y_test + 1
  y_test.labels <- Y.Labels[y_test.idx]
  
  dnnb_final.eval.result$targets <- y_test.labels
  
  rm(y_test.labels,
     y_test.idx,
     dnnb_final.pred.values,
     dnnb_final.pred.values.idx,
     dnnb_final.predictions,
     dnnb_final.preds.ts,
     dnnb_final.preds)
  
  put_log("Saving the BDL MCC Model  Evaluation Result object...")
  saveRDS(dnnb_final.eval.result,
          file = dnn_mcc.final.eval.result.file)
  
  put_log("The BDL MCC Model  Evaluation Result object has been trained 
and saved in the following file:
  %1", dnn_mcc.final.eval.result.file)
  
}

rm(x_test,
   y_test)

# dnn_mcc.accuracy <- mean(dnnb_final.pred.values.idx == y_test.idx)
put_log("The overall Basic `DL MCC` Model accuracy: %1", 
        dnnb_final.eval.result$accuracy)
# 0.898338750451427

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".dnnb-final.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path))

dnnb.keras_tunes.plots.dat.dir <- file.path(dnn_mcc.tuner.dir, "plots.dat")

if(!dir.exists(dnnb.keras_tunes.plots.dat.dir))
  dir.create(dnnb.keras_tunes.plots.dat.dir)

dnnb_final.eval.conf.mx.img_file <- file.path(dnnb.keras_tunes.plots.dat.dir,
                                            "dnnb-final.eval.confusion-matrix.png")

dnnb_final.eval.plots_dat.file <- file.path(dnnb.keras_tunes.plots.dat.dir,
                                          "dnnb-final.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(dnnb_final.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(dnnb_final.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", dnnb_final.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = dnnb_final.eval.result$targets,
                                predicted.probabilities = dnnb_final.eval.result$predicted.probs,
                                predicted.values = dnnb_final.eval.result$predicted.values,
                                alg_name = "DL Basic Tuned",
                                plots_dat.file = dnnb_final.eval.plots_dat.file,
                                cm.export.img_file = dnnb_final.eval.conf.mx.img_file,
                                cm.print.image = T)
}

rm(dnnb_final.eval.result)

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
        file = dnnb_final.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", dnnb_final.eval.plots_dat.file)

# put_log("The Basic DL Model per-class accuracy:,
# %1", capture.output(plots.dat$PCA$acc.by_class))
{

  
  
  
  invisible(NULL)
}

rm(plots.dat)
log_close()


