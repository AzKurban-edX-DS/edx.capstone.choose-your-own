#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DNN-Based MCC (DNN MCC)  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s%%%%%%

# DNN-Bsase Learning Multiclass Classifier (DNN MCC)  Model Tuning

# References:
# R interface to Keras Tuner
# https://eagerai.github.io/kerastuneR/#r-interface-to-keras-tuner

# Automating neural network configuration with Keras Tuner
# June 9, 2020 by Chris
# https://machinecurve.com/index.php/2020/06/09/automating-neural-network-configuration-with-keras-tuner

open_logfile(".dnn-mccl.model-tuning")

stopifnot(file.exists(ds28x28.split.train_0.1.backup.file),
          exists("dnn_mcc.tuner.dir"),
          exists("tdnn_mcc.best_hp.config.file"))

# Disable the elapsed time limit for expressions
# options(timeout = max(1000, getOption("timeout")))
# options(expressions = 50000) # Increases nesting limit if needed

## Prepare Input Datasets for the DNN MCC Model Tuning -------------------------

put_log("Loading the Input Datasets of 28x28-size image data...")
ds <- load_datasets(ds28x28.split.train_0.1.backup.file)
train_set <- ds$train
test_set <- ds$test
rm(ds)

put_log("The Input Dataset of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.1.backup.file)

### Prepare a Training Set -----------------------------------------------------
start <- put_start_date()
stopifnot(file.exists(ds28x28.split.train_0.1.backup.file))

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

#### Size of the Training Set by Class -----------------------------------------

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

### Prepare a Test Set ----------------------------------------------------------
start <- put_start_date()

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

#### Size of the Test Set by Class ------------------------------------------

put_log("The Test Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>

  invisible(NULL)
}

rm(y.test.groups)

## Tuning the  DNN MCC Model ---------------------------------------------------
### Init the Model Tuner Paths --------------------------------------------


dnn_mcc.best_model.plot_img.file <- file.path(dnn_mcc.tuner.plots.dat.dir, 
                                               "tuner.best-model.png")

dnn_mcc.tuner.checkpoints.dir <- file.path(dnn_mcc.tuner.dir,
                                            "checkpoints")
dnn_mcc.tuner.checkpoint.file_path <- 
  file.path(dnn_mcc.tuner.checkpoints.dir, 
            "dnn_mcc.tuner.{epoch:02d}-{val_loss:.2f}.keras")

if(!dir.exists(dnn_mcc.tuner.checkpoints.dir))
  dir.create(dnn_mcc.tuner.checkpoints.dir)

### Process the Tuning ---------------------------------------------------------
if(!is.null(dev.list())) dev.off()

put_log("Tuning the DNN MCC Model on the full Train Dataset (`x_train`) of shape: %1...",
        paste0("(", str_flatten(shape(x_train), 
                                collapse = ","),")"))
start <- put_start_date()
# Log Start Time: 2026-06-29 09:22:51.209273

# dnn_mcc.tuner <- dl.tune.hwr_model(dnn_mcc.tunable_model,
#                                     x_train,
#                                     y_train,
#                                     dnn_mcc.tuner.dir,
#                                     dnn_mcc.tuner.checkpoint.file_path,
#                                     project_name = "DNN-MCC.Tuner")


hp <- HyperParameters()
dnn_mcc.tuner.max_layers <- 5L

# Choice of one value among a predefined set of possible values.
# Choice(name, values, ordered = NULL, default = NULL, parent_name = NULL, parent_values = NULL)
hp$Choice('learning_rate', 
          c(1e-1, 
            1e-2, 
            1e-3, 
            1e-4))

hp$Int('num_layers', 
       min_value = 2L,
       max_value = dnn_mcc.tuner.max_layers)

hp$Float('dropout_rate', 
       min_value = 0.1,
       max_value = 0.5,
       step = 0.05)

dnn_mcc.callback_list <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = dnn_mcc.tuner.checkpoint.file_path,
                            monitor = "val_loss",
                            save_best_only = TRUE,
                            verbose = 1)
)

for (i in seq(dnn_mcc.tuner.max_layers)) {
  hp$Int(paste0("units_", i),
         min_value = N.classes,
         max_value = 28*28,
         step = 32)    
}

dnn_mcc.tuner = RandomSearch(
  hypermodel =  dnn_mcc.tuner.build_model,
  # hypermodel =  function(hp) dnn_mcc.tunable_model(hp,
  #                                           c(28, 28),
  #                                           N.classes,
  #                                           0.2),
  max_trials = 5,
  hyperparameters = hp,
  tune_new_entries = T,
  objective =  'val_accuracy',
  directory = dnn_mcc.tuner.dir,
  project_name = "DNN-MCC.Tuner"
  )

dnn_mcc.tuner |> fit_tuner(x = x_train,
                           y = y_train,
                           callbacks = dnn_mcc.callback_list,
                           # validation_split = 0.2,
                           validation_data = tuple(x_test, y_test),
                           epochs = 100L)
# rm(x_train,
#    y_train)

put_log("The DNN MCC Model Tuning has been completed.")
put_end_date(start)

# Best val_accuracy So Far: 0.8961053490638733
# Total elapsed time: 03h 38m 12s

put_log("The DNN MCC Tuning Results Summary:
%1", capture.output(dnn_mcc.tuner$results_summary()))

# <keras_tuner.src.tuners.randomsearch.RandomSearch object at 0x000001F3BD376D90>

class(dnn_mcc.tuner)

# This prints a summary of the search space and lists the top trial results
dnn_mcc.tuner.results <- kerastuneR::plot_tuner(dnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The DNN MCC Tuning Results:
%1", capture.output(dnn_mcc.tuner.results))
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

rm(dnn_mcc.tuner.results)

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

### Extract & Save the Best Hyper-parameter Configuration ----------------------


# dnn_mcc.tuner.best_trial.ls <- dnn_mcc.tuner$oracle$get_best_trials(num_trials = 10L)
# str(dnn_mcc.tuner.best_trial.ls)
# 
# dnn_mcc.tuner.best_trial.last_epochs <- sapply(dnn_mcc.tuner.best_trial.ls, 
#                                                function(trial){
#                                                  trial$best_step
#                                                })
# 
# dnn_mcc.retrain_epochs <- max(dnn_mcc.tuner.best_trial.last_epochs)


dnn_mcc.tuner.best_hp.ls <- dnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)
# str(dnn_mcc.tuner.best_hp.ls)

dnn_mcc.tuner.best_hp <- dnn_mcc.tuner.best_hp.ls[[1]]


# class(dnn_mcc.tuner.best_hp)
# [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
# [2] "python.builtin.object"        

#dnn_mcc.tuner.best_hp
# <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>

put_log("The best Hyperparameters values:
%1", capture.output(dnn_mcc.tuner.best_hp$values))

tdnn_mcc.best_hp.config <- dnn_mcc.tuner.best_hp$get_config()
put_log("The best Hyperparameters configuration:
%1", capture.output(tdnn_mcc.best_hp.config))

put_log("Saving the Best Hyper-parameter Configuration...")
saveRDS(tdnn_mcc.best_hp.config,
        file = tdnn_mcc.best_hp.config.file)

put_log("The Best Hyper-parameter Configuration has been saved in the following file:
  %1", tdnn_mcc.best_hp.config.file)

log_close()
# Log End Time: 2026-08-26 00:50:05.99562
# Log Elapsed Time: 0 00:48:52

