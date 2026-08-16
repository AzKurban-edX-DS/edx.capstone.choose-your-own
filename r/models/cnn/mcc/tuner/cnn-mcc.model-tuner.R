#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%
# Convolutional Neuron Network-Based Multiclass Classifier (CNN MCC)  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".setup.cnn_mcc.model-tuner")
start <- put_start_date()
stopifnot(file.exists(train.img28x28mx.array.file_path),
          dir.exists(data.cnn_mcc.tuner.best.dir))

### Init File Paths -----------------------------------------------------------

cnn_mcc.best_model.file <- file.path(data.cnn_mcc.tuner.best.dir, 
                                     "cnn_mcc.best-model.keras")

cnn_mcc.tuner.best.plot_img.file <- file.path(cnn_mcc.best.plots.dat.dir,
                                               "cnn-mcc.tuner.best-model.png")

data.cnn_mcc.tuner.checkpoints.dir <- file.path(data.cnn_mcc.tuner.dir, "checkpoints")

if(!dir.exists(data.cnn_mcc.tuner.checkpoints.dir))
  dir.create(data.cnn_mcc.tuner.checkpoints.dir)


cnn_mcc.checkpoint.tuner.file_path <- 
  file.path(data.cnn_mcc.tuner.checkpoints.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

### Prepare a Training Set for the Model Training ---------------------------------
put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")

split3d.list <- split.img28x28mx_array(train.img28x28mx.array.file_path,
                                       seed = N.classes,
                                       test_ratio = 0.9)

put_log("The Default Split Dataset object structure:
%1", capture.output(str(split3d.list)))

x3d.train_set <- split3d.list$train_set
# str(x3d.train_set)

put_log("The Training Set has been saved in the object `x3d.train_set`, 
which contains a training sample stored in the `x_train` variable having the following shape:
%1", capture.output(shape(x3d.train_set$x.train)))
# shape(132912, 28, 28)

x3d.test_set <- split3d.list$test_set
# str(x3d.test_set)

put_log("The Test Set data is stored in the object `x3d.test_set`, 
having the following structure:
%1", capture.output(str(x3d.test_set)))

put_log("Saving the Test Set to backup file for later use...")
saveRDS(x3d.test_set,
        file = cnn_mcc.x3d.test_set.bakup)

put_log("The Test Set for Basic CNN MCC Model has been saved to the following file:
%1", cnn_mcc.x3d.test_set.bakup)

rm(split3d.list,
   x3d.test_set)

x3d_train <- x3d.train_set$x.train

y.train.groups <- ds.get_classIDs.grouped(x3d_train)
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

y_train <- y.train.groups$classID
rm(y.train.groups)

stopifnot(sum(as.character(y_train) != rownames(x3d_train)) == 0)
y_train <- as.array(as.integer(y_train) - 1)
str(y_train)
dim(y_train)

stopifnot(min(y_train) == 0)
stopifnot(max(y_train) == 38)

put_log("Reshaping the Training Set to make it compatible
with the Convolutional Neural Network (CNN)...")

# Add channel into the dimension
x_train <- array_reshape(x3d_train, 
                         c(nrow(x3d.train_set$x.train), 
                           n.img_rows, 
                           n.img_cols, 
                           1))

stopifnot(length(y_train) == nrow(x_train))

put_log("The Training Set has been reshaped as follows:
%1", capture.output(shape(x_train)))
# shape(132912, 28, 28)

str(x_train)
dim(x_train)

rm(x3d.train_set,
   x3d_train)

log_close()

## Tuning CNN MCC Model --------------------------------------------------------
open_logfile(".run.cnn_mcc.model-tuner")

cnn_mcc.hypermodel <- CNN_MCC.HyperModel(num_classes = N.classes)

cnn_mcc.tuner <- Hyperband(cnn_mcc.hypermodel,
                           objective = 'val_accuracy',
                           # max_epochs = 100,
                           hyperband_iterations = 2,
                           directory = data.cnn_mcc.tuner.dir,
                           project_name = 'CNN-MCC.Tuning')

# cnn_mcc.tuner <- RandomSearch(cnn_mcc.hypermodel,
#                            objective = 'val_accuracy',
#                            # hyperparameters = cnn_mcc.hypermodel,
#                            seed = length(y_train),
#                            max_trials = 5,
#                            directory = data.cnn_mcc.tuner.dir,
#                            project_name = 'CNN-MCC.RS-Tuning')

cnn_mcc.callbacks <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = cnn_mcc.checkpoint.tuner.file_path,
                            # monitor = "val_loss",
                            # mode = "auto",
                            save_best_only = TRUE,
                            verbose = 1))

cnn_mcc.tuner |> fit_tuner(x = x_train,
                           y = y_train,
                           callbacks = cnn_mcc.callbacks,
                           validation_split = 0.2,
                           epochs = 100)

# cnn_mcc.tuner$search(x = x_train,
#                      y = y_train,
#                      callbacks = cnn_mcc.callbacks,
#                      validation_split = 0.2,
#                      epochs = 30)

### CNN MCC Model Tuning Results Summary ---------------------------------------
put_log("The Model Tuning Results Summary:
%1", capture.output(cnn_mcc.tuner$results_summary()))
{
# > cnn_mcc.tuner$results_summary()
# Results summary
# Results in data/models/dl.keras3/cnn/multiclass/tuner\CNN-MCC.Tuning
# Showing 10 best trials
# Objective(name="val_accuracy", direction="max")
# 
# Trial 0234 summary
# Hyperparameters:
#   conv_blocs: 5
# filters_1: 192
# filters_2: 64
# filters_3: 224
# dropout2: 0.5
# hidden_size: 448
# dropout1: 0.5
# learning_rate: 0.003937767129539985
# filters_4: 128
# filters_5: 32
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 2
# tuner/round: 2
# tuner/trial_id: 0228
# Score: 0.8739064931869507
# 
# Trial 0488 summary
# Hyperparameters:
#   conv_blocs: 4
# filters_1: 224
# filters_2: 32
# filters_3: 160
# dropout2: 0.5
# hidden_size: 320
# dropout1: 0.2
# learning_rate: 0.002904610757828663
# filters_4: 32
# filters_5: 128
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 2
# tuner/round: 2
# tuner/trial_id: 0483
# Score: 0.8739064931869507
# 
# Trial 0389 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 128
# filters_2: 96
# filters_3: 160
# dropout2: 0.4
# hidden_size: 416
# dropout1: 0.5
# learning_rate: 0.00518044142551884
# filters_4: 224
# filters_5: 256
# tuner/epochs: 12
# tuner/initial_epoch: 4
# tuner/bracket: 4
# tuner/round: 2
# tuner/trial_id: 0357
# Score: 0.8736048340797424
# 
# Trial 0401 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 128
# filters_2: 96
# filters_3: 160
# dropout2: 0.4
# hidden_size: 416
# dropout1: 0.5
# learning_rate: 0.00518044142551884
# filters_4: 224
# filters_5: 256
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 4
# tuner/round: 4
# tuner/trial_id: 0396
# Score: 0.8736048340797424
# 
# Trial 0235 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 160
# filters_2: 256
# filters_3: 224
# dropout2: 0.5
# hidden_size: 464
# dropout1: 0.30000000000000004
# learning_rate: 0.007950871082139998
# filters_4: 32
# filters_5: 192
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 2
# tuner/round: 2
# tuner/trial_id: 0230
# Score: 0.8720965385437012
# 
# Trial 0400 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 32
# filters_2: 224
# filters_3: 224
# dropout2: 0.2
# hidden_size: 432
# dropout1: 0.5
# learning_rate: 0.0035013676680471597
# filters_4: 96
# filters_5: 256
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 4
# tuner/round: 4
# tuner/trial_id: 0398
# Score: 0.8720965385437012
# 
# Trial 0142 summary
# Hyperparameters:
#   conv_blocs: 4
# filters_1: 224
# filters_2: 192
# filters_3: 128
# dropout2: 0.4
# hidden_size: 432
# dropout1: 0.30000000000000004
# learning_rate: 0.004840503032482819
# filters_4: 256
# filters_5: 64
# tuner/epochs: 34
# tuner/initial_epoch: 12
# tuner/bracket: 4
# tuner/round: 3
# tuner/trial_id: 0134
# Score: 0.8717948794364929
# 
# Trial 0245 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 224
# filters_2: 224
# filters_3: 160
# dropout2: 0.5
# hidden_size: 496
# dropout1: 0.2
# learning_rate: 0.007272185647755973
# filters_4: 128
# filters_5: 32
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 1
# tuner/round: 1
# tuner/trial_id: 0241
# Score: 0.8717948794364929
# 
# Trial 0398 summary
# Hyperparameters:
#   conv_blocs: 2
# filters_1: 32
# filters_2: 224
# filters_3: 224
# dropout2: 0.2
# hidden_size: 432
# dropout1: 0.5
# learning_rate: 0.0035013676680471597
# filters_4: 96
# filters_5: 256
# tuner/epochs: 34
# tuner/initial_epoch: 12
# tuner/bracket: 4
# tuner/round: 3
# tuner/trial_id: 0388
# Score: 0.8717948794364929
# 
# Trial 0483 summary
# Hyperparameters:
#   conv_blocs: 4
# filters_1: 224
# filters_2: 32
# filters_3: 160
# dropout2: 0.5
# hidden_size: 320
# dropout1: 0.2
# learning_rate: 0.002904610757828663
# filters_4: 32
# filters_5: 128
# tuner/epochs: 34
# tuner/initial_epoch: 12
# tuner/bracket: 2
# tuner/round: 1
# tuner/trial_id: 0476
# Score: 0.8708899021148682
invisible()
}
### CNN MCC Model Tuning Results: Best Trial Summary ---------------------------

# This prints the top trials, their hyperparameters, and execution details
put_log("CNN MCC Model Tuning Results, Best Trial Summary:
%1", capture.output(results_summary(cnn_mcc.tuner,
                                    num_trials = 1L)))
# Results summary
# Results in data/models/dl.keras3/cnn/multiclass/tuner\CNN-MCC.Tuning
# Showing 1 best trials
# Objective(name="val_accuracy", direction="max")
# 
# Trial 0234 summary
# Hyperparameters:
#   conv_blocs: 5
# filters_1: 192
# filters_2: 64
# filters_3: 224
# dropout2: 0.5
# hidden_size: 448
# dropout1: 0.5
# learning_rate: 0.003937767129539985
# filters_4: 128
# filters_5: 32
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 2
# tuner/round: 2
# tuner/trial_id: 0228
# Score: 0.8739064931869507

### CNN MCC Model Tuning Results Visualization ---------------------------------
# This prints a summary of the search space and lists the top trial results
cnn_mcc.tuner.result <- kerastuneR::plot_tuner(cnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(cnn_mcc.tuner.result))

### Retrieving the Best Model --------------------------------------------------

# Retrieve the best model from the search

# put_log("Loading the CNN MCC Model Tuner object...")
# cnn_mcc.tuner <- readRDS(cnn_mcc.best_model.file)
# 
# put_log("The CNN MCC Model Tuner object has been loaded from the following file:
#   %1", cnn_mcc.best_model.file)
# put_end_date(start)

cnn_mcc.tuner

class(cnn_mcc.tuner)
# [1] "keras_tuner.src.tuners.hyperband.Hyperband"  "keras_tuner.src.engine.tuner.Tuner"         
# [3] "keras_tuner.src.engine.base_tuner.BaseTuner" "keras_tuner.src.engine.stateful.Stateful"   
# [5] "python.builtin.object"                      

cnn_mcc.best_models <- kerastuneR::get_best_models(tuner = cnn_mcc.tuner, num_models = 1L)
cnn_mcc.best_model <- cnn_mcc.best_models[[1]]
rm(cnn_mcc.best_models)

put_log("Saving the CNN MCC Best Model...")
keras3::save_model(cnn_mcc.best_model,
                   file = cnn_mcc.best_model.file,
                   overwrite = TRUE)

put_log("The CNN MCC Best Model object has been saved in the following file:
  %1", cnn_mcc.best_model.file)
put_end_date(start)

cnn_mcc.best_model$summary()
# View completed epochs of this best model
# If restore_best_weights = TRUE, this tells you the optimal epoch
# best_epoch <- cnn_mcc.best_model$history$params$epochs

cnn_mcc.best_model |> plot_keras_model(to_file = cnn_mcc.tuner.best.plot_img.file,
                                        show_shapes = TRUE)

cnn_mcc.tuner.best_trials <- cnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
cnn_mcc.best_trial <- cnn_mcc.tuner.best_trials[[1]]
cnn_mcc.best_trial$summary()
cnn_mcc.best_trial$best_step

cnn_mcc.best_trial$metrics$get_history('val_accuracy')

# rm(cnn_mcc.tuner.best_trials,
#    cnn_mcc.best_trial)

### Re-training the Final Tuned CNN MCC Model ----------------------------------
stopifnot(file.exists(cnn_mcc_final.retrain.script.path))

put_log("Re-training the CNN-based Multiclass Classifier Model...")

# source(cnn_mcc_final.retrain.script.path, 
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

log_close()
# Log Elapsed Time: 18:30:48













