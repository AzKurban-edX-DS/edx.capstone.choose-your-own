#%%%%%%%%%%%%%%%%%%%%%
# CNN MCC Model Tuning
#%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.model-tuning")

stopifnot(file.exists(train.img28x28mx.array.file_path))

start <- put_start_date()

## Prepare Input Datasets for the DNN MCC Model Tuning -------------------------

put_log("Loading the Input Datasets of 28x28-size image data...")
ds <- load28x28x1.datasets(ds28x28.split.train_0.1.backup.file)
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
shape(x_train)

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

rm(y.test.groups)




## Model Tuning ----------------------------------------------------------------

### Init the Model Tuner Paths -------------------------------------------------

cnn_mcc.tuner.prj1.dir <- file.path(cnn_mcc.tuner.dir,"project1")

tcnn_mcc.best_model.file <- file.path(cnn_mcc.tuner.prj1.dir, 
                                     "best-model.keras")

tcnn_mcc.best_model.plot_img.file <- file.path(cnn_mcc.tuner.prj1.dir,
                                               "best-model.png")

cnn_mcc.tuner.checkpoints.lr1e_4.dir <- file.path(cnn_mcc.tuner.prj1.dir, 
                                                  "checkpoints")
if(!dir.exists(cnn_mcc.tuner.prj1.dir))
  dir.create(cnn_mcc.tuner.prj1.dir)

if(!dir.exists(cnn_mcc.tuner.checkpoints.lr1e_4.dir))
  dir.create(cnn_mcc.tuner.checkpoints.lr1e_4.dir)

cnn_mcc.tuner.checkpoints.file_path <- 
  file.path(cnn_mcc.tuner.checkpoints.lr1e_4.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

### Process the Tuning ---------------------------------------------------------

cnn_mcc.max_conv_blocks = 2

cnn_mcc.tuners <- list()
cnn_mcc.tuners[[1]] <- NULL

for(i in 2:cnn_mcc.max_conv_blocks) {
  cnn_mcc.hypermodel <- CNN_MCC.HyperModel(num_classes = N.classes,
                                           conv_blocks = i)
  cnn_mcc.tuner <- Hyperband(cnn_mcc.hypermodel,
                             objective = 'val_accuracy',
                             # max_epochs = 100,
                             hyperband_iterations = 2,
                             directory = cnn_mcc.tuner.prj1.dir,
                             project_name = 'tuner.dat')
  
  tcnn_mcc.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = cnn_mcc.tuner.checkpoints.file_path,
                              # monitor = "val_loss",
                              # mode = "auto",
                              save_best_only = TRUE,
                              verbose = 1))
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  cnn_mcc.fit_tuner.result <- try({
    # Run the tuner fit process
    cnn_mcc.tuner |> fit_tuner(x = x_train,
                               y = y_train,
                               callbacks = tcnn_mcc.callbacks,
                               # validation_split = 0.2,
                               validation_data = tuple(x_test, y_test),
                               epochs = 100L)
  }, silent = T)
  
  stopCluster(cl)
  stopImplicitCluster()
  
  if("try-error" %in% class(add_block.result)) {
    put_log("Failed to tune the model with %1 Convolution Blocks.", i)
    put_log(cnn_mcc.fit_tuner.result)
    
    put_log("The model with %1 Convolution Blocks HAS NOT BEEN TUNED.", i)
    break
  }
  
  if(!is.null(cnn_mcc.hypermodel$error)) break
  
  cnn_mcc.tuners[[i]] <- cnn_mcc.tuner

  # This prints a summary of the search space and lists the top trial results
  cnn_mcc.tuner.result <- kerastuneR::plot_tuner(cnn_mcc.tuner)
  # the list will show the plot and the data.frame of tuning results
  
  put_log("The CNN MCC Tuning Results:
%1", capture.output(cnn_mcc.tuner.result))
  
  class(cnn_mcc.tuner)
  # [1] "keras_tuner.src.tuners.hyperband.Hyperband"  "keras_tuner.src.engine.tuner.Tuner"         
  # [3] "keras_tuner.src.engine.base_tuner.BaseTuner" "keras_tuner.src.engine.stateful.Stateful"   
  # [5] "python.builtin.object"                      
  
  tcnn_mcc.best_trials <- cnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
  tcnn_mcc.best_trial <- tcnn_mcc.best_trials[[1]]
  tcnn_mcc.best_trial$summary()
  tcnn_mcc.best_trial$best_step
  
}


### Tuning Results Summary -----------------------------------------------------
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

#### Tuning Results: Best Trial Summary ----------------------------------------

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

### Tuning Results Visualization -----------------------------------------------
# This prints a summary of the search space and lists the top trial results
cnn_mcc.tuner.result <- kerastuneR::plot_tuner(cnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(cnn_mcc.tuner.result))

### Retrieving the Best Model --------------------------------------------------

cnn_mcc.tuner

tcnn_mcc.best_models <- kerastuneR::get_best_models(tuner = cnn_mcc.tuner, num_models = 1L)
tcnn_mcc.best_model <- tcnn_mcc.best_models[[1]]
rm(tcnn_mcc.best_models)

put_log("Saving the CNN MCC Best Model...")
keras3::save_model(tcnn_mcc.best_model,
                   file = tcnn_mcc.best_model.file,
                   overwrite = TRUE)

put_log("The CNN MCC Best Model object has been saved in the following file:
  %1", tcnn_mcc.best_model.file)
put_end_date(start)

tcnn_mcc.best_model$summary()
# View completed epochs of this best model
# If restore_best_weights = TRUE, this tells you the optimal epoch
# best_epoch <- tcnn_mcc.best_model$history$params$epochs

tcnn_mcc.best_model |> plot_keras_model(to_file = tcnn_mcc.best_model.plot_img.file,
                                        show_shapes = TRUE)

tcnn_mcc.best_trials <- cnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
tcnn_mcc.best_trial <- tcnn_mcc.best_trials[[1]]
tcnn_mcc.best_trial$summary()
tcnn_mcc.best_trial$best_step

tcnn_mcc.best_trial$metrics$get_history('val_accuracy')

### Extract & Save the Best Hyper-parameter Configuration ----------------------

cnn_mcc.tuner.best_hp.ls <- cnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)
# str(cnn_mcc.tuner.best_hp.ls)

cnn_mcc.tuner.best_hp <- cnn_mcc.tuner.best_hp.ls[[1]]

put_log("The best Hyperparameters values:
%1", capture.output(cnn_mcc.tuner.best_hp$values))

tcnn_mcc.best_hp.config <- cnn_mcc.tuner.best_hp$get_config()
put_log("The best Hyperparameters configuration:
%1", capture.output(tcnn_mcc.best_hp.config))

put_log("Saving the Best Hyper-parameter Configuration...")
saveRDS(tcnn_mcc.best_hp.config,
        file = tcnn_mcc.best_hp.config.file)

put_log("The Best Hyper-parameter Configuration has been saved in the following file:
  %1", tcnn_mcc.best_hp.config.file)


## Finalizing ------------------------------------------------------------------

# rm(tcnn_mcc.best_trials,
#    tcnn_mcc.best_trial)

put_end_date(start)

log_close()
# Log Elapsed Time: 18:30:48
