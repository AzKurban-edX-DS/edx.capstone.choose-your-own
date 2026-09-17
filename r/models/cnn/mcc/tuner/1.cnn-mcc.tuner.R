#%%%%%%%%%%%%%%%%%%%%%
# CNN MCC Model Tuning
#%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.model-tuning.prepare-datasets")
stopifnot(file.exists(ds28x28.split.train_0.1.backup.file))

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


log_close()

## Model Tuning ----------------------------------------------------------------


# cnn_mcc.max_conv_blocks = 2

cnn_mcc.tuners <- list()
cnn_mcc.tuners[[1]] <- NULL


### Tune the Model Architecture --------------------------------------------

open_logfile(".cnn_mcc.model-tuning.architecture")


cnn_mcc.tuner.proj.arch.dir <- file.path(cnn_mcc.tuner.dir, 
                                         'proj.architecture')

cnn_mcc.tuner.checkpoints.dir <- file.path(cnn_mcc.tuner.proj.arch.dir, 
                                           "checkpoints")

hp <- HyperParameters()



hp$Choice('conv_blocks', c(2L, 3L))
hp$Choice('conv_padding', c('same', 'valid'))

for (i in 1:3) {
  hp$Int(paste0('conv', i, '_filters'),
         min_value = 32L,
         max_value = 128L,
         step = 32L)
  
  hp$Choice(paste0('conv', i,'_kernel.size'), c(3L, 5L))
}




dense_units <- hp$Int('dense_units',
                      min_value = 128L,
                      max_value = 512L,
                      step = 64L)

# dropout1 <- hp$Float('dropout1',
#                        min_value = 0.2,
#                        max_value = 0.5,
#                        step = 0.05,
#                        default = 0.25)
# 
dropout1 <- hp$Float('dropout1',
                       min_value = 0.2,
                       max_value = 0.5,
                       step = 0.1)

dropout2 <- hp$Float('dropout2',
                       min_value = 0.2,
                       max_value = 0.5,
                       step = 0.1)

hp$Fixed("learning_rate", value = tcnn_mcc.best_lr)


# hp$Fixed("conv1_filters", value = conv1_filters)
# hp$Int("conv1_filters", value = conv1_filters)



hp$Int("conv2_filters", 
       min_value = conv1_filters,
       max_value = conv_filters.max,
       step = 32L)

cnn_mcc.tune_result <- cnn_mcc.Hyperband.fit_tuner(hp,
                                                   x_train,
                                                   y_train,
                                                   validation.data = tuple(x_test, y_test),
                                                   project.dir = cnn_mcc.tuner.proj.arch.dir,
                                                   project.name = 'tuner.dat',
                                                   checkpoints.dir = cnn_mcc.tuner.checkpoints.dir)


if(!is.null(cnn_mcc.tune_result$error) ||
   !is.null(cnn_mcc.tune_result$hypermodel$error)) {
  put_log("Some error(s) occurred while tuning.")
  
  if(!is.null(cnn_mcc.tune_result$error))
    put_log(cnn_mcc.tune_result$error)
  
  if(!is.null(cnn_mcc.tune_result$hypermodel$error))
    put_log(cnn_mcc.tune_result$hypermodel$error)
}

cnn_mcc.tuner <- cnn_mcc.tune_result$tuner
cnn_mcc.tuners$tuned_by.learning_rate <- cnn_mcc.tuner

# This prints a summary of the search space and lists the top trial results
cnn_mcc.tuner.result <- kerastuneR::plot_tuner(cnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(cnn_mcc.tuner.result))


# class(cnn_mcc.tuner)
# [1] "keras_tuner.src.tuners.hyperband.Hyperband"  "keras_tuner.src.engine.tuner.Tuner"         
# [3] "keras_tuner.src.engine.base_tuner.BaseTuner" "keras_tuner.src.engine.stateful.Stateful"   
# [5] "python.builtin.object"                      

# tcnn_mcc.best_trials <- cnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
# tcnn_mcc.best_trial <- tcnn_mcc.best_trials[[1]]
# tcnn_mcc.best_trial$summary()

cnn_mcc.tuner$results_summary()
{
  invisible()
}

put_log("The best step of the best trial: %1", tcnn_mcc.best_trial$best_step)

cnn_mcc.tuner.best_hp <- 
  cnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)[[1]]

put_log("The best Hyperparameters values:
%1", capture.output(cnn_mcc.tuner.best_hp$values))

tcnn_mcc.best_c2filters <- cnn_mcc.tuner.best_hp$values$conv2_filters
# 0.003610324

log_close()
### Tune `learning rate` parameter ---------------------------------------------

open_logfile(".cnn_mcc.model-tuning.learning-rate")

cnn_mcc.tuner.proj.dir <- file.path(cnn_mcc.tuner.dir, 
                                    paste0('proj.', 
                                           n.cnv_blocks, 
                                           'conv-blocks.lr'))

tcnn_mcc.best_model.file <- file.path(cnn_mcc.tuner.proj.dir, 
                                      paste0('best-model.', 
                                             n.cnv_blocks, 
                                             'cb.lr', 
                                             '.keras'))

tcnn_mcc.best_model.plot_img.file <- file.path(cnn_mcc.tuner.proj.dir,
                                               paste0('best-model.', 
                                                      n.cnv_blocks, 
                                                      'cb.lr', 
                                                      '.png'))

cnn_mcc.tuner.checkpoints.dir <- file.path(cnn_mcc.tuner.proj.dir, 
                                           "checkpoints")

hp$Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")


cnn_mcc.tune_result <- cnn_mcc.Hyperband.fit_tuner(x_train,
                                                   y_train,
                                                   validation.data = tuple(x_test, y_test),
                                                   project.dir = cnn_mcc.tuner.proj.dir,
                                                   project.name = 'tuner.dat',
                                                   best_model.file = tcnn_mcc.best_model.file,
                                                   best_model.plot_img.file = tcnn_mcc.best_model.plot_img.file,
                                                   checkpoints.dir = cnn_mcc.tuner.checkpoints.dir,
                                                   num.classes = N.classes,
                                                   tune.new_entries = FALSE,
                                                   hp = hp,
                                                   conv_blocks = n.cnv_blocks,
                                                   cnvFilters.min = 32L)


if(!is.null(cnn_mcc.tune_result$error) ||
   !is.null(cnn_mcc.tune_result$hypermodel$error)) {
  put_log("Some error(s) occurred while tuning.")
  
  if(!is.null(cnn_mcc.tune_result$error))
    put_log(cnn_mcc.tune_result$error)
  
  if(!is.null(cnn_mcc.tune_result$hypermodel$error))
    put_log(cnn_mcc.tune_result$hypermodel$error)
}

cnn_mcc.tuner <- cnn_mcc.tune_result$tuner
cnn_mcc.tuners$tuned_by.learning_rate <- cnn_mcc.tuner

# This prints a summary of the search space and lists the top trial results
cnn_mcc.tuner.result <- kerastuneR::plot_tuner(cnn_mcc.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(cnn_mcc.tuner.result))


# class(cnn_mcc.tuner)
# [1] "keras_tuner.src.tuners.hyperband.Hyperband"  "keras_tuner.src.engine.tuner.Tuner"         
# [3] "keras_tuner.src.engine.base_tuner.BaseTuner" "keras_tuner.src.engine.stateful.Stateful"   
# [5] "python.builtin.object"                      

cnn_mcc.tuner$results_summary()
{
# Trial 0488 summary
# Hyperparameters:
# learning_rate: 0.0036103239402985906
# tuner/epochs: 100
# tuner/initial_epoch: 34
# tuner/bracket: 2
# tuner/round: 2
# dense_units: 128
# dropout1: 0.25
# dropout2: 0.5
# conv1_filter: 32
# conv1_kernel.size: 3
# conv2_filter: 64
# conv2_kernel.size: 3
# tuner/trial_id: 0482
# Score: 0.9034014344215393
invisible()
}

# tcnn_mcc.best_trials <- cnn_mcc.tuner$oracle$get_best_trials(num_trials = 1L)
# tcnn_mcc.best_trial <- tcnn_mcc.best_trials[[1]]
# tcnn_mcc.best_trial$summary()

# put_log("The best step of the best trial: %1", tcnn_mcc.best_trial$best_step)

cnn_mcc.tuner.best_hp <- 
  cnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)[[1]]

put_log("The best Hyperparameters values:
%1", capture.output(cnn_mcc.tuner.best_hp$values))

tcnn_mcc.best_lr <- cnn_mcc.tuner.best_hp$values$learning_rate
tcnn_mcc.best_lr
# 0.003610324

log_close()
# =========================================================================
# Log End Time: 2026-09-14 17:59:39.159371
# Log Elapsed Time: 0 11:45:48
# =========================================================================


## Retrieving the Best Model --------------------------------------------------

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


log_close()

## Tuning Results: Best Trial Summary ----------------------------------------

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

## Extract & Save the Best Hyper-parameter Configuration ----------------------

# cnn_mcc.tuner.best_hp.ls <- cnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)
# str(cnn_mcc.tuner.best_hp.ls)
# cnn_mcc.tuner.best_hp <- cnn_mcc.tuner.best_hp.ls[[1]]

cnn_mcc.tuner.best_hp <- 
  cnn_mcc.tuner$get_best_hyperparameters(num_trials = 1L)[[1]]

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
