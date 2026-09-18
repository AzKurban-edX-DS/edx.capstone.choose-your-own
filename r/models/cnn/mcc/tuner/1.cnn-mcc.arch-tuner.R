#%%%%%%%%%%%%%%%%%%%%%
# CNN MCC Model Tuning
#%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.model-tuning.architecture")
stopifnot(exists("cnn_mcc.tuner.dir"),
          exists("tcnn_mcc.arch.best_hp.config.file"))

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

## Tuning the Model Architecture -----------------------------------------------

tuner.proj.dir <- file.path(cnn_mcc.tuner.dir, 
                                         '1.arch-tuning.prj')

tuner.checkpoints.dir <- file.path(tuner.proj.dir, 
                                                "checkpoints")

tuner.best_model.plot.img_file <- file.path(tuner.proj.dir,
                                                         paste0('arch-tuned.best-model.plot', 
                                                                '.png'))

### Process the Tuning --------------------------------------------------------

hp <- HyperParameters()

hp$Choice('conv_blocks', c(2L, 3L))
hp$Choice('conv_padding', c('same', 'valid'))

# for (i in 1:3) {
#   hp$Int(paste0('conv', i, '_filters'),
#          min_value = 32L,
#          max_value = 256L,
#          step = 32L)
#   
#   hp$Choice(paste0('conv', i,'_kernel.size'), c(2L, 3L, 4L, 5L))
# }

dense_units <- hp$Int('dense_units',
                      min_value = 128L,
                      max_value = 512L,
                      step = 64L)

dropout1 <- hp$Float('dropout1',
                       min_value = 0.2,
                       max_value = 0.5,
                       step = 0.05)

dropout2 <- hp$Float('dropout2',
                       min_value = 0.2,
                       max_value = 0.5,
                       step = 0.05)

hp$Fixed("learning_rate", value = 1e-4)

tuner.result <- 
  cnn_mcc.Hyperband.fit_tuner(hp,
                              x_train,
                              y_train,
                              validation.data = tuple(x_test, y_test),
                              project.dir = tuner.proj.dir,
                              project.name = 'tuner.dat',
                              checkpoints.dir = tuner.checkpoints.dir)


if(!is.null(tuner.result$error) ||
   !is.null(tuner.result$hypermodel$error)) {
  put_log("Some error(s) occurred while tuning.")
  
  if(!is.null(tuner.result$error))
    put_log(tuner.result$error)
  
  if(!is.null(tuner.result$hypermodel$error))
    put_log(tuner.result$hypermodel$error)
}

tuner <- tuner.result$tuner

# This prints a summary of the search space and lists the top trial results
tuning_result <- kerastuneR::plot_tuner(tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(tuning_result))
rm(tuning_result)

tuner$results_summary()
{
  invisible()
}

tuner.best_hp <- 
  tuner$get_best_hyperparameters(num_trials = 1L)[[1]]

put_log("The best Hyperparameters values:
%1", capture.output(tuner.best_hp$values))
{
  # $conv_blocks
  # [1] 2
  # 
  # $conv_padding
  # [1] "same"
  # 
  # $dense_units
  # [1] 320
  # 
  # $dropout1
  # [1] 0.3
  # 
  # $dropout2
  # [1] 0.2
  # 
  # $learning_rate
  # [1] 1e-04
  # 
  # $conv1_filters
  # [1] 224
  # 
  # $conv1_kernel.size
  # [1] 5
  # 
  # $conv2_filters
  # [1] 224
  # 
  # $conv2_kernel.size
  # [1] 5
  # 
  # $conv3_filters
  # [1] 256
  # 
  # $conv3_kernel.size
  # [1] 4
  # 
  # $`tuner/epochs`
  # [1] 10
  # 
  # $`tuner/initial_epoch`
  # [1] 4
  # 
  # $`tuner/bracket`
  # [1] 1
  # 
  # $`tuner/round`
  # [1] 1
  # 
  # $`tuner/trial_id`
  # [1] "0052"  
  
  invisible()
}

best_hp.config <- tuner.best_hp$get_config()
put_log("The best Hyperparameters configuration:
%1", capture.output(best_hp.config))


put_log("Saving the Best Hyper-parameter Configuration...")
saveRDS(best_hp.config,
        file = tcnn_mcc.arch.best_hp.config.file)

rm(best_hp.config)

put_log("The Best Hyperparameters Configuration have been saved to the following file:
%1", tcnn_mcc.arch.best_hp.config.file)


best_trial <- 
  tuner$oracle$get_best_trials(num_trials = 1L)[[1]]

put_log("The best step of the best trial: %1", best_trial$best_step)
# 9

best_trial$summary()
# Trial 0054 summary
# Hyperparameters:
# conv_blocks: 2
# conv_padding: same
# dense_units: 320
# dropout1: 0.30000000000000004
# dropout2: 0.2
# learning_rate: 0.0001
# conv1_filters: 224
# conv1_kernel.size: 5
# conv2_filters: 224
# conv2_kernel.size: 5
# conv3_filters: 256
# conv3_kernel.size: 4
# tuner/epochs: 10
# tuner/initial_epoch: 4
# tuner/bracket: 1
# tuner/round: 1
# tuner/trial_id: 0052
# Score: 0.8752390742301941

rm(best_trial)

#### Retrieving the Best Model ------------------------------------------------

#best_model <- tcnn_mcc.best_models[[1]]
# rm(tcnn_mcc.best_models)


best_model <- 
  kerastuneR::get_best_models(tuner = tuner, 
                              num_models = 1L)[[1]]

best_model |> plot_keras_model(to_file = cnn_mcc.lr_tuner.best_model.plot.img_file,
                               show_shapes = T)

# put_log("Saving the CNN MCC Best Model...")
# keras3::save_model(tcnn_mcc.best_model,
#                    file = best_model.file,
#                    overwrite = TRUE)

# put_log("The CNN MCC Best Model object has been saved in the following file:
#   %1", best_model.file)

rm(best_mode)
#### (Alternatively) Building the model from the Best Hyper-parameters ---------

best_hp.model <- 
  tuner.result$hypermodel$build(tuner.best_hp)

put_log("Summary of the Tuned Model built from the best hyper-parameters:
%1", capture.output(best_hp.model))

rm(best_hp.model)

log_close()
# =========================================================================
# Log End Time: 2026-09-17 15:52:15.277699
# Log Elapsed Time: 0 06:03:02
# =========================================================================
