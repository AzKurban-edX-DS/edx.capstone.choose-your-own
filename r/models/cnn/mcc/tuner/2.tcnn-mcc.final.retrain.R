#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%
# CNN MCC  Model Tuning: Retrain the Best Model
#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".setup.cnn_mcc.retrain-best")
start <- put_start_date()

stopifnot(exists("cnn_mcc.tuner"),
          file.exists(train.img28x28mx.array.file_path),
          dir.exists(data.cnn_mcc.tuner.best.dir),
          exists("cnn_mcc.final.file"))



### Prepare a Training Set for the Model Training ---------------------------------
put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")

split3d.list <- split.img28x28mx_array(train.img28x28mx.array.file_path,
                                       test_ratio = 0.2)

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
y_train <- y.train.groups$classID

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

#### Size of the Training Set by Class ----------------------------------------
put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
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

rm(y.train.groups)

log_close()

### Init File Paths ------------------------------------------------------------
open_logfile(".cnn_mcc.retrain-best")

cnn_mcc.tuner.final.plot_img.file <- file.path(data.cnn_mcc.tuner.best.dir,
                                              "cnn-mcc.tuner.final-model.png")

data.cnn_mcc.tuner.best.checkpoints.dir <- file.path(data.cnn_mcc.tuner.best.dir,
                                                     "checkpoints")

if(!dir.exists(data.cnn_mcc.tuner.best.checkpoints.dir))
  dir.create(data.cnn_mcc.tuner.best.checkpoints.dir)
  
cnn_mcc.best.checkpoint.file <- 
  file.path(data.cnn_mcc.tuner.best.checkpoints.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

## Re-training the Best Model --------------------------------------------------

cnn_mcc.tuner.best_hps <- cnn_mcc.tuner$get_best_hyperparameters(num_trials = 10L)
str(cnn_mcc.tuner.best_hps)

# Inspect the values of the top configuration
# cnn_mcc.tuner.best_hps[[1]]$values
cnn_mcc.tuner.best_hp <- cnn_mcc.tuner.best_hps[[1]]

put_log("The best Hyperparameters configuration:
%1", capture.output(cnn_mcc.tuner.best_hp$get_config()))

# class(cnn_mcc.tuner.best_hp)
# [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
# [2] "python.builtin.object"        

#cnn_mcc.tuner.best_hp
# <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>

put_log("The best Hyperparameters values:
%1", capture.output(cnn_mcc.tuner.best_hp$values))

# 1. Re-build a clean model structure using the winning hyperparams
cnn_mcc.final <- cnn_mcc.tuner$hypermodel$build(cnn_mcc.tuner.best_hp)
# print(cnn_mcc.final)
# cnn_mcc.final$summary()

put_log("The Final tuned tuned Final Model Summary: 
%1", capture.output(cnn_mcc.final))

cnn_mcc.final |> plot_keras_model(to_file = cnn_mcc.tuner.final.plot_img.file,
                                        show_shapes = T)

#best_models <- tuner |> get_best_models(num_models = 1L)
# best_5_models[[1]] %>% plot_keras_model()

cnn_mcc.best.callbacks <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = cnn_mcc.best.checkpoint.file,
                            monitor = "val_loss",
                            save_best_only = TRUE,
                            verbose = 1))

put_log("Training the tuned Final MCC Model...")
start <- put_start_date()

cnn_mcc.final.train_history <- cnn_mcc.final |> 
  fit(x_train, 
      y_train.cat, 
      epochs = 100, 
      # batch_size = 128, 
      callbacks = cnn_mcc.best.callbacks,
      validation_split = 0.2
  )

put_log("Saving re-trained final tuned Final MCC Model...")
keras3::save_model(cnn_mcc.final,
                   filepath = cnn_mcc.final.file,
                   overwrite = TRUE)

put_log("The re-trained final tuned Final MCC Model has been trained 
and saved in the following file:
  %1", cnn_mcc.final.file)

put_log("Saving the tuned Final MCC Model History...")
saveRDS(cnn_mcc.final.train_history,
        file = cnn_mcc.final.train_history.file)

put_log("The re-trained final tuned Final MCC Model History has been trained 
and saved in the following file:
  %1", cnn_mcc.final.train_history.file)
put_end_date(start)
# Time difference of 38.48235 mins

# rm(x_train,
#    y_train,
#    cnn_mcc.tuner)

put_log("The re-trained `tuned Final MCC` Model has been trained with the following results
%1", cnn_mcc.final)

plot(cnn_mcc.final.train_history)
str(cnn_mcc.final.train_history)

# rm(cnn_mcc.final.train_history)

### Evaluating the Re-trained Model --------------------------------------------

stopifnot(file.exists(tcnn_mcc.final.eval.script.path))

# source(tcnn_mcc.final.eval.script.path, 
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

log_close()
# Log Elapsed Time: 0 00:13:05
