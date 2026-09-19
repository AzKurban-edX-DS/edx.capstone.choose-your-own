#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%
# CNN MCC  Model Tuning: Retrain the Best Model
#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.retrain-final")
start <- put_start_date()

stopifnot(file.exists(ds28x28.split.train_0.8.backup.file))

## Prepare a Training Set -----------------------------------------------------

put_log("Loading the Training Set of 28x28x1-shape image data...")

train_set <- load28x28x1.train_set(ds28x28.split.train_0.8.backup.file)
put_log("The Training Set of 28x28x1-shape image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)

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

rm(train_set)

## Init File Paths ------------------------------------------------------------

cnn_mcc.tuner.final.plot_img.file <- file.path(cnn_mcc.tuner.dir,
                                              "cnn-mcc.tuner.final-model.png")

cnn_mcc.tuner.final.checkpoints.dir <- file.path(cnn_mcc.tuner.dir,
                                                     "checkpoints.final")

if(!dir.exists(cnn_mcc.tuner.final.checkpoints.dir))
  dir.create(cnn_mcc.tuner.final.checkpoints.dir)
  
cnn_mcc.best.checkpoint.file <- 
  file.path(cnn_mcc.tuner.final.checkpoints.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

## Re-training the Final Model --------------------------------------------------

put_log("Loading the Best Hyper-parameter Configuration from file...")
best_hp.config <- readRDS(tcnn_mcc.best_hp.config.file)

put_log("The Best Hyper-parameter Configuration has been loaded from the following file:
  %1", tcnn_mcc.best_hp.config.file)

# Build the HyperParameters object from the configuration
kt <- import("keras_tuner")
best_hp <- kt$HyperParameters$from_config(best_hp.config)
rm(best_hp.config)

put_log("The best Hyperparameters values:
%1", capture.output(best_hp$values))
{
  # The best Hyperparameters values:
  #   $learning_rate
  # [1] 0.007185949
  # 
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
  # [1] 15
  # 
  # $`tuner/initial_epoch`
  # [1] 5
  # 
  # $`tuner/bracket`
  # [1] 2
  # 
  # $`tuner/round`
  # [1] 2
  # 
  # $`tuner/trial_id`
  # [1] "0013"
  invisible()  
}

# 1. Re-build a clean model structure using the winning hyperparams
hypermodel <- CNN_MCC.HyperModel(N.classes)
cnn_mcc.final <- hypermodel$build(best_hp)
# print(cnn_mcc.final)
# cnn_mcc.final$summary()

put_log("The Final tuned tuned Final Model Summary: 
%1", capture.output(cnn_mcc.final))

cnn_mcc.best.callbacks <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = cnn_mcc.best.checkpoint.file,
                            monitor = "val_loss",
                            save_best_only = TRUE,
                            verbose = 1))

put_log("Training the tuned Final MCC Model...")
start <- put_start_date()

tcnn_mcc.final.train_history <- cnn_mcc.final |> 
  fit(x_train, 
      y_train, 
      epochs = 100, 
      # batch_size = 128, 
      callbacks = cnn_mcc.best.callbacks,
      validation_split = 0.2
  )

put_log("Saving re-trained final tuned Final MCC Model...")
keras3::save_model(cnn_mcc.final,
                   filepath = tcnn_mcc.final.file,
                   overwrite = TRUE)

put_log("The re-trained final tuned Final MCC Model has been trained 
and saved in the following file:
  %1", tcnn_mcc.final.file)

put_log("Saving the tuned Final MCC Model History...")
saveRDS(tcnn_mcc.final.train_history,
        file = tcnn_mcc.final.train_history.file)

put_log("The re-trained final tuned Final MCC Model History has been trained 
and saved in the following file:
  %1", tcnn_mcc.final.train_history.file)
put_end_date(start)
# Time difference of 38.48235 mins

# rm(x_train,
#    y_train, y_train.cat)

put_log("The re-trained `tuned Final MCC` Model has been trained with the following results
%1", cnn_mcc.final)

plot(tcnn_mcc.final.train_history)
str(tcnn_mcc.final.train_history)

# rm(tcnn_mcc.final.train_history)

### Evaluating the Re-trained Model --------------------------------------------

# stopifnot(file.exists(tcnn_mcc.final.eval.script.path))

# source(tcnn_mcc.final.eval.script.path, 
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

log_close()
# Log Elapsed Time: 0 00:13:05
