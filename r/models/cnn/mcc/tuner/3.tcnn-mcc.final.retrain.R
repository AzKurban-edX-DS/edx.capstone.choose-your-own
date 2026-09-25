#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%
# CNN MCC  Model Tuning: Retrain the Final Model
#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.retrain-final")
start <- put_start_date()

stopifnot(file.exists(ds.imbalanced.final_retraining.file))

### Prepare Input Datasets for Retraining the Final Model ----------------------

put_log("Loading the Input Datasets of 28x28-size image data...")
ds <- load28x28x1.datasets(ds.imbalanced.final_retraining.file)
train_set <- ds$train
test_set <- ds$test
rm(ds)

put_log("The Input Dataset of 28x28-size image data has been loaded from the following file:
%1", ds.imbalanced.final_retraining.file)

#### Prepare a Training Set -----------------------------------------------------


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

##### Size of the Training Set by Class -----------------------------------------

put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  # 1 #       12480
  # 2 $       12959
  # 3 &       10400
  # 4 @       30407
  # 5 0       52403
  # 6 1       35018
  # 7 2       31478
  # 8 3       31996
  # 9 4       30489
  # 10 5       25852
  # 11 6       31103
  # 12 7       32864
  # 13 8       31036
  # 14 9       30655
  # 15 A       13762
  # 16 B        6930
  # 17 C       10845
  # 18 D       12405
  # 19 E       26101
  # 20 F        9296
  # 21 G        4354
  # 22 H        9706
  # 23 I       11098
  # 24 J        3407
  # 25 K        3467
  # 26 L       17316
  # 27 M        9670
  # 28 N       17136
  # 29 P        8871
  # 30 Q        3764
  # 31 R       16398
  # 32 S       20728
  # 33 T       24673
  # 34 U       13108
  # 35 V        5796
  # 36 W        5812
  # 37 X        4084
  # 38 Y        5404
  # 39 Z        3890
  invisible(NULL)
}

rm(y.train.groups)

#### Prepare a Test Set ----------------------------------------------------------
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

##### Size of the Test Set by Class ------------------------------------------

put_log("The Test Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # 1 #        3120
  # 2 $        3240
  # 3 &        2600
  # 4 @        7602
  # 5 0       13101
  # 6 1        8755
  # 7 2        7870
  # 8 3        8000
  # 9 4        7623
  # 10 5        6463
  # 11 6        7776
  # 12 7        8216
  # 13 8        7759
  # 14 9        7664
  # 15 A        3441
  # 16 B        1733
  # 17 C        2712
  # 18 D        3102
  # 19 E        6526
  # 20 F        2324
  # 21 G        1089
  # 22 H        2427
  # 23 I        2775
  # 24 J         852
  # 25 K         867
  # 26 L        4330
  # 27 M        2418
  # 28 N        4285
  # 29 P        2218
  # 30 Q         942
  # 31 R        4100
  # 32 S        5182
  # 33 T        6169
  # 34 U        3277
  # 35 V        1449
  # 36 W        1453
  # 37 X        1022
  # 38 Y        1351
  # 39 Z         973
  invisible(NULL)
}

rm(y.test.groups)

### Init File Paths ------------------------------------------------------------

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
hypermodel <- CNN_MCC.HyperModel(N.classes,
                                 macro_f1_score = T)

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
      y_train.cat, 
      epochs = 100, 
      # batch_size = 128, 
      callbacks = cnn_mcc.best.callbacks,
      # validation_split = 0.2
      validation_data = tuple(x_test, y_test)
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

# rm(x_train,
#    y_train)

put_log("The re-trained `tuned Final MCC` Model has been trained with the following results
%1", cnn_mcc.final)

plot(tcnn_mcc.final.train_history)
str(tcnn_mcc.final.train_history)

log_close()
# =========================================================================
# Log End Time: 2026-09-22 05:51:02.922724
# Log Elapsed Time: 0 01:29:00
# =========================================================================
