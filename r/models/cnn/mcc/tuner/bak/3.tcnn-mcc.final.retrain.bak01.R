#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%
# CNN MCC  Model Tuning: Retrain the Final Model
#%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.retrain-final")
start <- put_start_date()

stopifnot(file.exists(ds28x28.split.train_0.8.backup.file))

### Prepare a Training Set -----------------------------------------------------

put_log("Loading the Binary Image 28x28 array set from the backup file...")
ftrain_set <- readRDS(train.img28x28mx.array.file_path)
put_log("The Binary Image 28x28 array set has been loaded from the following file:
%1", train.img28x28mx.array.file_path)

x <- ftrain_set$img28x28mx.array
x.class_groups <- 

x_train <- array_reshape(x, 
                   c(nrow(x), 
                     28, 
                     28, 
                     1))
str(x_train)
shape(x_train)

y.train.groups <- ds.get_classIDs.grouped(x)
rm(x)

stopifnot(sum(as.character(y.train.groups$classID) != rownames(x_train)) == 0)

y_train <- as.array(as.integer(y.train.groups$classID) - 1)
str(y_train)
dim(y_train)

stopifnot(min(y_train) == 0,
          max(y_train) == 38,
          dim(y_train) == nrow(x_train))

y_train.cat <- to_categorical(y_train)
colnames(y_train.cat) <- Y.Labels

#### Size of the Training Set by Class -----------------------------------------
put_log("This imbalanced Training Set is prepared to use for the final retraining of the best model:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  # 1 #       15600
  # 2 $       16199
  # 3 &       13000
  # 4 @       38009
  # 5 0       65504
  # 6 1       43773
  # 7 2       39348
  # 8 3       39996
  # 9 4       38112
  # 10 5       32315
  # 11 6       38879
  # 12 7       41080
  # 13 8       38795
  # 14 9       38319
  # 15 A       17203
  # 16 B        8663
  # 17 C       13557
  # 18 D       15507
  # 19 E       32627
  # 20 F       11620
  # 21 G        5443
  # 22 H       12133
  # 23 I       13873
  # 24 J        4259
  # 25 K        4334
  # 26 L       21646
  # 27 M       12088
  # 28 N       21421
  # 29 P       11089
  # 30 Q        4706
  # 31 R       20498
  # 32 S       25910
  # 33 T       30842
  # 34 U       16385
  # 35 V        7245
  # 36 W        7265
  # 37 X        5106
  # 38 Y        6755
  # 39 Z        4863
  invisible(NULL)
}

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
