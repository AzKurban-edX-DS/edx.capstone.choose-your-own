#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Re-train Tuned DNN-Based MCC (TDNN MCC) Final Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stopifnot(file.exists())

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


class(dnn_mcc.tuner.best_hp)
# [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
# [2] "python.builtin.object"        

dnn_mcc.tuner.best_hp
# <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>

dnn_mcc.best_hp.config <- dnn_mcc.tuner.best_hp$get_config()
put_log("The best Hyperparameters configuration:
%1", capture.output(dnn_mcc.best_hp.config))


put_log("The best Hyperparameters values:
%1", capture.output(dnn_mcc.tuner.best_hp$values))

# restored_hp <- HyperParameters$from_config(loaded_config)
#dnn_mcc.tuner.best_hp <- kerastuneR::HyperParameters$from_config(dnn_mcc.best_hp.config)

# Build the HyperParameters object from the configuration
kt <- import("keras_tuner")
dnn_mcc.tuner.best_hp <- kt$HyperParameters$from_config(dnn_mcc.best_hp.config)


# 1. Re-build a clean model structure using the winning hyperparams
# dnn_mcc.final_model <- dnn_mcc.tuner$hypermodel$build(dnn_mcc.tuner.best_hp)
  
dnn_mcc.final_model <- dnn_mcc.tuner.build_model(dnn_mcc.tuner.best_hp)

# print(dnn_mcc.final_model)
# dnn_mcc.final_model$summary()

# 2. Pass the restored object into your original model-building function
best_model <- build_model(restored_hp)

put_log("The Tuned DNN-Based Final Model Summary: 
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



