#%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model
#%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3

options(timeout = max(300, getOption("timeout")))

## Preparing Datasets for BDL MCC Model Tuning ---------------------------------
open_logfile(".prepare-dataset-for-dl.model-tuning")
start <- put_start_date()
# stopifnot(file.exists(my_emnist.split.file_path))
stopifnot(exists("x3d.train_set"))
stopifnot(exists("x3d.test_set"))

str(x3d.train_set)
str(x3d.test_set)

x.train <- x3d.train_set$x.train
# storage.mode(x.train) <- "integer"

# x.train <- x.train[seq(1e4),,]
str(x.train)
dim(x.train)

x.test <- x3d.test_set$x.test
# storage.mode(x.test) <- "integer"
dim(x.test)

x.test.files <- x3d.test_set$x.files

y.train.groups <- ds.get_classIDs.grouped(x.train)
y_train <- y.train.groups$classID

# y_train <- y_train[seq(1e4)]
stopifnot(sum(as.character(y_train) != rownames(x.train)) == 0)

y.train <- as.array(as.integer(y_train) - 1)
str(y.train)
dim(y.train)

stopifnot(min(y.train) == 0)
stopifnot(max(y.train) == 38)
stopifnot(dim(y.train) == nrow(x.train))

y.test.groups <- ds.get_classIDs.grouped(x.test)
y_test <- y.test.groups$classID

# y_test <- y_test[seq(1e4)]
stopifnot(sum(as.character(y_test) != rownames(x.test)) == 0)

y.test <- as.array(as.integer(y_test) - 1)
str(y.test)
dim(y.test)

stopifnot(min(y.test) == 0)
stopifnot(max(y.test) == 38)
stopifnot(dim(y.test) == nrow(x.test))


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
}

put_log("The Test Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         852
  #  2 $         852
  #  3 &         852
  #  4 @         852
  #  5 0         852
  #  6 1         852
  #  7 2         852
  #  8 3         852
  #  9 4         852
  # 10 5         852
  # 11 6         852
  # 12 7         852
  # 13 8         852
  # 14 9         852
  # 15 A         852
  # 16 B         852
  # 17 C         852
  # 18 D         852
  # 19 E         852
  # 20 F         852
  # 21 G         852
  # 22 H         852
  # 23 I         852
  # 24 J         852
  # 25 K         852
  # 26 L         852
  # 27 M         852
  # 28 N         852
  # 29 P         852
  # 30 Q         852
  # 31 R         852
  # 32 S         852
  # 33 T         852
  # 34 U         852
  # 35 V         852
  # 36 W         852
  # 37 X         852
  # 38 Y         852
  # 39 Z         852  
}

#> [1] 16653   784
str(x.train)
str(y.train)

dim(x.train)
dim(y.train)

str(x.test)
str(y.test)

dim(x.test)
dim(y.test)
#> [1] 817379


#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/


y.train.cat <- to_categorical(y.train)
colnames(y.train.cat) <- Y.Labels
dim(y.train.cat)
str(y.train.cat)
head(y.train.cat)
# max(y.train.cat)

y.test.cat <- to_categorical(y.test)
colnames(y.test.cat) <- Y.Labels
dim(y.test.cat)
str(y.test.cat)
head(y.test.cat)

log_close()

## Tuning the  BDL MCC Model ---------------------------------------------------
### Open Log File --------------------------------------------------------------
open_logfile(paste0(".dl-basic-model.tuning_", 
                    paste0("x.train(",
                           str_flatten(shape(x.train), 
                                       collapse = ","),")")))

### Init the Model Tuner Paths --------------------------------------------

if(!dir.exists(dl.keras3.path))
  dir.create(dl.keras3.path)

dl.basic.dir_path <- file.path(dl.keras3.path, "dl.basic")

if(!dir.exists(dl.basic.dir_path))
  dir.create(dl.basic.dir_path)

dl.basic.tuning.dir <- file.path(dl.basic.dir_path,
                                            "tuning")
if(!dir.exists(dl.basic.tuning.dir))
  dir.create(dl.basic.tuning.dir)

dl.basic.layers_dynamic.dir <- file.path(dl.basic.tuning.dir, "layers-dynamic")

if(!dir.exists(dl.basic.layers_dynamic.dir))
  dir.create(dl.basic.layers_dynamic.dir)

dl.basic.keras_tuner.dir <- file.path(dl.basic.tuning.dir, "keras-tuner")

if(!dir.exists(dl.basic.keras_tuner.dir))
  dir.create(dl.basic.keras_tuner.dir)

dl_basic.model_tuner.file_path <- file.path(dl.basic.keras_tuner.dir, 
                                      "dl-basic.model-tuner.rds")

dl_basic.final_model.file_path <- file.path(dl.basic.keras_tuner.dir, 
                                      "dl-basic.final-model.keras")

dlb.final_model.train_history.file_path <- file.path(dl.basic.keras_tuner.dir, 
                                                    "dl_basic.final_model.train_history.rds")

dl.basic_tune.plot_img.dir <- file.path(dl.basic.keras_tuner.dir, "plot.img")

dl.basic.best_model.plot_img.file <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner.best-model.png")
dl.basic.final_model.plot_img.file <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner.final-model.png")

dl.basic.final.conf_mx.bak <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner-final.confusion-matrix.rds")

dl.basic.final.conf_mx.img <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner-final.confusion-matrix.png")


if(!dir.exists(dl.basic_tune.plot_img.dir))
  dir.create(dl.basic_tune.plot_img.dir)

dl.basic_tuner.checkpoints.dir <- file.path(dl.basic.keras_tuner.dir,
                                            "checkpoints")
if(!dir.exists(dl.basic_tuner.checkpoints.dir))
  dir.create(dl.basic_tuner.checkpoints.dir)

dl.basic_tuner.checkpoint.file_path <- 
  file.path(dl.basic_tuner.checkpoints.dir, 
            "dl.basic_tuner.{epoch:02d}-{val_loss:.2f}.keras")

dl.basic_best.checkpoints.dir <- file.path(dl.basic.keras_tuner.dir,
                                            "checkpoints.best")
if(!dir.exists(dl.basic_best.checkpoints.dir))
  dir.create(dl.basic_best.checkpoints.dir)

dl.basic_best.checkpoint.file_path <- 
  file.path(dl.basic_best.checkpoints.dir, 
            "dl.basic_best.{epoch:02d}-{val_loss:.2f}.keras")

### Performing the Model Tuning ------------------------------------------------

# if(file.exists(dl_basic.model_tuner.file_path)) {
#   put_log("Loading the BDL Model Tuner object...")
#   
#   dl_basic.tuner <- readRDS(dl_basic.model_tuner.file_path)
#   
#   put_log("The BDL Model Tuner object has been loaded from the backup file:
# %1", dl_basic.model_tuner.file_path)
#   
# } else {
# }

put_log("Tuning the DL Basic Model on the full Train Dataset (`x.train`) of shape: %1...",
        paste0("(", str_flatten(shape(x.train), 
                                collapse = ","),")"))
start <- put_start_date()
# Log Start Time: 2026-06-29 09:22:51.209273

dl_basic.tuner <- dl.tune.hwr_model(dl_basic.tunable_model,
                                    x.train,
                                    y.train,
                                    dl.basic.keras_tuner.dir,
                                    dl.basic_tuner.checkpoint.file_path,
                                    project_name = "DL.Basic.Tuner")

put_log("The DL Basic Model Tuning has been completed.")
put_end_date(start)

# Best val_accuracy So Far: 0.8961053490638733
# Total elapsed time: 03h 38m 12s

put_log("Saving the BDL Model Tuner object...")
saveRDS(dl_basic.tuner,
        file = dl_basic.model_tuner.file_path)

put_log("The BDL Model Tuner object has been saved in the following file:
  %1", dl_basic.model_tuner.file_path)
put_end_date(start)

dl_basic.tuner$results_summary()

# <keras_tuner.src.tuners.randomsearch.RandomSearch object at 0x000001F3BD376D90>

class(dl_basic.tuner)

# This prints a summary of the search space and lists the top trial results
dlb.model_tuner.result = kerastuneR::plot_tuner(dl_basic.tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The DL Basic Tuning Results:
%1", capture.output(dlb.model_tuner.result))
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

# This prints the top trials, their hyperparameters, and execution details
dl_basic.tuner |> results_summary(num_trials = 1L)
{
  # Results summary
  # Results in data/models/dl.keras3/dl.basic/tuning/keras-tuner\DL.Basic.Tuner
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
dl_basic.best_models <- kerastuneR::get_best_models(tuner = dl_basic.tuner, num_models = 1L)
dl_basic.best_model <- dl_basic.best_models[[1]]

dl_basic.best_model$summary()
# View completed epochs of this best model
# If restore_best_weights = TRUE, this tells you the optimal epoch
# best_epoch <- dl_basic.best_model$history$params$epochs

dl_basic.best_model |> plot_keras_model(to_file = dl.basic.best_model.plot_img.file,
                                        show_shapes = TRUE)

dl_basic.tuner.best_trials <- dl_basic.tuner$oracle$get_best_trials(num_trials = 1L)
dl_basic.best_trial <- dl_basic.tuner.best_trials[[1]]
dl_basic.best_trial$summary()
dl_basic.best_trial$best_step

dl_basic.best_trial$metrics$get_history('val_accuracy')


### Close Log ------------------------------------------------------------------
log_close()
# Log Elapsed Time: 0 03:38:15

## Re-training the Best Model --------------------------------------------------

open_logfile("re-training.best.dl.basic-model")


if(file.exists(dl_basic.final_model.file_path)) {
  put_log("Loading pre-trained BDL MCC Model...")
  
  dl_basic.final_model <- keras3::load_model(dl_basic.final_model.file_path)
  
  put_log("The BDL MCC Model has been loaded from the backup file:
%1", dl_basic.final_model.file_path)
  
  if(file.exists(dlb.final_model.train_history.file_path)){
    put_log("Loading the BDL MCC Model Train History...")
    
    dlb.final_model.train_history <- readRDS(dlb.final_model.train_history.file_path)
    
    put_log("The BDL MCC Model has been loaded from the backup file:
%1", dlb.final_model.train_history.file_path)
  } else {
    warning("The BDL MCC Model backup does not exist:
", dlb.final_model.train_history.file_path)
  }
} else {
  dl_basic.tuner.best_trial.ls <- dl_basic.tuner$oracle$get_best_trials(num_trials = 10L)
  str(dl_basic.tuner.best_trial.ls)
  
  dl_basic.tuner.best_trial.last_epochs <- sapply(dl_basic.tuner.best_trial.ls, 
                                                  function(trial){
                                                    trial$best_step
                                                  })
  
  dl_basic.retrain_epochs <- max(dl_basic.tuner.best_trial.last_epochs)
  
  
  dl_basic.tuner.best_hp.ls <- dl_basic.tuner$get_best_hyperparameters(num_trials = 1L)
  # str(dl_basic.tuner.best_hp.ls)
  
  dl_basic.tuner.best_hp <- dl_basic.tuner.best_hp.ls[[1]]
  
  put_log("The best Hyperparameters configuration:
%1", capture.output(dl_basic.tuner.best_hp$get_config()))
  
  class(dl_basic.tuner.best_hp)
  # [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
  # [2] "python.builtin.object"        
  
  dl_basic.tuner.best_hp
  # <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>
  
  put_log("The best Hyperparameters values:
%1", capture.output(dl_basic.tuner.best_hp$values))
  
  # 1. Re-build a clean model structure using the winning hyperparams
  dl_basic.final_model <- dl_basic.tuner$hypermodel$build(dl_basic.tuner.best_hp)
  # print(dl_basic.final_model)
  # dl_basic.final_model$summary()
  
  put_log("The Final tuned BDL Model Summary: 
%1", capture.output(dl_basic.final_model))
  
  dl_basic.final_model |> plot_keras_model(to_file = dl.basic.final_model.plot_img.file,
                                           show_shapes = T)
  
  #best_models <- tuner |> get_best_models(num_models = 1L)
  # best_5_models[[1]] %>% plot_keras_model()
  
  dl.basic_best.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = dl.basic_best.checkpoint.file_path,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  put_log("Training the BDL MCC Model...")
  start <- put_start_date()
  
  dlb.final_model.train_history <- dl_basic.final_model |> 
    fit(x.train, 
        y.train, 
        epochs = dl_basic.retrain_epochs, 
        # batch_size = 128, 
        callbacks = dl.basic_best.callbacks,
        validation_split = 0.2
    )
  
  put_log("Saving re-trained final BDL MCC Model...")
  keras3::save_model(dl_basic.final_model,
                     filepath = dl_basic.final_model.file_path,
                     overwrite = TRUE)
  
  put_log("The re-trained final BDL MCC Model has been trained 
and saved in the following file:
  %1", dl_basic.final_model.file_path)
  
  put_log("Saving the BDL MCC Model History...")
  saveRDS(dlb.final_model.train_history,
          file = dlb.final_model.train_history.file_path)
  
  put_log("The re-trained final BDL MCC Model History has been trained 
and saved in the following file:
  %1", dlb.final_model.train_history.file_path)
  put_end_date(start)
  # Time difference of 38.48235 mins
}

put_log("The re-trained `BDL MCC` Model has been trained with the following results
%1", dl_basic.final_model)

plot(dlb.final_model.train_history)
str(dlb.final_model.train_history)

log_close()

## The Best BDL MCC Model Evaluation ----------------------------------------------------
put_log("Evaluating DL Model...")
bdl_best.eval.result <- dl_basic.final_model |> evaluate(x.test, y.test)
put_log("BDL MCC Bset Model evaluation result:
%1", capture.output(str(bdl_best.eval.result)))
# List of 2
 # $ accuracy: num 0.9
 # $ loss    : num 0.374

put_end_date(start)
# Time difference of 1.668308 mins

bdl_best.preds <- dl_basic.final_model |> predict(x.test)
str(bdl_best.preds)
# put_end_date(start)
# Time difference of  mins

colnames(bdl_best.preds) <- Y.Labels
head(bdl_best.preds[,1:5])
#                 #            $            &            @            0
# [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
# [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
# [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
# [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
# [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
# [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
dim(bdl_best.preds)
#> [1] 33228    39

bdl_best.preds.ts <- as_tensor(bdl_best.preds)
str(bdl_best.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

bdl_best.predictions <- bdl_best.preds.ts |> op_argmax(2)
bdl_best.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(bdl_best.predictions)
#> [1] 33228
# bdl_best.predictions$numpy()


# y.test
# as.integer(y.test)

bdl_best.pred.values.idx <- bdl_best.predictions$numpy()
head(bdl_best.pred.values.idx)
min(bdl_best.pred.values.idx)
max(bdl_best.pred.values.idx)

bdl_best.pred.values <- Y.Labels[bdl_best.pred.values.idx]
head(bdl_best.pred.values)

y.test.idx <- y.test + 1
# y_test <- Y.Labels[y.test.idx]

dl.basic.accuracy <- mean(bdl_best.pred.values.idx == y.test.idx)
put_log("The overall Basic `DL MCC` Model accuracy: %1",dl.basic.accuracy)
# 0.899963885879379


## Visualizing the Evaluation Results ------------------------------------------

TEST_SET.TARGETS <- y.test.cat
EVAL.PREDICTED_PROBABILITIES <- bdl_best.preds
EVAL.PREDICTION_VALUES <- bdl_best.pred.values

stopifnot(file.exists(shared_visualization.script.path))
start <- put_start_date()

source(shared_visualization.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


dl.basic.accuracy.by_class <- CURRENT_MODEL.ACCURACY_BY_CLASS

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(dl.basic.accuracy.by_class))
{
# class  accuracy
  #' # 1.0000000
  #' $ 1.0000000
  #' & 1.0000000
  #' 0 0.9683099
  #' 1 0.7546948
  #' 2 0.9061033
  #' 3 0.9565728
  #' 4 0.9330986
  #' 5 0.8873239
  #' 6 0.9272300
  #' 7 0.9882629
  #' 8 0.9143192
  #' 9 0.9518779
  #' @ 1.0000000
  #' A 0.8485915
  #' B 0.8779343
  #' C 0.9565728
  #' D 0.9025822
  #' E 0.9237089
  #' F 0.9436620
  #' G 0.6514085
  #' H 0.9354460
  #' I 0.7030516
  #' J 0.9319249
  #' K 0.9213615
  #' L 0.4049296
  #' M 0.9530516
  #' N 0.9342723
  #' P 0.9659624
  #' Q 0.7406103
  #' R 0.9190141
  #' S 0.8955399
  #' T 0.9366197
  #' U 0.9471831
  #' V 0.9260563
  #' W 0.9636150
  #' X 0.9389671
  #' Y 0.8814554
  #' Z 0.9072770
}


dl.basic.conf.mx <- CURRENT_MODEL.CONFUSION_MATRIX

put_log("The confusion matrix based on the `BDL MCC` Model evaluation results has been created:
%1", capture.output(dl.basic.conf.mx))

put_end_date(start)

rm(TEST_SET.TARGETS,
   EVAL.PREDICTED_PROBABILITIES,
   EVAL.PREDICTION_VALUES,
   CURRENT_MODEL.CONFUSION_MATRIX)

conf.mx <- plot.conf.mx(y.test.cat, 
                        bdl_best.pred.values,
                        dl.basic.final.conf_mx.bak,
                        dl.basic.final.conf_mx.img)
conf.mx

dl.basic.final.conf.mx.plot <- readRDS(dl.basic.final.conf_mx.bak)
print(dl.basic.final.conf.mx.plot)

log_close()
