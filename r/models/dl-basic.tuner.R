#%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model
#%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3

## Preparing Datasets for DL Basic Model Tuning ---------------------------------
open_logfile(".prepare-dataset-for-dl.model-tuning")
start <- put_start_date()
# stopifnot(file.exists(my_emnist.split.file_path))
stopifnot(exists("x3d.train_set"))
stopifnot(exists("x3d.test_set"))

# ds_flatten <- load_flatten_datasets("ds_flatten.split_list", 
#                                     my_emnist.split.file_path)
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

put_log("The Train Set is balanced by set of Classes:
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

y.test.groups <- ds.get_classIDs.grouped(x.test)
y.test <- y.test.groups$classID

stopifnot(sum(as.character(y.test) != rownames(x.test)) == 0)

put_log("The Train Set is balanced by set of Classes:
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

dim(x.train)
#> [1] 16653   784
str(x.train)

str(y.train)
length(y.train)

str(x.test)
str(y.test)
length(y.test)
#> [1] 817379


#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/


y.train.cat <- to_categorical(y.train)
colnames(y.train.cat) <- y.labels
dim(y.train.cat)
str(y.train.cat)
head(y.train.cat)
# max(y.train.cat)

y.test.cat <- to_categorical(y.test)
colnames(y.test.cat) <- y.labels
dim(y.test.cat)
str(y.test.cat)
head(y.test.cat)

log_close()

## Training & Tuning DL Basic Model --------------------------------------------
open_logfile(".dl.basic-model-tuning")
start <- put_start_date()
# Log Start Time: 2026-06-29 09:22:51.209273

### Init DL Basic Model Tuner Paths --------------------------------------------

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

dl_basic.final_model.file_path <- file.path(dl.basic.keras_tuner.dir, 
                                      "dl-basic.final-model.keras")

dlb.final_model.train_history.file_path <- file.path(dl.basic.keras_tuner.dir, 
                                                    "dl_basic.final_model.train_history.rds")

dl.basic_tune.plot_img.dir <- file.path(dl.basic.keras_tuner.dir, "plot.img")

dl.basic.best_model.plot_img.file <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner.best-model.png")
dl.basic.final_model.plot_img.file <- file.path(dl.basic_tune.plot_img.dir, 
                                               "tuner.final-model.png")

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

### Tuning Basic DL MCC Model ---------------------------------------------------

dl_basic.tuner <- dl.tune.hwr_model(dl_basic.tunable_model,
                                    x.train,
                                    y.train,
                                    dl.basic.keras_tuner.dir,
                                    dl.basic_tuner.checkpoint.file_path,
                                    project_name = "DL.Basic.Tuner")

# Best val_accuracy So Far: 0.8961053490638733
# Total elapsed time: 03h 38m 12s

dl_basic.tuner
# <keras_tuner.src.tuners.randomsearch.RandomSearch object at 0x000001F3BD376D90>

class(dl_basic.tuner)

# This prints a summary of the search space and lists the top trial results
dlb.model_tuner.result = kerastuneR::plot_tuner(dl_basic.tuner)
# the list will show the plot and the data.frame of tuning results
dlb.model_tuner.result 
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
dl_basic.tuner |> results_summary(num_trials = 1)

# Retrieve the best model from the search
dl_basic.best_models <- kerastuneR::get_best_models(tuner = dl_basic.tuner, num_models = 1L)
dl_basic.best_model <- dl_basic.best_models[[1]]
dl_basic.best_model
# View completed epochs of this best model
# If restore_best_weights = TRUE, this tells you the optimal epoch
# best_epoch <- dl_basic.best_model$history$params$epochs

dl_basic.best_model |> plot_keras_model(to_file = dl.basic.best_model.plot_img.file,
                                        show_shapes = TRUE)

dl_basic.tuner.best_trials <- dl_basic.tuner$oracle$get_best_trials(num_trials = 5L)
dl_basic.best_trial <- dl_basic.tuner.best_trials[[1]]
dl_basic.best_trial$summary()


log_close()
# Log Elapsed Time: 0 03:38:15

### Re-training the Best Model --------------------------------------------------

open_logfile("re-training.best.dl.basic-model")

dl_basic.tuner.best_hp.ls <- tuner$get_best_hyperparameters()
str(dl_basic.tuner.best_hp.ls)

dl_basic.tuner.best_hp <- dl_basic.tuner.best_hp.ls[[1]]

class(dl_basic.tuner.best_hp)
# [1] "keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters"
# [2] "python.builtin.object"        

str(dl_basic.tuner.best_hp)
# <keras_tuner.src.engine.hyperparameters.hyperparameters.HyperParameters object at 0x000001F55F89D010>

dl_basic.tuner.best_hp$values

# 1. Re-build a clean model structure using the winning hyperparams
dl_basic.final_model <- tuner$hypermodel$build(dl_basic.tuner.best_hp)
print(dl_basic.final_model)
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
      epochs = 40, 
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

put_log("The re-trained `BDL MCC` Model has been trained with the following results
%1", dl_basic.final_model)



plot(dlb.final_model.train_history)
str(dlb.final_model.train_history)

log_close()

## BDL MCC Model Evaluation ----------------------------------------------------
put_log("Evaluating DL Model...")
bdl.eval.result <- dl_basic.final_model |> evaluate(x.test, y.test)
put_log("DL Model evaluation result:
%1", capture.output(str(bdl.eval.result)))
# List of 2
#  $ accuracy: num 0.907
#  $ loss    : num 0.299

put_end_date(start)
# Time difference of 1.668308 mins

bdl.preds <- dl_basic.final_model |> predict(x.test) 
put_end_date(start)
# Time difference of  mins

colnames(bdl.preds) <- y.labels
head(bdl.preds[,1:5])
#                 #            $            &            @            0
# [1,] 3.792269e-07 8.851480e-07 2.578369e-08 3.807849e-07 3.528773e-05
# [2,] 1.730552e-13 6.454766e-15 1.392669e-09 6.399740e-10 3.266690e-07
# [3,] 9.197757e-16 3.064377e-12 1.566798e-11 5.085300e-16 8.313370e-07
# [4,] 7.945133e-08 5.979302e-07 3.491478e-08 4.118463e-09 3.721448e-06
# [5,] 1.046998e-12 9.161977e-15 2.002677e-24 1.114895e-18 6.859786e-13
# [6,] 4.039502e-14 1.164542e-14 2.524913e-17 2.605324e-14 5.818604e-10

dim(bdl.preds)
#> [1] 33228    39

bdl.preds.ts <- as_tensor(bdl.preds)
str(bdl.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

bdl.predictions <- bdl.preds.ts |> op_argmax(2)
bdl.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(bdl.predictions)
#> [1] 33228
# bdl.predictions$numpy()


# y.test
# as.integer(y.test)

bdl.pred.values.idx <- bdl.predictions$numpy()
head(bdl.pred.values.idx)

bdl.pred.values <- y.labels[bdl.pred.values.idx]
head(bdl.pred.values)

dl.basic.accuracy <- mean(bdl.pred.values.idx == as.integer(y.test))
put_log("The overall Basic `DL MCC` Model accuracy: %1",dl.basic.accuracy)
# 0.906735283495847


put_log("`BDL MCC` Model Evaluation: Calculating a ROC curve for each class...")
dl.basic.roc_curves <- calc.roc_curves(y.test,
                                       bdl.preds,
                                       y.labels)
put_log("`BDL MCC` Model Evaluation: The per-class ROC curve calculation 
has been completed.")

plot(dl.basic.roc_curves[[1]], 
     main = "ROC Curves for the `Basic Deep Learning Multiclass Classifier` Model")
for (class.idx in 2:N.classes) {
  lines(dl.basic.roc_curves[[class.idx]], col = class.idx)
}


# Confusion Matrix data suitable for Visualization using the `cvms` package:
# Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
put_log("`BDL MCC` Model: Creating a confusion matrix in a format suitable for visualization 
using the `cvms` package...")
dl.basic.conf.mx <- confusion_matrix(as.character(y.test),
                                     as.character(bdl.pred.values))
put_log("The confusion matrix based on the `BDL MCC` Model evaluation results has been created:
%1", capture.output(dl.basic.conf.mx))  

# put_log("Plotting the confusion matrix, please wait...")
# start <- put_start_date()
# cl <- makeCluster(N_pcCores)
# registerDoParallel(cl)
#
# 
# dev.off()
# plot_confusion_matrix(dl.basic.conf.mx,
#                       palette = "Greens",
#                       font_counts = font(size = 3,
#                                          color = "red"),
#                       add_normalized = FALSE,
#                       add_col_percentages = FALSE,
#                       add_row_percentages = FALSE)
# 
# stopCluster(cl)
# stopImplicitCluster()
# put_end_date(start)

dl.basic.accuracy.by_class <- MCClassifier.accuracy.by_class(y.labels,
                                                             y.test,
                                                             bdl.pred.values)
dl.basic.accuracy.by_class
{
#' class  accuracy
  #' # 1.0000000
  #' $ 1.0000000
  #' & 1.0000000
  #' @ 1.0000000
  #' 0 0.9671362
  #' 1 0.8638498
  #' 2 0.9143192
  #' 3 0.9565728
  #' 4 0.9319249
  #' 5 0.9025822
  #' 6 0.9284038
  #' 7 0.9800469
  #' 8 0.9295775
  #' 9 0.9507042
  #' A 0.8791080
  #' B 0.9025822
  #' C 0.9589202
  #' D 0.9295775
  #' E 0.9495305
  #' F 0.9577465
  #' G 0.6725352
  #' H 0.9237089
  #' I 0.6525822
  #' J 0.9272300
  #' K 0.9272300
  #' L 0.3967136
  #' M 0.9659624
  #' N 0.9389671
  #' P 0.9683099
  #' Q 0.7159624
  #' R 0.9448357
  #' S 0.9084507
  #' T 0.9518779
  #' U 0.9330986
  #' V 0.9401408
  #' W 0.9624413
  #' X 0.9518779
  #' Y 0.8685446
  #' Z 0.9096244
}

put_log("`BDL MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(y.labels,
                            dl.basic.accuracy.by_class,
                            title.prefix = "Basic DL Multiclass")
put_end_date(start)



log_close()


