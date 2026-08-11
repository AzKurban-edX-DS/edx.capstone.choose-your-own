#%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model
#%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3



## Loading Split Dataset allocated 20% for the Test set (default) ------------
open_logfile(".split.20%test.balanced_subset")

start <- put_start_date()
stopifnot(file.exists(my_emnist.split.file_path))

ds_flatten <- load_flatten_datasets("ds_flatten.split_list", 
                                    my_emnist.split.file_path)
x.train <- ds_flatten$x.train
x.test <- ds_flatten$x.test
x.test.files <- ds_flatten$x.files

y.train.groups <- ds.get_classIDs.grouped(x.train)
y.train <- y.train.groups$classID

stopifnot(sum(as.character(y.train) != rownames(x.train)) == 0)

put_log("The Training Set is balanced by set of Classes:
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

put_log("The Training Set is balanced by set of Classes:
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

## Init DL Basic Model Paths ---------------------------------------------------

if(!dir.exists(dl.keras3.path))
  dir.create(dl.keras3.path)

dl.basic.dir_path <- file.path(dl.keras3.path, "dl.basic")

if(!dir.exists(dl.basic.dir_path))
  dir.create(dl.basic.dir_path)

dl.basic.tuning.dir <- file.path(dl.basic.dir_path,
                                            "tuning")
if(!dir.exists(dl.basic.tuning.dir))
  dir.create(dl.basic.tuning.dir)

dl.basic.checkpoints.dir <- file.path(dl.basic.dir_path,
                                            "checkpoints")
if(!dir.exists(dl.basic.checkpoints.dir))
  dir.create(dl.basic.checkpoints.dir)

dl.basic.checkpoint.file_path <- 
  file.path(dl.basic.checkpoints.dir, 
            "dl.basic.{epoch:02d}-{val_loss:.2f}.keras")

dl.basic.model.file_path <- file.path(dl.basic.dir_path, 
                             "dl.basic.pre-trained.model.keras")

dl.basic.model.train_history.file_path <- file.path(dl.basic.dir_path, 
                             "dl.basic.model.train_history.bak.rds")

## Tuning Basic DL MCC Model ---------------------------------------------------

n.input_shape <- ncol(x.train)
# 784

dl.basic.inputs <- layer_input(shape = c(n.input_shape))

### 1 Hidden Layer ----------------------------------------------

stopifnot(dir.exists(dl.basic.tuning.dir))

hl1.dir <- file.path(dl.basic.tuning.dir,"1-hidden-layer")
if(!dir.exists(hl1.dir))
  dir.create(hl1.dir)

dl.model.hl1.tuning_results.file_path <- 
  file.path(hl1.dir, "1hl.units_tuning.results.rds")

n.hl.units <- c(784, 562, 512, 256, 128, 64)

put_log("Tuning DL Basic Model for 1 Hidden Layer with different number of units...")
start <- put_start_date()

dl.basic.tuning.result <- lapply(n.hl.units, function(n.units) {
  stopifnot(dir.exists(hl1.dir))
  
  n_units.dir <- file.path(hl1.dir, paste0(n.units, "units"))
  
  if(!dir.exists(n_units.dir))
    dir.create(n_units.dir)
  
  model.file_path <- file.path(n_units.dir, "dl.basic.model.keras")
  
  model.train_history.file_path <- file.path(n_units.dir, 
                                             "dl.basic.model.train_history.rds")
  
  checkpoints.dir <- file.path(n_units.dir, "checkpoints")
  
  if(!dir.exists(checkpoints.dir))
    dir.create(checkpoints.dir)
  
  checkpoints.file_path <- 
    file.path(checkpoints.dir, 
              "dl.basic.hl1.{epoch:02d}-{val_loss:.2f}.keras")

  dl.basic.outputs <- dl.basic.inputs |>
    layer_dense(units = n.units, activation = "relu", 
                input_shape = c(n.input_shape)) |>
    layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    layer_dense(units = N.classes, activation = "softmax")
  
  
  dl.basic.model <- keras_model(dl.basic.inputs, dl.basic.outputs)
  dl.basic.model
  
  dl.basic.model |> compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = keras3::optimizer_adamax(0.001),
    metrics = "accuracy"
  )
  
  summary(dl.basic.model)

 
  model.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = checkpoints.file_path,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  ### Training the Basic DL MCC Model ******************************************
  
  put_log("Training the BDL MCC Model...")
  start <- put_start_date()
  
  model.train_history <- dl.basic.model |> 
    fit(x.train, 
        y.train, 
        epochs = 50, 
        # batch_size = 128, 
        callbacks = model.callbacks,
        validation_split = 0.2
    )
  
  put_log("Saving pre-trained BDL MCC Model...")
  save_model(dl.basic.model,
             filepath = model.file_path,
             overwrite = TRUE)
  
  put_log("The BDL MCC Model has been trained 
and saved in the following file:
  %1", model.file_path)
  
  put_log("Saving the BDL MCC Model History...")
  saveRDS(model.train_history,
          file = model.train_history.file_path)
  
  put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", model.train_history.file_path)
  # Time difference of 38.48235 mins
  
  put_log("Evaluating DL Model...")
  eval.result <- dl.basic.model |> evaluate(x.test, y.test)
  put_log("DL Model evaluation result:
%1", capture.output(str(eval.result)))
  # List of 2
  #  $ accuracy: num 0.907
  #  $ loss    : num 0.299
  
  put_end_date(start)
  
  
  list(model.file_path = model.file_path,
       train_history = model.train_history,
       eval.result = eval.result,
       n.units = n.units)
  
})

names(dl.basic.tuning.result) <- as.character(n.hl.units)
str(dl.basic.tuning.result)

put_log("Saving the BDL MCC Model History...")
saveRDS(dl.basic.tuning.result,
        file = dl.model.hl1.tuning_results.file_path)

put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", dl.model.hl1.tuning_results.file_path)
put_end_date(start)

best_tuning.result <- dl.basic.tuning.result$`784`
str(best_tuning.result)

plot(best_tuning.result$train_history)

## Building Basic DL MCC Model -------------------------------------------------

open_logfile("dl.basic-model")


if(file.exists(dl.basic.model.file_path)) {
  put_log("Loading pre-trained BDL MCC Model...")
  
  dl.basic.model <- load_model(dl.basic.model.file_path)
  
  put_log("The BDL MCC Model has been loaded from the backup file:
%1", dl.basic.model.file_path)
  
  if(file.exists(dl.basic.model.train_history.file_path)){
    put_log("Loading the BDL MCC Model Train History...")
    
    dl.basic.train_history <- readRDS(dl.basic.model.train_history.file_path)
    
    put_log("The BDL MCC Model has been loaded from the backup file:
%1", dl.basic.model.train_history.file_path)
  } else {
    warning("The BDL MCC Model backup does not exist:
", dl.basic.model.train_history.file_path)
  }
} else {
  ### Defining & Compiling the Basic DL MCC Model ******************************
  
  n.input_shape <- ncol(x.train)
  # 784
  
  n.hl.units <- 512 # ceiling(n.input_shape*2/3+N.classes)
                    # 562
  
  
  dl.basic.inputs <- layer_input(shape = c(n.input_shape))
  
  dl.basic.outputs <- dl.basic.inputs |>
    layer_dense(units = n.hl.units, activation = "relu", 
                input_shape = c(n.input_shape)) |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = N.classes, activation = "softmax")
  
  
  dl.basic.model <- keras_model(dl.basic.inputs, dl.basic.outputs)
  dl.basic.model
  
  dl.basic.model |> compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = keras3::optimizer_adamax(0.001),
    metrics = "accuracy"
  )
  
  summary(dl.basic.model)
  
  ### Training the Basic DL MCC Model ******************************************
  
  dl.basic.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = dl.basic.checkpoint.file_path,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  

  put_log("Training the BDL MCC Model...")
  start <- put_start_date()
  
  dl.basic.train_history <- dl.basic.model |> 
    fit(x.train, 
        y.train, 
        epochs = 100, 
        # batch_size = 128, 
        callbacks = dl.basic.callbacks,
        validation_split = 0.2
        )

  put_log("Saving pre-trained BDL MCC Model...")
  save_model(dl.basic.model,
             filepath = dl.basic.model.file_path,
             overwrite = FALSE)
  
  put_log("The BDL MCC Model has been trained 
and saved in the following file:
  %1", dl.basic.model.file_path)

  put_log("Saving the BDL MCC Model History...")
  saveRDS(dl.basic.train_history,
          file = dl.basic.model.train_history.file_path)
  
  put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", dl.basic.model.train_history.file_path)
  put_end_date(start)
  # Time difference of 38.48235 mins
}

put_log("The Basic `DL MCC` Model has been trained with the following results
%1", dl.basic.model)


plot(dl.basic.train_history)
str(dl.basic.train_history)

## BDL MCC Model Evaluation ----------------------------------------------------
put_log("Evaluating DL Model...")
bdl.eval.result <- dl.basic.model |> evaluate(x.test, y.test)
put_log("DL Model evaluation result:
%1", capture.output(str(bdl.eval.result)))
# List of 2
#  $ accuracy: num 0.907
#  $ loss    : num 0.299

put_end_date(start)
# Time difference of 1.668308 mins

bdl.preds <- dl.basic.model |> predict(x.test) 
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


