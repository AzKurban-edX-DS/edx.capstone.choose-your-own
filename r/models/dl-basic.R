#%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic Deep Learning Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%

### Basic DL Classifier -----------------------------------------------------------
# Reference:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/

#### Loading Split Dataset allocated 20% for the Test set (default) ------------
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

log_close()

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

#### Init DL Basic Model Paths -------------------------------------------------

if(!dir.exists(dl.keras3.path))
  dir.create(dl.keras3.path)

dl.basic.dir_path <- file.path(dl.keras3.path, "dl.basic")

if(!dir.exists(dl.basic.dir_path))
  dir.create(dl.basic.dir_path)

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

#### Building Deep Learning Basic Multiclass Classifier (DLB MCC) Model --------
open_logfile("dl.basic-model")


if(file.exists(dl.basic.model.file_path)) {
  put_log("Loading pre-trained DLB MCC Model...")
  
  dl.basic.model <- load_model(dl.basic.model.file_path)
  
  put_log("The DLB MCC Model has been loaded from the backup file:
%1", dl.basic.model.file_path)
  
  if(file.exists(dl.basic.model.train_history.file_path)){
    put_log("Loading the DLB MCC Model Train History...")
    
    dl.basic.train_history <- readRDS(dl.basic.model.train_history.file_path)
    
    put_log("The DLB MCC Model has been loaded from the backup file:
%1", dl.basic.model.train_history.file_path)
  } else {
    warning("The DLB MCC Model backup does not exist:
", dl.basic.model.train_history.file_path)
  }
} else {
  n.input_shape <- ncol(x.train)
  # 784
  
  n.output <- length(y.labels)
  # 39
  
  n.hl.units <- ceiling(n.input_shape*2/3+n.output)
  # 562
  
  dl.basic.model <- keras_model_sequential() |>
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
    layer_dense(units = n.output, activation = "softmax")
  
  summary(dl.basic.model)
  
  dl.basic.model |> compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  summary(dl.basic.model)
  
  #### Training DL Basic Muliclass Classifier (MCC) Model **********************
  
  dl.basic.callbacks <- list(
    callback_model_checkpoint(filepath = dl.basic.checkpoint.file_path,
                              monitor = "val_accuracy",
                              mode = max,
                              # save_best_only = TRUE,
                              verbose = 1)
  )

  put_log("Training the DLB MCC Model...")
  start <- put_start_date()
  
  dl.basic.train_history <- dl.basic.model |> 
    fit(x.train, 
        y.train.cat, 
        epochs = 100, 
        batch_size = 512, 
        validation_split = 0.15,
        callbaks = dl.basic.callbacks
        )
  
  
  put_log("Saving pre-trained DLB MCC Model...")
  save_model(dl.basic.model,
             filepath = dl.basic.model.file_path,
             overwrite = FALSE)
  
  put_log("The DLB MCC Model has been trained 
and saved in the following file:
  %1", dl.basic.model.file_path)

  
  put_log("Saving the DLB MCC Model History...")
  saveRDS(dl.basic.train_history,
          file = dl.basic.model.train_history.file_path)
  
  put_log("The DLB MCC Model History has been trained 
and saved in the following file:
  %1", dl.basic.model.train_history.file_path)
  put_end_date(start)
  # Time difference of 38.48235 mins
}

plot(dl.basic.train_history)
str(dl.basic.train_history)
#### DL Basic Model Evaluation ----------------------------------------------
put_log("Evaluating DL Model...")
start <- put_start_date()
dl.eval.result <- dl.basic.model |> evaluate(x.test, y.test.cat)
put_log("DL Model evaluation result:
%1", capture.output(str(dl.eval.result)))
# List of 2
#  $ accuracy: num 0.796
#  $ loss    : num 1.52

put_end_date(start)
# Time difference of 1.668308 mins

start <- put_start_date()
dl.preds <- dl.basic.model |> predict(x.test) 
put_end_date(start)
# Time difference of  mins

colnames(dl.preds) <- y.labels
head(dl.preds[,1:5])
#                 #            $            &            @            0
# [1,] 8.469580e-25 2.824278e-25 1.338040e-31 1.180656e-36 3.503761e-16
# [2,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
# [3,] 9.542136e-15 3.782388e-15 1.619873e-16 3.058267e-19 4.522308e-08
# [4,] 7.770129e-13 1.467184e-19 8.129414e-25 4.881509e-21 9.999547e-01
# [5,] 4.731567e-38 0.000000e+00 0.000000e+00 0.000000e+00 5.828601e-27
# [6,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00

dim(dl.preds)
#> [1] 684467     39

dl.preds.ts <- as_tensor(dl.preds)
str(dl.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

dl.predictions <- dl.preds.ts |> op_argmax(2)
dl.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(dl.predictions)
#> [1] 684467
# dl.predictions$numpy()


# y.test
# as.integer(y.test)

dl.model.accuracy <- mean(dl.predictions$numpy() == as.integer(y.test))
put_log("DL Model accuracy: %1",dl.model.accuracy)

# [1] 0.7955373

##### Close Log ------------------------------------------------------------------
log_close()

# ---------------------------
# Reference:
#
# 
start <- put_start_date()
put_end_date(start)






























