#%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model
#%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# Reference:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

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
    layer_dense(units = N.classes, activation = "softmax")
  
  summary(dl.basic.model)
  
  dl.basic.model |> compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  summary(dl.basic.model)
  
  ### Training the Basic DL MCC Model ******************************************
  
  dl.basic.callbacks <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = dl.basic.checkpoint.file_path,
                              monitor = "val_loss",
                              # mode = max,
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  

  put_log("Training the BDL MCC Model...")
  start <- put_start_date()
  
  dl.basic.train_history <- dl.basic.model |> 
    fit(x.train, 
        y.train.cat, 
        epochs = 100, 
        batch_size = 128, 
        callbaks = dl.basic.callbacks,
        validation_split = 0.15
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
start <- put_start_date()
bdl.eval.result <- dl.basic.model |> evaluate(x.test, y.test.cat)
put_log("DL Model evaluation result:
%1", capture.output(str(bdl.eval.result)))
# List of 2
#  $ accuracy: num 0.796
#  $ loss    : num 1.52

put_end_date(start)
# Time difference of 1.668308 mins

start <- put_start_date()
bdl.preds <- dl.basic.model |> predict(x.test) 
put_end_date(start)
# Time difference of  mins

colnames(bdl.preds) <- y.labels
head(bdl.preds[,1:5])
#                 #            $            &            @            0
# [1,] 8.469580e-25 2.824278e-25 1.338040e-31 1.180656e-36 3.503761e-16
# [2,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
# [3,] 9.542136e-15 3.782388e-15 1.619873e-16 3.058267e-19 4.522308e-08
# [4,] 7.770129e-13 1.467184e-19 8.129414e-25 4.881509e-21 9.999547e-01
# [5,] 4.731567e-38 0.000000e+00 0.000000e+00 0.000000e+00 5.828601e-27
# [6,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00

dim(bdl.preds)
#> [1] 33228    39

bdl.preds.ts <- as_tensor(bdl.preds)
str(bdl.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

bdl.predictions <- bdl.preds.ts |> op_argmax(2)
bdl.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(bdl.predictions)
#> [1] 684467
# bdl.predictions$numpy()


# y.test
# as.integer(y.test)

bdl.pred.values.idx <- bdl.predictions$numpy()
head(bdl.pred.values.idx)

bdl.pred.values <- y.labels[bdl.pred.values.idx]
head(bdl.pred.values)

dl.basic.accuracy <- mean(bdl.pred.values.idx == as.integer(y.test))
put_log("The overall Basic `DL MCC` Model accuracy: %1",dl.basic.accuracy)
# [1] 0.9045985


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
#'     # 1.0000000
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 1.0000000
#'     0 0.9589202
#'     1 0.7464789
#'     2 0.8814554
#'     3 0.9565728
#'     4 0.9354460
#'     5 0.9213615
#'     6 0.9107981
#'     7 0.9812207
#'     8 0.9307512
#'     9 0.8591549
#'     A 0.8708920
#'     B 0.9143192
#'     C 0.9248826
#'     D 0.9284038
#'     E 0.9507042
#'     F 0.9460094
#'     G 0.7077465
#'     H 0.9366197
#'     I 0.6737089
#'     J 0.9295775
#'     K 0.9213615
#'     L 0.5316901
#'     M 0.9694836
#'     N 0.9366197
#'     P 0.9812207
#'     Q 0.7417840
#'     R 0.9553991
#'     S 0.8920188
#'     T 0.9530516
#'     U 0.9096244
#'     V 0.9284038
#'     W 0.9659624
#'     X 0.9330986
#'     Y 0.8931925
#'     Z 0.9014085
}

put_log("`BDL MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(y.labels,
                            dl.basic.accuracy.by_class,
                            title.prefix = "Basic DL Multiclass")
put_end_date(start)



log_close()


