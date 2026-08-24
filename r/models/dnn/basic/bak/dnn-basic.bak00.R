#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DNN-Based Basic (DNNB) MCC  Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

open_logfile(".dnnb-mcc.model.training")

stopifnot(file.exists(my_emnist.split.file_path))

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3

## Preparing a Training Set -------------------------------------------------- 
start <- put_start_date()


put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.split.file_path)

put_log("The Split Flatten Dataset has the following structure:
%1", capture.output(str(ds)))

x_train <- ds$train$x
y_train <- ds$train$class_groups$classID

stopifnot(sum(as.character(y_train) != rownames(x_train)) == 0)
stopifnot(nrow(x_train) == length(y_train))

### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/


y_train.cat <- to_categorical(y_train)
rm(y_train)

colnames(y_train.cat) <- Y.Labels
dim(y_train.cat)
str(y_train.cat)
head(y_train.cat)
# max(y_train.cat)

### Size of the Training Set by Class ----------------------------------------
put_log("The Train set is balanced with respect to the set of classes:
%1", capture.output(print(ds$train$class_groups$groupByClass, n = N.classes)))
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

## Analyzing Image Data --------------------------------------------------------
max_img_pixels <- max(rowSums(x_train))
put_log("The maximum number of pixels in the Training Set images is as follows: %1",
        max_img_pixels)
#> 593

# References:
# https://community.deeplearning.ai/t/number-of-the-hidden-units-in-hidden-layers/24916
# https://milvus.io/ai-quick-reference/how-do-you-decide-the-number-of-neurons-per-layer
# https://tracyrenee61.medium.com/how-do-you-calculate-the-correct-number-of-neurons-to-place-in-the-hidden-layers-of-a-neural-19cbaf68452c
# https://arxiv.org/pdf/1707.09725#page=11
# https://app.speechify.com/item/bde53a50-bd0e-4328-8273-27caf5174899?type=WEB
# https://www.kaggle.com/discussions/general/321114
# https://ijettjournal.org/assets/volume-3/issue-6/IJETT-V3I6P206.pdf
# https://www.heatonresearch.com/2017/06/01/hidden-layers.html [JH_NHL]

n.hl.units <- ceiling(28*28/3*2 + 39)
# 562

put_log("For the couple of hidden layers of our first deep learning model, 
we will choose %1 neurons, which meets the value suggested by the rule-of-thumb methods 
(outlined in the article 'Heaton Research: The Number of Hidden Layers' [JH_NHL2017-06-01]), 
and also does not exceed the maximum number of black pixels (%2) making up every (pre-processed) image 
contained in the training set.",
        n.hl.units,
        max_img_pixels)

## Init DNNB MCC Model Paths ---------------------------------------------------

dnn_mcc.basic.checkpoints.dir <- file.path(data.dnn_mcc.basic.dir,
                                            "checkpoints")
if(!dir.exists(dnn_mcc.basic.checkpoints.dir))
  dir.create(dnn_mcc.basic.checkpoints.dir)

dnn_basic.checkpoint.file_path <- 
  file.path(dnn_mcc.basic.checkpoints.dir, 
            "dnn_basic.{epoch:02d}-{val_loss:.2f}.keras")

## Building DNNB MCC Model -----------------------------------------------------

n.input_shape <- ncol(x_train)
# 784

#### Defining & Compiling the Basic DL MCC Model -------------------------------

n.input_shape <- ncol(x_train)
# 784

dnn_basic.inputs <- layer_input(shape = c(n.input_shape))

dnn_basic.outputs <- dnn_basic.inputs |>
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = N.classes, activation = "softmax")


dnn_basic.model <- keras_model(dnn_basic.inputs, dnn_basic.outputs)

dnn_basic.model |> compile(
  loss = "categorical_crossentropy",
  optimizer = keras3::optimizer_adamax(0.001),
  metrics = "accuracy"
)

summary(dnn_basic.model)

### Training the Basic DL MCC Model --------------------------------------------

dnn_basic.callbacks <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = dnn_basic.checkpoint.file_path,
                            monitor = "val_loss",
                            save_best_only = TRUE,
                            verbose = 1)
)

put_log("Training the BDL MCC Model...")
start <- put_start_date()

dnn_basic.train_history <- dnn_basic.model |> 
  fit(x_train, 
      y_train.cat, 
      epochs = 100, 
      # batch_size = 128, 
      callbacks = dnn_basic.callbacks,
      validation_split = 0.2
  )

put_log("Saving pre-trained BDL MCC Model...")
keras3::save_model(dnn_basic.model,
                   filepath = dnn_basic.model.file_path,
                   overwrite = FALSE)

put_log("The BDL MCC Model has been trained 
and saved in the following file:
  %1", dnn_basic.model.file_path)

put_log("Saving the BDL MCC Model History...")
saveRDS(dnn_basic.train_history,
        file = dnn_basic.model.train_history.file_path)

put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", dnn_basic.model.train_history.file_path)
put_end_date(start)



rm(x_train,
   y_train.cat)

put_log("The trained Basic DL MCC Model summary:
%1", dnn_basic.model)

plot(dnn_basic.train_history)

put_log("Structure of the Basic DL MCC Model training history:
%1", capture.output(str(dnn_basic.train_history)))

rm(dnn_basic.train_history)
log_close()
# Log Elapsed Time: 0 00:04:07
