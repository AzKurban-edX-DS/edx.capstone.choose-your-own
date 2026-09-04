#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DNN-Based Basic (DNNB) MCC  Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".dnnb-mcc.model.training")
start <- put_start_date()

stopifnot(file.exists(my_emnist.split.file_path))

# Dense Neural Network Based Basic Multiclass Classifier (DNNB MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3

## Prepare a Training Set for DNN-Based Basic MCC Model ------------------------

put_log("Loading the Training Set of 28x28-size image data...")
train_set <- load.train_set(ds28x28.split.train_0.8.backup.file)

put_log("The Training Set of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)


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

## Analyzing Image Data --------------------------------------------------------

max_img_pixels <- max(rowSums(x_train))
# put_log("The maximum number of pixels in the Training Set images is as follows: %1",
#         max_img_pixels)
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

dnnb_mcc.checkpoint.file_path <- 
  file.path(dnn_mcc.basic.checkpoints.dir, 
            "dnnb_mcc.{epoch:02d}-{val_loss:.2f}.keras")

## Building DNNB MCC Model -----------------------------------------------------
### Defining & Compiling the Basic DNNB MCC Model -----------------------------

n.input_shape <- ncol(x_train)
# 784

dnnb_mcc.inputs <- layer_input(shape = c(28, 28))

dnnb_mcc.outputs <- dnnb_mcc.inputs |>
  layer_flatten() |>
  layer_dense(units = n.hl.units, activation = "relu") |>
  layer_dropout(rate = 0.25) |> 
  layer_dense(units = N.classes, activation = "softmax")


dnnb_mcc <- keras_model(dnnb_mcc.inputs, dnnb_mcc.outputs)

dnnb_mcc |> compile(loss = "sparse_categorical_crossentropy",
                           optimizer = keras3::optimizer_adamax(0.001),
                           metrics = "accuracy")
summary(dnnb_mcc)

### Training the Basic DNNB MCC Model --------------------------------------------

dnnb_mcc.callbacks <- list(
  callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
  callback_model_checkpoint(filepath = dnnb_mcc.checkpoint.file_path,
                            monitor = "val_loss",
                            save_best_only = TRUE,
                            verbose = 1)
)

put_log("Training the BDNNB MCC Model...")
start <- put_start_date()

# dnnb_mcc.train_history <- tdnn_mcc.final |> 
dnnb_mcc.train_history <- dnnb_mcc |>
  fit(x_train, 
      y_train, 
      epochs = 100, 
      # batch_size = 128, 
      callbacks = dnnb_mcc.callbacks,
      validation_split = 0.2
  )

put_log("Saving pre-trained BDNNB MCC Model...")
keras3::save_model(dnnb_mcc,
                   filepath = dnnb_mcc.file,
                   overwrite = T)

put_log("The BDNNB MCC Model has been trained 
and saved in the following file:
  %1", dnnb_mcc.file)

put_log("Saving the BDNNB MCC Model History...")
saveRDS(dnnb_mcc.train_history,
        file = dnnb_mcc.train_history.file)

put_log("The BDNNB MCC Model History has been trained 
and saved in the following file:
  %1", dnnb_mcc.train_history.file)
put_end_date(start)



# rm(x_train,
#    y_train.cat)

put_log("The trained Basic DNNB MCC Model summary:
%1", dnnb_mcc)

plot(dnnb_mcc.train_history)

put_log("Structure of the Basic DNNB MCC Model training history:
%1", capture.output(str(dnnb_mcc.train_history)))


## Finalizing ------------------------------------------------------------------

rm(dnnb_mcc.train_history)
log_close()
# =========================================================================
# Log End Time: 2026-09-03 23:20:23.963723
# Log Elapsed Time: 0 00:04:06
# =========================================================================
