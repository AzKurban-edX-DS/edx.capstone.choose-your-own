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

stopifnot(exists("x0.1.train.flatten"),
          exists("y0.1.train.flatten"),
          exists("x0.9.test.flatten"),
          exists("y0.9.test.flatten"))

start <- put_start_date()

str(x.train.flatten)
dim(x.train.flatten)
#> [1] 16653   784

str(y.train.flatten)
length(y.train.flatten)

str(x.test.flatten)
str(y.test.flatten)
length(y.test.flatten)
#> [1] 817379

shape(x.train.flatten)
# shape(132873, 784)

max_img_pixels <- max(rowSums(x.train.flatten))
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

#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/


y.train.flatten.cat <- to_categorical(y.train.flatten)
colnames(y.train.flatten.cat) <- Y.Labels
dim(y.train.flatten.cat)
str(y.train.flatten.cat)
head(y.train.flatten.cat)
# max(y.train.flatten.cat)

y.test.flatten.cat <- to_categorical(y.test.flatten)
colnames(y.test.flatten.cat) <- Y.Labels
dim(y.test.flatten.cat)
str(y.test.flatten.cat)
head(y.test.flatten.cat)

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

dl.basic.layers_dynamic.dir <- file.path(dl.basic.tuning.dir, "layers-dynamic")

if(!dir.exists(dl.basic.layers_dynamic.dir))
  dir.create(dl.basic.layers_dynamic.dir)

dl.basic.keras_tuner.dir <- file.path(dl.basic.tuning.dir, "keras-tuner")

if(!dir.exists(dl.basic.keras_tuner.dir))
  dir.create(dl.basic.keras_tuner.dir)

dl.basic.checkpoints.dir <- file.path(dl.basic.dir_path,
                                            "checkpoints")
if(!dir.exists(dl.basic.checkpoints.dir))
  dir.create(dl.basic.checkpoints.dir)

dl.basic.checkpoint.file_path <- 
  file.path(dl.basic.checkpoints.dir, 
            "dl.basic.{epoch:02d}-{val_loss:.2f}.keras")

dl.basic.model.file_path <- file.path(dl.basic.dir_path, 
                             "dl.basic.pre-trained.model.keras")

dl.basic.model.train.flatten_history.file_path <- file.path(dl.basic.dir_path, 
                             "dl.basic.model.train.flatten_history.bak.rds")

## Building Basic DL MCC Model -------------------------------------------------

open_logfile("dl.basic-model")

n.input_shape <- ncol(x.train.flatten)
# 784

if(file.exists(dl.basic.model.file_path)) {
  put_log("Loading pre-trained BDL MCC Model...")
  
  dl.basic.model <- keras3::load_model(dl.basic.model.file_path)
  
  put_log("The BDL MCC Model has been loaded from the backup file:
%1", dl.basic.model.file_path)
  
  if(file.exists(dl.basic.model.train.flatten_history.file_path)){
    put_log("Loading the BDL MCC Model Train History...")
    
    dl.basic.train.flatten_history <- readRDS(dl.basic.model.train.flatten_history.file_path)
    
    put_log("The BDL MCC Model has been loaded from the backup file:
%1", dl.basic.model.train.flatten_history.file_path)
  } else {
    warning("The BDL MCC Model backup does not exist:
", dl.basic.model.train.flatten_history.file_path)
  }
} else {
  ### Defining & Compiling the Basic DL MCC Model ******************************
  
  n.input_shape <- ncol(x.train.flatten)
  # 784
  
  dl.basic.inputs <- layer_input(shape = c(n.input_shape))
  
  dl.basic.outputs <- dl.basic.inputs |>
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = N.classes, activation = "softmax")
  
  
  dl.basic.model <- keras_model(dl.basic.inputs, dl.basic.outputs)
  # dl.basic.model
  
  dl.basic.model |> compile(
    loss = "categorical_crossentropy",
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
  
  dl.basic.train.flatten_history <- dl.basic.model |> 
    fit(x.train.flatten, 
        y.train.flatten.cat, 
        epochs = 100, 
        # batch_size = 128, 
        callbacks = dl.basic.callbacks,
        validation_split = 0.2
        )

  put_log("Saving pre-trained BDL MCC Model...")
  keras3::save_model(dl.basic.model,
             filepath = dl.basic.model.file_path,
             overwrite = FALSE)
  
  put_log("The BDL MCC Model has been trained 
and saved in the following file:
  %1", dl.basic.model.file_path)

  put_log("Saving the BDL MCC Model History...")
  saveRDS(dl.basic.train.flatten_history,
          file = dl.basic.model.train.flatten_history.file_path)
  
  put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", dl.basic.model.train.flatten_history.file_path)
  put_end_date(start)
  # Time difference of 38.48235 mins
}

put_log("The Basic `DL MCC` Model has been trained with the following results
%1", dl.basic.model)


plot(dl.basic.train.flatten_history)
str(dl.basic.train.flatten_history)

## BDL MCC Model Evaluation ----------------------------------------------------
put_log("Evaluating DL Model...")
bdl.eval.result <- dl.basic.model |> evaluate(x.test.flatten, y.test.flatten.cat)
put_log("DL Model evaluation result:
%1", capture.output(str(bdl.eval.result)))
# List of 2
#  $ accuracy: num 0.897
#  $ loss    : num 0.314

put_end_date(start)
# Time difference of 1.668308 mins

bdl.preds <- dl.basic.model |> predict(x.test.flatten) 
put_end_date(start)
# Time difference of  mins

colnames(bdl.preds) <- Y.Labels
head(bdl.preds[,1:5])
#                 #            $            &            @            0
# [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
# [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
# [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
# [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
# [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
# [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
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


# y.test.flatten
# as.integer(y.test.flatten)

bdl.pred.values.idx <- bdl.predictions$numpy()
head(bdl.pred.values.idx)

bdl.pred.values <- Y.Labels[bdl.pred.values.idx]
head(bdl.pred.values)

dl.basic.accuracy <- mean(bdl.pred.values.idx == as.integer(y.test.flatten))
put_log("The overall Basic `DL MCC` Model accuracy: %1",dl.basic.accuracy)
# 0.897195136631756

## Evaluation Results: Visualization ------------------------------------------

dl.basic.accuracy.by_class <- MCClassifier.accuracy.by_class(Y.Labels,
                                                             y.test.flatten,
                                                             bdl.pred.values)
dl.basic.accuracy.by_class
{
#' class  accuracy
    #' # 1.0000000
    #' $ 1.0000000
    #' & 1.0000000
    #' @ 1.0000000
    #' 0 0.9612676
    #' 1 0.7605634
    #' 2 0.8967136
    #' 3 0.9530516
    #' 4 0.9131455
    #' 5 0.8697183
    #' 6 0.9166667
    #' 7 0.9753521
    #' 8 0.9424883
    #' 9 0.8345070
    #' A 0.8697183
    #' B 0.9072770
    #' C 0.9319249
    #' D 0.9225352
    #' E 0.9272300
    #' F 0.9436620
    #' G 0.6737089
    #' H 0.9190141
    #' I 0.6901408
    #' J 0.9307512
    #' K 0.9295775
    #' L 0.4518779
    #' M 0.9577465
    #' N 0.9190141
    #' P 0.9518779
    #' Q 0.7523474
    #' R 0.9295775
    #' S 0.8908451
    #' T 0.9190141
    #' U 0.9401408
    #' V 0.8920188
    #' W 0.9671362
    #' X 0.9366197
    #' Y 0.9049296
    #' Z 0.9084507
  invisible(NULL)
}

# Confusion Matrix data suitable for Visualization using the `cvms` package:
# Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
put_log("`BDL MCC` Model: Creating a confusion matrix in a format suitable for visualization 
using the `cvms` package...")
dl.basic.conf.mx <- confusion_matrix(as.character(y.test.flatten),
                                     as.character(bdl.pred.values))
put_log("The confusion matrix based on the `BDL MCC` Model evaluation results has been created:
%1", capture.output(dl.basic.conf.mx))  


put_log("Plotting ROC curves for the Multiclass Classifier (MCC) based on the current model...")
dl.basic.roc_curves <- plot.ROC.curves(y.test.flatten,
                                       bdl.preds)


put_log("`BDL MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(Y.Labels,
                            dl.basic.accuracy.by_class,
                            title.prefix = "Basic DL Multiclass")

put_log("Plotting the confusion matrix, please wait...")
start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)


dev.off()
plot_confusion_matrix(dl.basic.conf.mx,
                      palette = "Greens",
                      font_counts = font(size = 3,
                                         color = "red"),
                      add_normalized = FALSE,
                      add_col_percentages = FALSE,
                      add_row_percentages = FALSE)

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

put_end_date(start)
log_close()


