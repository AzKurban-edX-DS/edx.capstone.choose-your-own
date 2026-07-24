#%%%%%%%%%%%%%%%%%%%%%
# BDL MCC  Model
#%%%%%%%%%%%%%%%%%%%%%

# Basic Deep Learning Multiclass Classifier (BDL MCC)  Model

# References:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# ref.bib: DL_R3_E2-S7.3

## Prepare Input Datasets -----------------------------------------------------
open_logfile(".dl.basic-model.prepare-ds")
stopifnot(file.exists(my_emnist.split.file_path))

start <- put_start_date()

### Loading Split Flattened Dataset allocated 20% for the Test Set -------------

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.split.file_path)
str(ds)

x_train <- ds$train$x

put_log("The Train set is balanced with respect to the set of classes:
%1", capture.output(print(ds$train$class_groups$groupByClass, n = N.classes)))
{
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
  invisible(NULL)
}

y_train <- ds$train$class_groups$classID

stopifnot(sum(as.character(y_train) != rownames(x_train)) == 0)
stopifnot(nrow(x_train) == length(y_train))

x_test <- ds$test$x

put_log("The Test set is balanced with respect to the set of classes:
%1", capture.output(print(ds$test$class_groups$groupByClass, n = N.classes)))
{
  # # A tibble: 39 × 2
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
  invisible(NULL)
}

y_test <- ds$test$class_groups$classID

stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0)
stopifnot(nrow(x_test) == length(y_test))

rm(ds)
log_close()

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

y_test.cat <- to_categorical(y_test)
colnames(y_test.cat) <- Y.Labels
dim(y_test.cat)
str(y_test.cat)
head(y_test.cat)

### Analyzing Image Data --------------------------------------------------------
max_img_pixels <- max(rowSums(x_train))
put_log("The maximum number of pixels in the Train Set images is as follows: %1",
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

log_close()

## Init DL Basic Model Paths ---------------------------------------------------

open_logfile(".dl.basic-model.build")

dl.basic.tuning.dir <- file.path(data.dl_basic.dir,
                                            "tuning")
if(!dir.exists(dl.basic.tuning.dir))
  dir.create(dl.basic.tuning.dir)

dl.basic.layers_dynamic.dir <- file.path(dl.basic.tuning.dir, "layers-dynamic")

if(!dir.exists(dl.basic.layers_dynamic.dir))
  dir.create(dl.basic.layers_dynamic.dir)

dl.basic.keras_tuner.dir <- file.path(dl.basic.tuning.dir, "keras-tuner")

if(!dir.exists(dl.basic.keras_tuner.dir))
  dir.create(dl.basic.keras_tuner.dir)

dl.basic.checkpoints.dir <- file.path(data.dl_basic.dir,
                                            "checkpoints")
if(!dir.exists(dl.basic.checkpoints.dir))
  dir.create(dl.basic.checkpoints.dir)

dl.basic.checkpoint.file_path <- 
  file.path(dl.basic.checkpoints.dir, 
            "dl.basic.{epoch:02d}-{val_loss:.2f}.keras")

dl.basic.model.file_path <- file.path(data.dl_basic.dir, 
                             "dl.basic.pre-trained.model.keras")

dl.basic.model.train_history.file_path <- file.path(data.dl_basic.dir, 
                             "dl.basic.model.train_history.bak.rds")

bdl.eval.result.file <- file.path(data.dl_basic.dir,
                                        "bdl.eval.result.rds")

## Building Basic DL MCC Model -------------------------------------------------

n.input_shape <- ncol(x_train)
# 784

if(file.exists(dl.basic.model.file_path)) {
  put_log("Loading pre-trained BDL MCC Model...")
  
  dl.basic.model <- keras3::load_model(dl.basic.model.file_path)
  
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
  #' *** Defining & Compiling the Basic DL MCC Model ***************************
  
  n.input_shape <- ncol(x_train)
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
  
  #' *** Training the Basic DL MCC Model ***************************************
  
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
    fit(x_train, 
        y_train.cat, 
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
  saveRDS(dl.basic.train_history,
          file = dl.basic.model.train_history.file_path)
  
  put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", dl.basic.model.train_history.file_path)
  put_end_date(start)
}

rm(x_train,
   y_train.cat)

put_log("The trained Basic DL MCC Model summary:
%1", dl.basic.model)

plot(dl.basic.train_history)

put_log("Structure of the Basic DL MCC Model training history:
%1", capture.output(str(dl.basic.train_history)))

rm(dl.basic.train_history)
log_close()
# Log Elapsed Time: 0 00:04:07
## BDL MCC Model Evaluation ----------------------------------------------------
open_logfile(".dl.basic-model.evaluate")

if(file.exists(bdl.eval.result.file)) {
  put_log("Loading the BDL MCC Model Evaluation Result object...")
  bdl.eval.result <- readRDS(bdl.eval.result.file)
  
  put_log("The BDL MCC Model Evaluation Result object has been loaded 
from the following file:
%1", bdl.eval.result.file)
} else {
  put_log("Evaluating DL Model...")
  bdl.eval.result <- dl.basic.model |> evaluate(x_test, y_test.cat)
  put_log("DL Model evaluation result:
%1", capture.output(str(bdl.eval.result)))
  # List of 2
  #  $ accuracy: num 0.897
  #  $ loss    : num 0.314
  
  put_log("The overall Basic DL MCC Model evaluation accuracy: %1",
          bdl.eval.result$accuracy)
  # 0.898338750451427
  
  
  put_end_date(start)
  # Time difference of 1.668308 mins
  
  bdl.eval.result$predicted.probs <- dl.basic.model |> predict(x_test) 
  put_end_date(start)
  # Time difference of  mins
  
  colnames(bdl.eval.result$predicted.probs) <- Y.Labels
  head(bdl.eval.result$predicted.probs[,1:5])
  #                 #            $            &            @            0
  # [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
  # [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
  # [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
  # [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
  # [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
  # [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
  dim(bdl.eval.result$predicted.probs)
  #> [1] 33228    39
  
  bdl.preds.ts <- as_tensor(bdl.eval.result$predicted.probs)
  str(bdl.preds.ts)
  #> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>
  
  bdl.predictions <- bdl.preds.ts |> op_argmax(2)
  bdl.predictions
  #> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
  shape(bdl.predictions)
  #> [1] 33228
  # bdl.predictions$numpy()
  
  
  bdl.pred.values.idx <- bdl.predictions$numpy()
  head(bdl.pred.values.idx)
  
  bdl.eval.result$predicted.values <- Y.Labels[bdl.pred.values.idx]
  head(bdl.eval.result$predicted.values)
  
  bdl.eval.result$targets <- y_test
  
  rm(bdl.preds.ts,
     bdl.predictions,
     bdl.pred.values.idx)
  
  put_log("Saving the BDL MCC Model History...")
  saveRDS(bdl.eval.result,
          file = bdl.eval.result.file)
  
  put_log("The BDL MCC Model History has been trained 
and saved in the following file:
  %1", bdl.eval.result.file)
  put_end_date(start)
}

# dl.basic.accuracy <- mean(bdl.pred.values.idx == as.integer(y_test))
# put_log("The overall Basic `DL MCC` Model accuracy: %1",dl.basic.accuracy)
# 0.898338750451427

rm(x_test,
   y_test,
   y_test.cat)

log_close()

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".dl-basic.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path))

dl_basic.plots.dat.dir <- file.path(data.dl_basic.dir, "plots.dat")

if(!dir.exists(dl_basic.plots.dat.dir))
  dir.create(dl_basic.plots.dat.dir)

dl_basic.eval.conf.mx.img_file <- file.path(dl_basic.plots.dat.dir,
                                            "dl-basic.eval.confusion-matrix.png")

dl_basic.eval.plots_dat.file <- file.path(dl_basic.plots.dat.dir,
                                          "dl-basic.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(dl_basic.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(dl_basic.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", dl_basic.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = bdl.eval.result$targets,
                                predicted.probabilities = bdl.eval.result$predicted.probs,
                                predicted.values = bdl.eval.result$predicted.values,
                                alg_name = "DL Basics",
                                plots_dat.file = dl_basic.eval.plots_dat.file,
                                cm.export.img_file = dl_basic.eval.conf.mx.img_file,
                                cm.print.image = T)
}

rm(bdl.eval.result)

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args,
   fit_rf.mtry_best)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

put_log("Saving the model-related plots input data object to file...")

saveRDS(plots.dat,
        file = dl_basic.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", dl_basic.eval.plots_dat.file)

# put_log("The Basic DL Model per-class accuracy:,
# %1", capture.output(plots.dat$PCA$acc.by_class))
{
#' class  accuracy
#'     # 1.0000000
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 1.0000000
#'     0 0.9577465
#'     1 0.6502347
#'     2 0.8673709
#'     3 0.9577465
#'     4 0.9295775
#'     5 0.8767606
#'     6 0.9213615
#'     7 0.9776995
#'     8 0.9225352
#'     9 0.8356808
#'     A 0.8685446
#'     B 0.9025822
#'     C 0.9366197
#'     D 0.9295775
#'     E 0.9284038
#'     F 0.9354460
#'     G 0.6913146
#'     H 0.9225352
#'     I 0.7453052
#'     J 0.9166667
#'     K 0.9237089
#'     L 0.5258216
#'     M 0.9565728
#'     N 0.9284038
#'     P 0.9589202
#'     Q 0.7746479
#'     R 0.9107981
#'     S 0.8826291
#'     T 0.9342723
#'     U 0.9589202
#'     V 0.8990610
#'     W 0.9671362
#'     X 0.9377934
#'     Y 0.8767606
#'     Z 0.9260563
  invisible(NULL)
}

rm(plots.dat)
log_close()


