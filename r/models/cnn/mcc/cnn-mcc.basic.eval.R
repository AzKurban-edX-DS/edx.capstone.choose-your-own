#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic CNN MCC Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".basic-cnn-model.eval.setup")
stopifnot(file.exists(cnn_mcc.x3d.test_set.bakup))

### Init File Paths ------------------------------------------------------------
cnn_mcc.eval.result.backup <- file.path(data.cnn_mcc.basic.dir,
                                       "cnn_mcc.eval.result.rds")

### Loading the Pre-trained CNN-based Multiclass Classifier Model ---------------
put_log("Evaluating the pre-trained CNN-based Multiclass Classifier Model...")

if (!exists("cnn_mcc.model")) {
  stopifnot(file.exists(cnn_mcc.basic.file_path))
  
  put_log("Loading the pre-trained CNN-based Multiclass Classifier model from the backup file...")
  cnn_mcc.model <- keras3::load_model(cnn_mcc.basic.file_path)
  put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnn_mcc.basic.file_path)
  
  if(file.exists(cnn_mcc.basic.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_mcc.train_history <- readRDS(cnn_mcc.basic.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.basic.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_mcc.basic.train_history.file_path)
  }
}


put_log("The Pre-trained CNN-Based Multiclass Classifier Model detains:

%1", capture.output(cnn_mcc.model))

if(exists("cnn_mcc.train_history")){
  plot(cnn_mcc.train_history)
  rm(cnn_mcc.train_history)
} 

### Preparing Validation Data ---------------------------------------------------
put_log("Preparing a Test Set...")
start <- put_start_date()

put_log("Loading the Test Set from backup...")
x3d.test_set <- readRDS(cnn_mcc.x3d.test_set.bakup)

put_log("The Test Set has been loaded from the following file:
%1", cnn_mcc.x3d.test_set.bakup)


put_log("The Test Set data is stored in the object `x3d.test_set`, 
having the following structure:
%1", capture.output(str(x3d.test_set)))

class.groups <- ds.get_classIDs.grouped(x3d.test_set$x.test)

y_test <- class.groups$classID
length(y_test)
#> [1] 33267

y_test.cat <- to_categorical(y_test)
colnames(y_test.cat) <- Y.Labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y_test.cat)))
#> [1] 33267    39

# str(y_test.cat)
head(y_test.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

put_log("Reshaping the Test Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x_test <- array_reshape(x3d.test_set$x.test, 
                        c(nrow(x3d.test_set$x.test), 
                          n.img_rows, 
                          n.img_cols, 
                          1))

x_test.files <- x3d.test_set$x.files
rm(x3d.test_set)

put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x_test)))
# shape(33228, 28, 28, 1)

#### class Identifies: Quick Analysis ---------------------------------------------

y_test.chars <- class.groups$groupByClass
#str(y_test.chars)

rm(class.groups)

char_n.max <- max(y_test.chars$n)
# 853
char_n.max == min(y_test.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Test Set is as follows:
%1", capture.output(print(y_test.chars, n = nrow(y_test.chars))))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         853
  #  2 $         853
  #  3 &         853
  #  4 @         853
  #  5 0         853
  #  6 1         853
  #  7 2         853
  #  8 3         853
  #  9 4         853
  # 10 5         853
  # 11 6         853
  # 12 7         853
  # 13 8         853
  # 14 9         853
  # 15 A         853
  # 16 B         853
  # 17 C         853
  # 18 D         853
  # 19 E         853
  # 20 F         853
  # 21 G         853
  # 22 H         853
  # 23 I         853
  # 24 J         853
  # 25 K         853
  # 26 L         853
  # 27 M         853
  # 28 N         853
  # 29 P         853
  # 30 Q         853
  # 31 R         853
  # 32 S         853
  # 33 T         853
  # 34 U         853
  # 35 V         853
  # 36 W         853
  # 37 X         853
  # 38 Y         853
  # 39 Z         853
  invisible()
}

rm(y_test.chars)
log_close()

## Evaluating the CNN-based Multiclass Classifier Model ----------------------
open_logfile(".basic-cnn-model.evaluation")

put_log("Evaluating the pre-trained Multiclass Classifier model...")
start <- put_start_date()

if(file.exists(cnn_mcc.eval.result.backup)) {
  put_log("Loading the Multiclass Classifier model Evaluation Results...")
  cnn_mcc.eval.result <- readRDS(cnn_mcc.eval.result.backup)
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been loaded from the following backup file:
%1", cnn_mcc.eval.result.backup)
  put_end_date(start)
} else {
  put_log("Evaluating CNN Model...")
  cnn_mcc.eval.result <- cnn_mcc.model |> evaluate(x_test, y_test.cat)
  put_log("CNN MCC Model evaluation has been completed with the following result:
%1", capture.output(cnn_mcc.eval.result))
  # $accuracy
  # [1] 0.8887953
  # 
  # $loss
  # [1] 0.3397374
  
  put_end_date(start)
  
  # model prediction
  put_log("CNN Model: constructing predictions...")
  
  cnn_mcc.eval.result$predicted.probs <- cnn_mcc.model |> predict(x_test) 
  put_log("CNN Model: predictions have been constructed.")
  put_end_date(start)
  # Time difference of 1.502232 mins

  dim(cnn_mcc.eval.result$predicted.probs)
  
  colnames(cnn_mcc.eval.result$predicted.probs) <- Y.Labels
  head(cnn_mcc.eval.result$predicted.probs[,1:5])
  
  cnn_preds.ts <- as_tensor(cnn_mcc.eval.result$predicted.probs)
  str(cnn_preds.ts)
  #> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>
  
  cnn_mcc.predictions <- cnn_preds.ts |> op_argmax(2)
  str(cnn_mcc.predictions)
  cnn_mcc.predictions
  #> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
  dim(cnn_mcc.predictions)
  #> [1] 684467
  
  cnn.prediction.values.idx <- cnn_mcc.predictions$numpy()
  head(cnn.prediction.values.idx)
  
  cnn_mcc.eval.result$predicted.values <- Y.Labels[cnn.prediction.values.idx]
  head(cnn_mcc.eval.result$predicted.values)
  
  cnn_mcc.eval.result$targets <- y_test
  
  rm(cnn_preds.ts,
     cnn_mcc.predictions,
     cnn.prediction.values.idx)
  
  put_log("Saving the Multiclass Classifier model Evaluation Results...")
  saveRDS(cnn_mcc.eval.result,
          file = cnn_mcc.eval.result.backup)
  
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", cnn_mcc.eval.result.backup)
}

put_log("CNN MCC Model evaluation result:
%1", capture.output(cnn_mcc.eval.result))
# $accuracy
# [1] 0.8887953
# 
# $loss
# [1] 0.3397374


cnn_mcc.accuracy <- mean(cnn_mcc.eval.result$predicted.values == y_test)
put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnn_mcc.accuracy)
# 0.888795259687278

rm(x_test,
   y_test,
   y_test.cat)

log_close()

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".basic-cnn-model.eval.visualization")
stopifnot(file.exists(model_visualization.shared.script.path))

cnn_mcc.eval.conf.mx.img_file <- file.path(cnn_mcc.basic.plots.dat.dir,
                                            "cnn_mcc.eval.confusion-matrix.png")

cnn_mcc.eval.plots_dat.file <- file.path(cnn_mcc.basic.plots.dat.dir,
                                          "cnn_mcc.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(cnn_mcc.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(cnn_mcc.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", cnn_mcc.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = cnn_mcc.eval.result$targets,
                                predicted.probabilities = cnn_mcc.eval.result$predicted.probs,
                                predicted.values = cnn_mcc.eval.result$predicted.values,
                                alg_name = "CNN Basic",
                                plots_dat.file = cnn_mcc.eval.plots_dat.file,
                                cm.export.img_file = cnn_mcc.eval.conf.mx.img_file,
                                cm.print.image = T)
}

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

if(!file.exists(cnn_mcc.eval.plots_dat.file)) {
  put_log("Saving the model-related plots input data object to file...")
  
  saveRDS(plots.dat,
          file = cnn_mcc.eval.plots_dat.file)
  
  put_log("The model-related plots input data object has been saved to the following file:
%1", cnn_mcc.eval.plots_dat.file)
}

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

## Review Some Errors --------------------------------------------------------- 

recg.err.info <- recognition_err.table(cnn_mcc.eval.result$predicted.values,
                                        cnn_mcc.eval.result$targets,
                                        x_test.files)
put_log("First 30 prediction errors:
%1", capture.output(head(recg.err.info, n = 30)))
{
  #    predicted actual                                         file
  # 1          9      G  data/raw/Vaibs.HW-Chars/Train/G/_1_2079.jpg
  # 2          L      1    data/raw/Vaibs.HW-Chars/Train/1/21673.jpg
  # 3          U      V      data/raw/Vaibs.HW-Chars/Train/V/610.jpg
  # 4          S      5    data/raw/Vaibs.HW-Chars/Train/5/29573.jpg
  # 5          S      5    data/raw/Vaibs.HW-Chars/Train/5/21880.jpg
  # 6          U      4    data/raw/Vaibs.HW-Chars/Train/4/31650.jpg
  # 7          1      L     data/raw/Vaibs.HW-Chars/Train/L/5851.jpg
  # 8          V      Y       data/raw/Vaibs.HW-Chars/Train/Y/83.jpg
  # 9          0      D     data/raw/Vaibs.HW-Chars/Train/D/2412.jpg
  # 10         Q      G  data/raw/Vaibs.HW-Chars/Train/G/_1_3172.jpg
  # 11         1      I     data/raw/Vaibs.HW-Chars/Train/I/9487.jpg
  # 12         9      Q  data/raw/Vaibs.HW-Chars/Train/Q/_1_1842.jpg
  # 13         C      E data/raw/Vaibs.HW-Chars/Train/E/_1_14130.jpg
  # 14         1      I    data/raw/Vaibs.HW-Chars/Train/I/12125.jpg
  # 15         L      I    data/raw/Vaibs.HW-Chars/Train/I/12325.jpg
  # 16         N      H      data/raw/Vaibs.HW-Chars/Train/H/682.jpg
  # 17         I      L    data/raw/Vaibs.HW-Chars/Train/L/11802.jpg
  # 18         Q      A data/raw/Vaibs.HW-Chars/Train/A/_1_10481.jpg
  # 19         1      L    data/raw/Vaibs.HW-Chars/Train/L/16686.jpg
  # 20         I      L     data/raw/Vaibs.HW-Chars/Train/L/6876.jpg
  # 21         I      L     data/raw/Vaibs.HW-Chars/Train/L/8059.jpg
  # 22         L      1    data/raw/Vaibs.HW-Chars/Train/1/34280.jpg
  # 23         B      6    data/raw/Vaibs.HW-Chars/Train/6/18502.jpg
  # 24         3      2    data/raw/Vaibs.HW-Chars/Train/2/35718.jpg
  # 25         8      G   data/raw/Vaibs.HW-Chars/Train/G/_1_272.jpg
  # 26         I      1    data/raw/Vaibs.HW-Chars/Train/1/24956.jpg
  # 27         R      T      data/raw/Vaibs.HW-Chars/Train/T/789.jpg
  # 28         G      5     data/raw/Vaibs.HW-Chars/Train/5/3938.jpg
  # 29         6      B  data/raw/Vaibs.HW-Chars/Train/B/_1_4978.jpg
  # 30         V      U    data/raw/Vaibs.HW-Chars/Train/U/15093.jpg
  invisible()
}

# dev.off()
print.image_grid(recg.err.info)
# dev.off()
# str(recg.err.info)
rm(recg.err.info)

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

rm(cnn_mcc.eval.result)

log_close()
