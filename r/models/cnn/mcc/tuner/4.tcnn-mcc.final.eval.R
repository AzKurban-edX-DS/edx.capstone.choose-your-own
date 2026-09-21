#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic CNN MCC Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".tuner.cnn-mcc.final-model.evaluation")
stopifnot(file.exists(tcnn_mcc.final.file),
          file.exists(ds28x28.split.train_0.8.backup.file))

### Preparing a Test Set for the Model Evaluation Job ---------------------------

put_log("Loading the Test Set of 28x28x1-shape image data...")
test_set <- load28x28x1.test_set(ds28x28.split.train_0.8.backup.file)
put_log("The Training Set of 28x28x1-shape image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)

x_test <- test_set$x
str(x_test)
dim(x_test)

y.test.groups <- test_set$class_groups

stopifnot(sum(as.character(y.test.groups$classID) != rownames(x_test)) == 0)

y_test <- as.array(as.integer(y.test.groups$classID) - 1)
str(y_test)
dim(y_test)

x_test.files <- test_set$files

#### Size of the Test Set by Class ------------------------------------------

put_log("The Training Set is balanced by the set of Classes:
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
  invisible()
}

rm(test_set)

### Loading the Pre-trained CNN-based Multiclass Classifier Model --------------

put_log("Loading pre-trained tuned Final MCC Model...")

cnn_mcc.final <- keras3::load_model(tcnn_mcc.final.file)

put_log("The Tuned Final MCC Model has been loaded from the backup file:
%1", tcnn_mcc.final.file)

put_log("The Tuned Final MCC Model Summary:
%1", cnn_mcc.final)
## Evaluating the CNN-based Multiclass Classifier Model ----------------------

put_log("Evaluating the pre-trained Multiclass Classifier model...")
start <- put_start_date()

put_log("Evaluating tuned Final CNN MCC Model...")
tcnn_mcc.final.eval.result <- cnn_mcc.final |> evaluate(x_test, y_test)
put_log("CNN MCC Model evaluation has been completed with the following result:
%1", capture.output(tcnn_mcc.final.eval.result))
# $accuracy
# [1] 0.9269291
# 
# $loss
# [1] 0.2350844


put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")

tcnn_mcc.final.eval.result$predicted.probs <- cnn_mcc.final |> predict(x_test) 
put_log("CNN Model: predictions have been constructed.")
put_end_date(start)
# Time difference of 1.502232 mins

dim(tcnn_mcc.final.eval.result$predicted.probs)

colnames(tcnn_mcc.final.eval.result$predicted.probs) <- Y.Labels
head(tcnn_mcc.final.eval.result$predicted.probs[,1:5])

cnn_preds.ts <- as_tensor(tcnn_mcc.final.eval.result$predicted.probs)
str(cnn_preds.ts)
#> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>

cnn_mcc.final.predictions <- cnn_preds.ts |> op_argmax(2)
str(cnn_mcc.final.predictions)
cnn_mcc.final.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(cnn_mcc.final.predictions)
#> [1] 684467

cnn.prediction.values.idx <- cnn_mcc.final.predictions$numpy()
head(cnn.prediction.values.idx)

tcnn_mcc.final.eval.result$predicted.values <- Y.Labels[cnn.prediction.values.idx]
head(tcnn_mcc.final.eval.result$predicted.values)

tcnn_mcc.final.eval.result$targets <- y_test

rm(cnn_preds.ts,
   cnn_mcc.final.predictions,
   cnn.prediction.values.idx)

put_log("Saving the Multiclass Classifier model Evaluation Results...")
saveRDS(tcnn_mcc.final.eval.result,
        file = tcnn_mcc.final.eval_result.file)

put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", tcnn_mcc.final.eval_result.file)

put_log("CNN MCC Model evaluation result:
%1", capture.output(tcnn_mcc.final.eval.result))
# $accuracy
# [1] 0.8887953
# 
# $loss
# [1] 0.3397374


cnn_mcc.final.accuracy <- mean(tcnn_mcc.final.eval.result$predicted.values == y_test)
put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnn_mcc.final.accuracy)
# 0.9269291

rm(x_test,
   y_test,
   y_test.cat)

log_close()



# Visualizing the Evaluation Results ------------------------------------------

open_logfile(".tuner.cnn-mcc.best-model.eval.visualization")

stopifnot(file.exists(model_visualization.shared.script.path))

cnn_mcc.final.eval.conf.mx.img_file <- file.path(cnn_mcc.tuner.plots.dat.dir,
                                            "dl-basic.eval.confusion-matrix.png")

cnn_mcc.final.eval.plots_dat.file <- file.path(cnn_mcc.tuner.plots.dat.dir,
                                          "dl-basic.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(cnn_mcc.final.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(cnn_mcc.final.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", cnn_mcc.final.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = tcnn_mcc.final.eval.result$targets,
                                predicted.probabilities = tcnn_mcc.final.eval.result$predicted.probs,
                                predicted.values = tcnn_mcc.final.eval.result$predicted.values,
                                alg_name = "CNN Basic",
                                plots_dat.file = cnn_mcc.final.eval.plots_dat.file,
                                cm.export.img_file = cnn_mcc.final.eval.conf.mx.img_file,
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

if(!file.exists(cnn_mcc.final.eval.plots_dat.file)) {
  put_log("Saving the model-related plots input data object to file...")
  
  saveRDS(plots.dat,
          file = cnn_mcc.final.eval.plots_dat.file)
  
  put_log("The model-related plots input data object has been saved to the following file:
%1", cnn_mcc.final.eval.plots_dat.file)
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
log_close()

## Review Some Errors --------------------------------------------------------- 

recg.err.info <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                        tcnn_mcc.final.eval.result$targets,
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

rm(tcnn_mcc.final.eval.result)

log_close()
