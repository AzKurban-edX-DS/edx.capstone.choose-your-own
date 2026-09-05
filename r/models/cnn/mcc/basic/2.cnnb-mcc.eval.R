#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic CNN MCC Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".cnnb-mcc.evaluation")

stopifnot(file.exists(cnnb_mcc.file,
                      ds28x28.split.train_0.8.backup.file))

start <- put_start_date()

## Preparing a Test Set for the Model Evaluation Job ---------------------------

put_log("Loading the Test Set of 28x28x1-shape image data...")
test_set <- load28x28x1.test_set(ds28x28.split.train_0.8.backup.file)
put_log("The Training Set of 28x28x1-shape image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)

x_test <- test_set$x

y_test <- test_set$class_groups$classID


stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0,
          length(y_test) == nrow(x_test))

str(x_test)
str(y_test)

dim(x_test)
dim(y_test)

x_test.files <- test_set$files

y_test.cat <- to_categorical(y_test)
colnames(y_test.cat) <- Y.Labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y_test.cat)))
# [1] 132912     39

# str(y_test.cat)
head(y_test.cat)
{
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  invisible()
}

### Size of the Training Set by Class ------------------------------------------

put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(test_set$class_groups$groupByClass, n = N.classes)))
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

## CNNB MCC Model Evaluation ---------------------------------------------------

cnnb_mcc.eval.result.file <- file.path(data.cnnb_mcc.dir,
                                       "cnnb_mcc.eval.result.rds")

put_log("Loading the pre-trained CNN-based Multiclass Classifier model...")
cnnb_mcc <- keras3::load_model(cnnb_mcc.file)
put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnnb_mcc.file)

put_log("The Pre-trained model summary:
%1", capture.output(cnnb_mcc))

put_log("Evaluating the pre-trained CNNB MCC model...")
start <- put_start_date()

put_log("Evaluating CNN Model...")
cnnb_mcc.eval.result <- cnnb_mcc |> evaluate(x_test, y_test.cat)
put_log("CNN MCC Model evaluation has been completed with the following result:
%1", capture.output(cnnb_mcc.eval.result))
# $accuracy
# [1] 0.8887953
# 
# $loss
# [1] 0.3397374

put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")

cnnb_mcc.eval.result$predicted.probs <- cnnb_mcc |> predict(x_test) 
put_log("CNN Model: predictions have been constructed.")
put_end_date(start)
# Time difference of 1.502232 mins

dim(cnnb_mcc.eval.result$predicted.probs)

colnames(cnnb_mcc.eval.result$predicted.probs) <- Y.Labels
head(cnnb_mcc.eval.result$predicted.probs[,1:5])

cnnb_mcc.preds.ts <- as_tensor(cnnb_mcc.eval.result$predicted.probs)
str(cnnb_mcc.preds.ts)
#> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>

cnnb_mcc.predictions <- cnnb_mcc.preds.ts |> op_argmax(2)
str(cnnb_mcc.predictions)
cnnb_mcc.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(cnnb_mcc.predictions)
#> [1] 684467

cnnb_mcc.prediction.values.idx <- cnnb_mcc.predictions$numpy()
head(cnnb_mcc.prediction.values.idx)

cnnb_mcc.eval.result$predicted.values <- Y.Labels[cnnb_mcc.prediction.values.idx]
head(cnnb_mcc.eval.result$predicted.values)

cnnb_mcc.eval.result$targets <- y_test

rm(cnnb_mcc.preds.ts,
   cnnb_mcc.predictions,
   cnnb_mcc.prediction.values.idx,
   x_test,
   y_test,
   y_test.cat)

cnnb_mcc.accuracy <- 
  mean(cnnb_mcc.eval.result$predicted.values == cnnb_mcc.eval.result$targets)

put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnnb_mcc.accuracy)
# 0.888795259687278

put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", cnnb_mcc.eval.result.file)

put_log("CNN MCC Model evaluation result:
%1", capture.output(cnnb_mcc.eval.result))
# $accuracy
# [1] 0.8887953
# 
# $loss
# [1] 0.3397374

put_log("Saving the Multiclass Classifier model Evaluation Results...")
saveRDS(cnnb_mcc.eval.result,
        file = cnnb_mcc.eval.result.file)

## Review Some Errors --------------------------------------------------------- 

# [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

recg.err.info <- recognition_err.table(cnnb_mcc.eval.result$predicted.values,
                                        cnnb_mcc.eval.result$targets,
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
# str(recg.err.info)

## Finalizing ------------------------------------------------------------------

rm(cnnb_mcc.eval.result,
   recg.err.info)

log_close()
# =========================================================================
# Log End Time: 2026-09-06 01:34:41.435753
# Log Elapsed Time: 0 00:00:07
# =========================================================================
