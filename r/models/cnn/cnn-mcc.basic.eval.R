#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic CNN MCC Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".basic-cnn-model.evaluation.setup")
stopifnot(file.exists(cnn_mcc.x3d.test_set.bakup))

### Loading the Pre-trained CNN-based Multiclass Classifier Model ---------------
put_log("Evaluating the pre-trained CNN-based Multiclass Classifier Model...")

if (!exists("cnn_mcc.model")) {
  stopifnot(file.exists(cnn_mcc.model.file_path))
  
  put_log("Loading the pre-trained CNN-based Multiclass Classifier model from the backup file...")
  cnn_mcc.model <- load_model(cnn_mcc.model.file_path)
  put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnn_mcc.model.file_path)
  
  if(file.exists(cnn_mcc.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_mcc.train_history <- readRDS(cnn_mcc.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_mcc.train_history.file_path)
  }
}

put_log("The Pre-trained CNN-Based Multiclass Classifier Model detains:

%1", capture.output(cnn_mcc.model))

if(exists("cnn_mcc.train_history")){
  plot(cnn_mcc.train_history)
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

put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x_test)))
# shape(33228, 28, 28, 1)

#### class Identifies: Quick Analysis ---------------------------------------------

y_test.chars <- class.groups$groupByClass
#str(y_test.chars)

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
}

log_close()

## Evaluating the CNN-based Multiclass Classifier Model ----------------------
open_logfile(".basic-cnn-model.evaluation")

put_log("Evaluating the pre-trained Multiclass Classifier model...")
start <- put_start_date()

if(file.exists(cnn_mcc.model.eval.file_path)) {
  put_log("Loading the Multiclass Classifier model Evaluation Results...")
  load(cnn_mcc.model.eval.file_path)
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been loaded from the following backup file:
%1", cnn_mcc.model.eval.file_path)
  put_end_date(start)
} else {
  put_log("Evaluating CNN Model...")
  cnn_mcc.eval.result <- cnn_mcc.model |> evaluate(x_test, y_test.cat)
  put_log("CNN MCC Model evaluation has been completed with the following result:
%1", capture.output(cnn_mcc.eval.result))
  # $accuracy
  # [1] 0.9193752
  # 
  # $loss
  # [1] 0.2359146
  
  put_end_date(start)
  
  # model prediction
  put_log("CNN Model: constructing predictions...")
  
  cnn_mcc.preds <- cnn_mcc.model |> predict(x_test) 
  put_log("CNN Model: predictions have been constructed.")
  put_end_date(start)
  # Time difference of 1.502232 mins
  
  put_log("Saving the Multiclass Classifier model Evaluation Results...")
  save(cnn_mcc.eval.result,
       cnn_mcc.preds,
       file = cnn_mcc.model.eval.file_path)
  
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", cnn_mcc.model.eval.file_path)

}

put_log("CNN MCC Model evaluation result:
%1", capture.output(cnn_mcc.eval.result))
# $accuracy
# [1] 0.9193752
# 
# $loss
# [1] 0.2359146

dim(cnn_mcc.preds)

colnames(cnn_mcc.preds) <- Y.Labels
head(cnn_mcc.preds[,1:5])

cnn_preds.ts <- as_tensor(cnn_mcc.preds)
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
cnn_mcc.prediction.values <- Y.Labels[cnn.prediction.values.idx]
head(cnn_mcc.prediction.values)

# Confusion Matrix data suitable for Visualization using the `cvms` package:
# Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
put_log("`CNN MCC` Model Evaluation: Creating a confusion matrix in a format 
suitable for visualization using the `cvms` package...")
cnn_mcc.conf.mx <- confusion_matrix(as.character(y_test),
                                           as.character(cnn_mcc.prediction.values))
put_log("The confusion matrix based on the `CNN MCC` Model evaluation results has been created:
%1", cnn_mcc.conf.mx)  

#### Accuracy by Class ---
y_test.idx <- seq_len(length(y_test))
# head(y_test.idx)

cnn_mcc.accuracy_by_class <- MCClassifier.accuracy.by_class(Y.Labels,
                                                                   y_test,
                                                                   cnn_mcc.prediction.values)
#### ROC Curves
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

put_log("Calculating a ROC curve for each class...")
cnn_mcc.roc_curves <- calc.roc_curves.cnn(y_test.cat,
                                          cnn_mcc.preds,
                                          Y.Labels)

cnn_mcc.accuracy <- mean(cnn_mcc.prediction.values == y_test)
put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnn_mcc.accuracy)
# 0.919375225713254
#> For final test (expected value):
#> CNN Model accuracy: 0.910364145658263

# cnn_mcc.conf.mx0 <- confusionMatrix(y_test, cnn_mcc.prediction.values)


### Logging Accuracies by class -------------------------------------------------

put_log("The total set of accuracies by class is as follows:
%1", capture.output(cnn_mcc.accuracy_by_class))
{
# class  accuracy
    #' # 1.0000000
    #' $ 1.0000000
    #' & 1.0000000
    #' @ 0.9976553
    #' 0 0.9847597
    #' 1 0.7713951
    #' 2 0.9331770
    #' 3 0.9777257
    #' 4 0.9425557
    #' 5 0.9320047
    #' 6 0.9531067
    #' 7 0.9859320
    #' 8 0.9484174
    #' 9 0.8944900
    #' A 0.9202814
    #' B 0.9284877
    #' C 0.9624853
    #' D 0.9449004
    #' E 0.9495897
    #' F 0.9671747
    #' G 0.7162954
    #' H 0.9437280
    #' I 0.6365768
    #' J 0.9495897
    #' K 0.9577960
    #' L 0.5087925
    #' M 0.9812427
    #' N 0.9542790
    #' P 0.9788980
    #' Q 0.7760844
    #' R 0.9495897
    #' S 0.9120750
    #' T 0.9577960
    #' U 0.9671747
    #' V 0.9495897
    #' W 0.9812427
    #' X 0.9554513
    #' Y 0.8944900
    #' Z 0.9261430
}

log_close()

### Visualization --------------------------------------------------------------

#### Class-wise accuracy
put_log("`CNN MCC` Model: Plotting bar chart of per-class accuracy...")
plot_bars.accuracy.by_class(Y.Labels,
                            cnn_mcc.accuracy_by_class[, 1],
                            title.prefix = "CNN-based Multiclass")

# Plot ROC curves
plot(cnn_mcc.roc_curves[[1]], main = "ROC Curves for CNN-based Multiclass Classification")
for (class.idx in 2:N.classes) {
  lines(cnn_mcc.roc_curves[[class.idx]], col = class.idx)
}

# put_log("Plotting the confusion matrix, please wait...")
# start <- put_start_date()
# cl <- makeCluster(N_pcCores)
# registerDoParallel(cl)
# 
# dev.off()
# plot_confusion_matrix(cnn_mcc.conf.mx$`Confusion Matrix`[[1]],
#                       palette = "Greens",
#                       font_counts = font(size = 3,
# 
#                                          color = "red"),
#                       add_normalized = FALSE,
#                       add_col_percentages = FALSE,
#                       add_row_percentages = FALSE)
# stopCluster(cl)
# stopImplicitCluster()
# put_end_date(start)

### Review Some Errors --------------------------------------------------------- 

recg.err.info <- recognition_err.table(cnn_mcc.prediction.values,
                                        y_test,
                                        x.test.files)
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
}

# dev.off()
print.image_grid(recg.err.info)
# dev.off()
# str(recg.err.info)

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
log_close()
