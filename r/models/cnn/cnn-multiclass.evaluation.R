## Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
put_log("Evaluating the pre-trained CNN-based Multiclass Classifier Model...")

stopifnot(exists("x3d.test_set"))

## Loading the Pre-trained CNN-based Multiclass Classifier Model ----------------------------------------------
if (!exists("cnn_multiclass.model")) {
  stopifnot(file.exists(cnn_multiclass.model.file_path))
  
  put_log("Loading the pre-trained CNN-based Multiclass Classifier model from the backup file...")
  cnn_multiclass.model <- load_model(cnn_multiclass.model.file_path)
  put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnn_multiclass.model.file_path)
  
  if(file.exists(cnn_multiclass.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_multiclass.train_history <- readRDS(cnn_multiclass.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_multiclass.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_multiclass.train_history.file_path)
  }
}

put_log("The Pre-trained CNN-Based Multiclass Classifier Model detains:

%1", capture.output(cnn_multiclass.model))

if(exists("cnn_multiclass.train_history")){
  plot(cnn_multiclass.train_history)
} 

## Preparing Validation Data ---------------------------------------------------
put_log("Preparing a Test Set...")
start <- put_start_date()

class.groups <- ds.get_classIDs.grouped(x3d.test_set$x.test)

y.test <- class.groups$classID
length(y.test)
#> [1] 33267

y.test.cat <- to_categorical(y.test)
colnames(y.test.cat) <- y.labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y.test.cat)))
#> [1] 33267    39

# str(y.test.cat)
head(y.test.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

put_log("Reshaping the Test Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x.test <- array_reshape(x3d.test_set$x.test, 
                        c(nrow(x3d.test_set$x.test), 
                          n.img_rows, 
                          n.img_cols, 
                          1))

x.test.files <- x3d.test_set$x.files
str(x3d.test_set)

put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

## class Identifies: Quick Analysis ---------------------------------------------

y.test.chars <- class.groups$groupByClass
#str(y.test.chars)

char_n.max <- max(y.test.chars$n)
# 853
char_n.max == min(y.test.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Test Set is as follows:
%1", capture.output(print(y.test.chars, n = nrow(y.test.chars))))
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

## Init Evaluation Results File Path -------------------------------------------------------------
cnn_multiclass.model.eval.file_path <- file.path(data.dl.cnn.multiclass.dir, 
                                            "cnn.multiclass.model.eval.RData")
## Evaluating the CNN-based Multiclass Classifier Model ----------------------
put_log("Evaluating the pre-trained Multiclass Classifier model...")
start <- put_start_date()

if(file.exists(cnn_multiclass.model.eval.file_path)) {
  put_log("Loading the Multiclass Classifier model Evaluation Results...")
  load(cnn_multiclass.model.eval.file_path)
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been loaded from the following backup file:
%1", cnn_multiclass.model.eval.file_path)
  put_end_date(start)
} else {
  put_log("Evaluating CNN Model...")
  start <- put_start_date()
  cnn_multiclass.eval.result <- cnn_multiclass.model |> evaluate(x.test, y.test.cat)
  put_log("CNN Model evaluation result:
%1", capture.output(str(cnn_multiclass.eval.result)))
  # List of 2
    # $ accuracy: num 0.919
    # $ loss    : num 0.236

  put_end_date(start)
  
  # model prediction
  put_log("CNN Model: constructing predictions...")
  
  cnn_multiclass.preds <- cnn_multiclass.model |> predict(x.test) 
  put_log("CNN Model: predictions have been constructed.")
  put_end_date(start)
  # Time difference of 1.502232 mins
  
  dim(cnn_multiclass.preds)
  
  colnames(cnn_multiclass.preds) <- y.labels
  head(cnn_multiclass.preds[,1:5])
  
  cnn_preds.ts <- as_tensor(cnn_multiclass.preds)
  str(cnn_preds.ts)
  #> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>
  
  cnn_multiclass.predictions <- cnn_preds.ts |> op_argmax(2)
  str(cnn_multiclass.predictions)
  cnn_multiclass.predictions
  #> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
  dim(cnn_multiclass.predictions)
  #> [1] 684467
  cnn.prediction.values.idx <- cnn_multiclass.predictions$numpy()
  head(cnn.prediction.values.idx)
  cnn.prediction.values <- y.labels[cnn.prediction.values.idx]
  head(cnn.prediction.values)
  
  cnn_multiclass.accuracy <- mean(cnn.prediction.values == y.test)
  put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnn_multiclass.accuracy)
  # 0.919375225713254
  #> For final test (expected value):
  #> CNN Model accuracy: 0.910364145658263
  
  # cnn_multiclass.conf.mx0 <- confusionMatrix(y.test, cnn.prediction.values)
  
  #### Confusion Matrix Visualization using `cvms` package
  # Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
  cnn_multiclass.conf.mx <- confusion_matrix(as.character(y.test),
                                             as.character(cnn.prediction.values))
  
  #### Accuracy by Class ---
  y.test.idx <- seq_len(length(y.test))
  # head(y.test.idx)
  
  cnn_multiclass.accuracy_by_class <- sapply(y.labels, function(label) {
    idx <- y.test.idx[y.test == label]
    n <- length(idx)
    # put_log("Class of character `%1` has %2 items.",
    #         label, n)
    
    accuracy <- mean(cnn.prediction.values[idx] == label)
    
    put_log("Accuracy for the class `%1` (of size %2) is %3.",
            label, n, accuracy) 
    
    accuracy
  }) |> matrix(ncol = 1, dimnames = list(class = y.labels, "accuracy")) 
  
  dim(cnn_multiclass.accuracy_by_class)
  
  #### ROC Curves
  # References:
  # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
  # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
  
  # Calculate ROC curve for each class
  roc_curves <- lapply(as.integer(y.labels), function(class.idx) {
    bin_labels <- y.test.cat[, class.idx]
    roc_curve <- roc(bin_labels, cnn_multiclass.preds[, class.idx])
  })
  
  
  put_log("Saving the Multiclass Classifier model Evaluation Results...")
  save(cnn_multiclass.eval.result,
       cnn_multiclass.preds,
       cnn.prediction.values,
       roc_curves,
       cnn_multiclass.conf.mx,
       cnn_multiclass.accuracy_by_class,
       file = cnn_multiclass.model.eval.file_path)
  
  put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", cnn_multiclass.model.eval.file_path)
  
}

### Logging Accuracies by class -------------------------------------------------

put_log("The total set of accuracies by class is as follows:
%1", capture.output(df.cnn_multiclass.accuracy_by_class))

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

### Visualization --------------------------------------------------------------

#### Class-wise accuracy


str(cnn_multiclass.conf.mx)

plot_confusion_matrix(cnn_multiclass.conf.mx$`Confusion Matrix`[[1]],
                      palette = "Greens",
                      font_counts = font(size = 3.5,
                                         
                                         color = "red"),
                      add_normalized = FALSE,
                      add_col_percentages = FALSE,
                      add_row_percentages = FALSE)


# Plot ROC curves
plot(roc_curves[[1]], main = "ROC Curves for CNN-based Multiclass Classification")
for (class.idx in 2:N.classes) {
  lines(roc_curves[[class.idx]], col = class.idx)
}

data.frame(class = y.labels,
           accuracy = cnn_multiclass.accuracy_by_class[, 1]) |>
  ggplot(mapping = aes(x = class,
                       y = accuracy)) +
  geom_col(fill = "steelblue",
           color = "black") +
  labs(x = "Handwritten Character Classl",
       y = "Accuracy",
       title = "CNN-based Multiclass Classifier Model: Class-wise Evaluation Results") +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1),
                     expand = c(0, 0, 0.005, 0))



### Review Some Errors --------------------------------------------------------- 

recg.err.info <- recognition_err.table(cnn.prediction.values,
                                        y.test,
                                        x.test.files) |>
  print.image_grid()

put_log("First 30 prediction errors:
%1", capture.output(recg.err.info$err.table))
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

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
## Close Log ------------------------------------------------------------------
log_close()
