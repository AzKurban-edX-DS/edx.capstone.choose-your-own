## Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
put_log("Evaluating the pre-trained CNN-based Multiclass Classifier Model...")

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

## Preparing Validation Set ----------------------------------------------------
put_log("Preparing a Test Set...")
start <- put_start_date()

stopifnot(exists("x3d.test"))

lbl.groups <- ds.get_classIDs.grouped(x3d.test)
#y_cnn.test <- as.factor(rownames(x3d.test))
y_cnn.test <- lbl.groups$classID
length(y_cnn.test)
#> [1] 33267

y_cnn.test.cat <- to_categorical(y_cnn.test)
colnames(y_cnn.test.cat) <- y.labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y_cnn.test.cat)))
#> [1] 33267    39

# str(y_cnn.test.cat)
head(y_cnn.test.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

put_log("Reshaping the Test Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x_cnn.test <- array_reshape(x3d.test, 
                            c(nrow(x3d.test), 
                              n.img_rows, 
                              n.img_cols, 
                              1))

put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

## class Identifies: Quick Analysis ---------------------------------------------

y_cnn.test.chars <- lbl.groups$groupByClass
#str(y_cnn.test.chars)

char_n.max <- max(y_cnn.test.chars$n)
# 853
char_n.max == min(y_cnn.test.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Test Set is as follows:
%1", capture.output(print(y_cnn.test.chars, n = nrow(y_cnn.test.chars))))
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
  cnn_multiclass.eval.result <- cnn_multiclass.model |> evaluate(x_cnn.test, y_cnn.test.cat)
  put_log("CNN Model evaluation result:
%1", capture.output(str(cnn_multiclass.eval.result)))
  # List of 2
  #  $ accuracy: num 0.861
  #  $ loss    : num 2.83
  
  put_end_date(start)
  
  # model prediction
  put_log("CNN Model: constructing predictions...")
  
  cnn_multiclass.preds <- cnn_multiclass.model |> predict(x_cnn.test) 
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
  
  cnn_multiclass.accuracy <- mean(cnn.prediction.values == y_cnn.test)
  put_log("CNN Model accuracy: %1", cnn_multiclass.accuracy)
  # CNN Model accuracy: 0.920281359906213
  #> For final test (expected value):
  #> CNN Model accuracy: 0.910364145658263
  
  # cnn_multiclass.conf.mx0 <- confusionMatrix(y_cnn.test, cnn.prediction.values)
  
  #### Confusion Matrix Visualization using `cvms` package
  # Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
  cnn_multiclass.conf.mx <- confusion_matrix(as.character(y_cnn.test),
                                             as.character(cnn.prediction.values))
  
  #### Accuracy by Class ---
  y_cnn.test.idx <- seq_len(length(y_cnn.test))
  # head(y_cnn.test.idx)
  
  cnn_multiclass.accuracy_by_class <- sapply(y.labels, function(label) {
    idx <- y_cnn.test.idx[y_cnn.test == label]
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
  # Reference:
  # https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/
  
  # Calculate ROC curve for each class
  roc_curves <- lapply(as.integer(y.labels), function(class.idx) {
    bin_labels <- y_cnn.test.cat[, class.idx]
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
head(cnn.prediction.values)
# [1] H 0 Y K 0 D
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

head(y_cnn.test)
# [1] H 0 4 K 0 D
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

err.idx <- which(cnn.prediction.values != y_cnn.test)
length(err.idx)
#  416
err.head.idx <- head(err.idx)
# 3 14 30 43 63 76


err.pred.values <- cnn.prediction.values[err.idx]
head(err.pred.values)

err.test.values <- y_cnn.test[err.idx]
head(err.test.values)

err.head.img <- x_cnn.test[err.head.idx,,,1]
dim(err.head.img)

# par(mfrow = c(6, 1))
# for(i in err.head.idx) {
#   char.image(x_cnn.test[i,,,1])
# }
# par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
## Close Log ------------------------------------------------------------------
log_close()
