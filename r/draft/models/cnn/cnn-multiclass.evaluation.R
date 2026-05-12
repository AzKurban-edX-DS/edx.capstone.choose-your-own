#### Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
put_log("Evaluating the pre-trained CNN-based Multiclass Classifier Model...")

#### Loading the Pre-trained CNN-based Multiclass Classifier Model ----------------------------------------------
if (!exists("cnn.multiclass.model")) {
  stopifnot(file.exists(cnn.multiclass.model.file_path))
  put_log("Loading the pre-trained CNN-based Multiclass Classifier model from the backup file...")
  cnn.multiclass.model <- load_model(cnn.multiclass.model.file_path)
  put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnn.multiclass.model.file_path)
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
                              img_rows, 
                              img_cols, 
                              1))

put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x3d.test)))
# shape(33267, 28, 28)

# class Identifies: Quick Analysis ---------------------------------------------

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

#### Evaluating the CNN-based Multiclass Classifier Model ----------------------
put_log("Evaluating CNN Model...")
start <- put_start_date()
cnn.eval.result <- cnn.multiclass.model |> evaluate(x_cnn.test, y_cnn.test.cat)
put_log("CNN Model evaluation result:
%1", capture.output(str(cnn.eval.result)))
# List of 2
#  $ accuracy: num 0.861
#  $ loss    : num 2.83

put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")

cnn_preds <- cnn.multiclass.model |> predict(x_cnn.test) 
put_log("CNN Model: predictions have been constructed.")
put_end_date(start)
# Time difference of 1.502232 mins

dim(cnn_preds)

colnames(cnn_preds) <- y.labels
head(cnn_preds[,1:5])

cnn_preds.ts <- as_tensor(cnn_preds)
str(cnn_preds.ts)
#> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>

cnn.predictionns <- cnn_preds.ts |> op_argmax(2)
str(cnn.predictionns)
cnn.predictionns
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(cnn.predictionns)
#> [1] 684467
cnn.prediction.values.idx <- cnn.predictionns$numpy()
head(cnn.prediction.values.idx)
cnn.prediction.values <- y.labels[cnn.prediction.values.idx]
head(cnn.prediction.values)

cnn.model.accuracy <- mean(cnn.prediction.values == y_cnn.test)
put_log("CNN Model accuracy: %1", cnn.model.accuracy)
# CNN Model accuracy: 0.920942675925091
#> For final test (expected value):
#> CNN Model accuracy: 0.910364145658263

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
### Close Log ------------------------------------------------------------------
log_close()
