#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Ensemble Classifier for all labels 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Preparing Validation Set ----------------------------------------------------

open_logfile(".cnn.ensemble.load-final-test-data")
stopifnot(exists("ft.x3d.test_set"))

start <- put_start_date()

put_log("Evaluating the pre-trained CNN-based Binary Classifier Models...")

put_log("Preparing a Test Set...")

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

# class Identifies: Quick Analysis ---------------------------------------------

y_cnn.test.chars <- lbl.groups$groupByClass
#str(y_cnn.test.chars)

char_n.max <- max(y_cnn.test.chars$n)
# 853
char_n.max == min(y_cnn.test.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Test Set is as follows:
%1", capture.output(print(y_cnn.test.chars, n = nrow(y_cnn.test.chars))))
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

### Close Log ---------------------------------------------------------------
log_close()

##### Open Log for Ensemble Classifier -----------------------------------------
open_logfile(".cnn-model.ensemble-classifier")
##### Build & Test Ensemble Classifier -----------------------------------------


cnn_binary.ensemble.backup.path <- file.path(data.cnn.binary.dir, 
                                                 "cnn.lbl-models.ensemble.RData")



if (file.exists(cnn_binary.ensemble.backup.path)) {
  put_log("CNN: loading the Ensemble Classifier Results from cache file: 
%1", cnn_binary.ensemble.backup.path)
  load(cnn_binary.ensemble.backup.path)
  put_log("CNN: the Ensemble Classifier Results have been loaded from the cache file:
%1", cnn_binary.ensemble.backup.path)
} else {
  stopifnot(file.exists(hwChar.CNN.binCls.models.backup.path))
  hwChar.CNN.binCls.models <- readRDS(hwChar.CNN.binCls.models.backup.path)
  
  put_log("CNN: the pre-trained CNN-based Binary Classifier model list 
has been loaded from the backup file:
%1", hwChar.CNN.binCls.models.backup.path)
  
  put_log("CNN: the pre-trained CNN-based Binary Classifier model list 
object structure:
%1", str(hwChar.CNN.binCls.models))
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  start <- put_start_date()
  
  preds.mx <- sapply(y.labels, function(label) {
    plot(hwChar.CNN.binCls.models[[label]]$train_history)
    lbl.saved_model.path <- hwChar.CNN.binCls.models[[label]]$saved_model.filepath 
    stopifnot(file.exists(lbl.saved_model.path))

    put_log("Loading the pre-trained CNN-based Binary Classifier model for `%1` label from the backup file...",
            label)
    lbl.pretrained_model <- load_model(lbl.saved_model.path)  
    put_log("The pre-trained CNN-based Binary Classifier model for `%1` label
has been loaded from the following backup file:
%2", label, lbl.saved_model.path)
    
    put_log("Summary of the model for handwritten character `%1`:
%2",label, capture.output(summary(lbl.pretrained_model)))
    
    put_log("Making predictions for handwritten character '%1'...", 
            label)
    lbl.pretrained_model |> predict(x_cnn.test)    
  })

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)

  class(preds.mx)
  dim(preds.mx)
  
  colnames(preds.mx) <- as.character(y.labels)
  str(preds.mx)
  head(preds.mx)
# --------------------------  
#                 #            $            &            @            0            1            2            3
# [1,] 4.641171e-20 1.548882e-10 1.557235e-10 6.024950e-13 5.237010e-05 3.076157e-01 8.249410e-04 4.708885e-09
# [2,] 2.551753e-08 2.539963e-12 8.521654e-19 2.655758e-12 8.219911e-01 1.346034e-07 8.461294e-04 2.707075e-03
# [3,] 1.468435e-02 9.997920e-01 2.706961e-07 1.955358e-06 3.854533e-05 9.779275e-08 1.342094e-09 1.146960e-07
# [4,] 2.497534e-12 3.828882e-12 9.587651e-10 1.020366e-09 1.756670e-10 2.131516e-06 4.937184e-04 2.147722e-03
# [5,] 4.718320e-15 2.655487e-14 2.435475e-20 3.687264e-14 7.323680e-06 1.367646e-08 1.141905e-03 9.948028e-06
# [6,] 1.695516e-18 1.427704e-15 1.662214e-14 2.688509e-12 2.733296e-08 3.461286e-11 5.689621e-04 5.423213e-09
#                 4            5            6            7            8            9            A            B
# [1,] 9.430872e-05 1.554916e-04 1.024598e-01 1.113366e-10 1.715523e-04 3.078325e-08 0.0005775922 7.518265e-02
# [2,] 5.910299e-09 8.495885e-11 1.828520e-07 5.197902e-05 5.125588e-02 4.233158e-05 0.4537974894 9.761820e-01
# [3,] 2.866311e-03 1.564778e-08 4.831781e-06 2.522276e-06 1.620789e-05 6.913731e-07 0.0211899877 9.354964e-04
# [4,] 7.939283e-01 6.362331e-05 9.217733e-10 2.179653e-06 4.392010e-06 8.255338e-06 0.0002773236 9.637148e-05
# [5,] 6.183688e-11 2.417175e-10 1.293381e-11 9.999998e-01 8.656563e-03 3.998992e-03 0.0000156950 6.781822e-05
# [6,] 6.300766e-03 1.245115e-02 4.462381e-03 3.377349e-14 2.023772e-05 3.438555e-09 0.0023563516 1.587958e-01
#                 C            D            E            F          G            H            I            J
# [1,] 1.515672e-01 9.152557e-04 9.505384e-04 0.0001272728 0.00102315 1.256792e-02 9.183752e-01 1.508095e-02
# [2,] 3.107886e-07 7.114547e-01 1.059276e-05 0.0005078831 0.09680843 2.406922e-04 1.938796e-03 3.132850e-02
# [3,] 7.272462e-03 2.748886e-03 4.795025e-02 0.0284309424 0.04930085 8.568558e-05 4.165435e-04 8.908035e-03
# [4,] 3.017192e-04 2.984889e-03 6.788269e-07 0.3488173485 0.03528162 1.904747e-02 2.114991e-02 9.848953e-02
# [5,] 6.195524e-06 6.621844e-06 1.277975e-08 0.0003612466 0.05059635 1.671403e-06 2.578446e-04 7.945859e-05
# [6,] 8.429906e-02 1.103552e-05 2.585334e-01 0.0367332138 0.01111498 3.461441e-02 3.595663e-05 2.638722e-05
#                 K            L            M            N            P           Q            R            S
# [1,] 0.2105598599 9.980495e-01 7.018268e-08 1.162049e-06 7.599045e-04 0.001935191 5.529792e-03 1.136623e-02
# [2,] 0.0028187719 2.188025e-04 2.312052e-03 2.650113e-01 9.275764e-01 0.795970917 4.644250e-02 7.596704e-05
# [3,] 0.0021762187 1.655143e-04 2.518801e-04 4.863248e-03 8.451994e-03 0.151361778 4.604540e-03 1.289139e-04
# [4,] 0.0620683692 2.131083e-03 7.248587e-06 1.330861e-06 1.310114e-05 0.002629695 3.479059e-04 3.430053e-02
# [5,] 0.0000395591 2.966547e-05 9.786534e-08 4.992985e-07 2.557669e-03 0.114322193 9.231791e-05 1.025346e-07
# [6,] 0.9995679855 1.704450e-03 3.790588e-03 1.385469e-03 3.670332e-03 0.002063679 9.653007e-01 7.165396e-06
#                 T            U            V            W            X            Y           Z
# [1,] 0.0043690787 5.504408e-03 2.206398e-04 8.326172e-04 0.0074910680 0.0006272868 0.013624570
# [2,] 0.0007396784 8.388584e-03 7.606152e-04 1.002794e-04 0.0012490270 0.0049288166 0.993805408
# [3,] 0.0192838795 2.243590e-04 2.582811e-05 8.478419e-04 0.0001511377 0.0013147829 0.113081537
# [4,] 0.9999873638 2.970505e-07 4.609880e-03 7.111942e-05 0.0201902632 0.0326905102 0.000461001
# [5,] 0.0014585953 2.985363e-05 8.863709e-05 1.281543e-05 0.0037410462 0.0012273150 0.032307394
# [6,] 0.0001354754 4.628201e-05 2.487600e-03 2.403874e-03 0.3637407124 0.0177062247 0.018065078  
# -------------------  
  
  
  length(y_cnn.test)
  # 4641
  
  preds.combined <- apply(preds.mx, 1, function(r) {
    r.max <- max(r)
    c(label = y.labels[which.max(r)], 
      P = r.max,
      Predicted = cnn.binclass.get_prediction_values(r.max))
  }) |> t()
  
  str(preds.combined)
 # num [1:4641, 1:3] 26 39 2 33 12 25 28 5 15 34 ...
 # - attr(*, "dimnames")=List of 2
 #  ..$ : NULL
 #  ..$ : chr [1:3] "label" "P" "Predicted"

  dim(preds.combined)
  # [1] 4641    3
  head(preds.combined)
#      label         P Predicted
# [1,]    26 0.9980495         1
# [2,]    39 0.9938054         1
# [3,]     2 0.9997920         1
# [4,]    33 0.9999874         1
# [5,]    12 0.9999998         1
# [6,]    25 0.9995680         1

  sum(preds.combined[, "Predicted"])
  # [1] 4640  

  unrecognized.idx <- which(preds.combined[, "Predicted"] == 0)  
# 3961

  unrecognized <- preds.combined[unrecognized.idx,]
#     label          P  Predicted 
# 24.0000000  0.3319259  0.0000000 
  
  y.predicted_optimistic <- y.labels[unrecognized["label"]]
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
  
    
  y.unrecognized <- y_cnn.test[unrecognized.idx]  
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

  y_cnn.test[unrecognized.idx] == y.predicted_optimistic 
  # TRUE
  
  
  dim(x_cnn.test)
  
  x.unrecognized <- x_cnn.test[unrecognized.idx,,,1]  
  dim(x.unrecognized)
  
  char.image(x.unrecognized)
  
  
  
predictions <- preds.combined[, "label"]

head(predictions) 
# [1] 26 39  2 33 12 25

head(y_cnn.test)

head(predictions == as.integer(y_cnn.test))
#> [1]  FALSE  TRUE FALSE  TRUE  TRUE  TRUE

cnn.bin.accuracy <- mean(predictions == as.integer(y_cnn.test)) 
put_log("CNN-based Binary Classifier Models Ensemble accuracy: %1", cnn.bin.accuracy)

# 0.8547727 

head(predictions)

wrong.idx <- seq_len(length(y_cnn.test))[predictions != as.integer(y_cnn.test) ]
head(wrong.idx)
length(wrong.idx)
# [1] 674

 preds.wrong <- sapply(wrong.idx, function(i) {
   c(true_val = y_cnn.test[i], pred = predictioins[i], index = i)
 }) |> t()
 
 str(preds.wrong)
 dim(preds.wrong)
 
 head(preds.wrong)
 
 df.preds.wrong <- data.frame(true_val = y.labels[preds.wrong[,"true_val"]],
                              pred = y.labels[preds.wrong[,"pred"]],
                              index = preds.wrong[,"index"])
length(df.preds.wrong$true_val)
 
head(df.preds.wrong)
#    true_val pred index
# 1        B    Z     2
# 2        W    N     7
# 3        H    A     9
# 4        N    U    16
# 5        F    P    22
# 6        Z    2    28

# char.image(x_cnn.test[2,,,1]) 
# char.image(x_cnn.test[7,,,1]) 
# char.image(x_cnn.test[9,,,1]) 
# char.image(x_cnn.test[16,,,1]) 
# char.image(x_cnn.test[22,,,1]) 
# char.image(x_cnn.test[28,,,1]) 

}

put_end_date(start)
#### Close Log -----------------------------------------------------------------
log_close()
#----------------------------------
err.idx <- which(cnn.prediction.values != as.integer(y_cnn.test))
length(err.idx)
# 94951
err.head.idx <- head(err.idx)



err.pred.values <- cnn.prediction.values[err.idx]
head(err.pred.values)

err.teslbl.values <- y_cnn.test[err.idx]
head(err.teslbl.values)

err.head.img <- x_cnn.test[err.head.idx,,,1]
dim(err.head.img)

# par(mfrow = c(6, 1))
# for(i in err.head.idx) {
#   char.image(x_cnn.test[i,,,1])
# }
# par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfronlbl.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/lateslbl.html
