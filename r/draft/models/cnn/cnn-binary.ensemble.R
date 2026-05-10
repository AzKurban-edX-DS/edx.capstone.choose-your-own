#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Ensemble Classifier for all labels 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##### Open Log for Ensemble Classifier -----------------------------------------
open_logfile(".cnn.ensemble.load-final-test-data")
## Load Final Test Data --------------------------------------------------------
if(!exists("x.final_test")) {
  stopifnot(file.exists(final_test.img28x28mx.array.file_path))
  x.final_test <- readRDS(final_test.img28x28mx.array.file_path)
  put_log("The Final Test Data has been loaded from the following backup file:
%1", final_test.img28x28mx.array.file_path)
}

put_log("The Final Test Data has the following structure:
%1", capture.output(str(x.final_test)))

y.final_test.groups <- ds.get_classIDs.grouped(x.final_test)
y.final_test <- y.final_test.groups$classID
str(y.final_test)
length(y.final_test)

y.final_test.int <- as.integer(y.final_test)
y.final_test.chars <- y.final_test.groups$groupByClass
str(y.final_test.chars)

print(y.final_test.chars, n = nrow(y.final_test.chars))
#--------
# A tibble: 39 × 2
#    classID     n
#    <fct>   <int>
#  1 #        1300
#  2 $        1350
#  3 &         520
#  4 @        1250
#  5 0         368
#  6 1        1675
#  7 2        1267
#  8 3        1301
#  9 4        1277
# 10 5        1388
# 11 6        1594
# 12 7         190
# 13 8        1198
# 14 9        1195
# 15 A         392
# 16 B         385
# 17 C         168
# 18 D         322
# 19 E         308
# 20 F         324
# 21 G         363
# 22 H         343
# 23 I         381
# 24 J         126
# 25 K         240
# 26 L         210
# 27 M         251
# 28 N         235
# 29 P         175
# 30 Q         405
# 31 R         366
# 32 S         168
# 33 T         384
# 34 U         210
# 35 V         224
# 36 W         182
# 37 X         119
# 38 Y         189
# 39 Z         181

# Create Balanced Test Set----------------------------------------------------- 

set.seed(length(y.labels)) # 39
sample_set <- sample_train_test_sets.x3d(x.final_test, test.ratio = 1)

x_ftest <- sample_set$test_set
rm(sample_set)

put_log("A Test Set created with the following structure:
%1", capture.output(str(x_ftest)))

y_ftest.groups <- ds.get_classIDs.grouped(x_ftest)
y_ftest <- y_ftest.groups$classID
str(y_ftest)
length(y_ftest)

y_ftest.int <- as.integer(y_ftest)
y_ftest.chars <- y_ftest.groups$groupByClass
str(y_ftest.chars)
# tibble [39 × 2] (S3: tbl_df/tbl/data.frame)
#  $ classID: Factor w/ 39 levels "#","$","&","@",..: 1 2 3 4 5 6 7 8 9 10 ...
#  $ n      : int [1:39] 4261 4261 4261 4261 4261 4261 4261 4261 4261 4261 ...

print(y_ftest.chars, n = nrow(y_ftest.chars))
#-----
#    classID     n
#    <fct>   <int>
#  1 #        4261
#  2 $        4261
#  3 &        4261
#  4 @        4261
#  5 0        4261
#  6 1        4261
#  7 2        4261
#  8 3        4261
#  9 4        4261
# 10 5        4261
# 11 6        4261
# 12 7        4261
# 13 8        4261
# 14 9        4261
# 15 A        4261
# 16 B        4261
# 17 C        4261
# 18 D        4261
# 19 E        4261
# 20 F        4261
# 21 G        4261
# 22 H        4261
# 23 I        4261
# 24 J        4261
# 25 K        4261
# 26 L        4261
# 27 M        4261
# 28 N        4261
# 29 P        4261
# 30 Q        4261
# 31 R        4261
# 32 S        4261
# 33 T        4261
# 34 U        4261
# 35 V        4261
# 36 W        4261
# 37 X        4261
# 38 Y        4261
# 39 Z        4261
# -----------------
max(y_ftest.chars$n) == min(y_ftest.chars$n)
#> [1] TRUE

max(y_ftest.chars$n)
#> [1] 4261

### Close Log ---------------------------------------------------------------
log_close()

##### Open Log for Ensemble Classifier -----------------------------------------
open_logfile(".cnn-model.ensemble-classifier")
##### Build Ensemble Classifier ------------------------------------------------

if (file.exists(cnn_models.ensemble.cache_file.path)) {
  put_log("CNN: loading the Ensemble Classifier Results from cache file: 
%1", cnn_models.ensemble.cache_file.path)
  load(cnn_models.ensemble.cache_file.path)
  put_log("CNN: the Ensemble Classifier Results have been loaded from the cache file:
%1", cnn_models.ensemble.cache_file.path)
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

    lbl.pretrained_model <- load_model(lbl.saved_model.path)  
    
    put_log("Summary of the model for handwritten character `%1`:
%2",label, capture.output(summary(lbl.pretrained_model)))
    
    put_log("Making predictions for handwritten character '%1'...", 
            label)
    lbl.pretrained_model |> predict(x_ftest)    
  })
  put_end_date()
  
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
  
  
  length(y_ftest)
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
# 3406

  unrecognized <- preds.combined[unrecognized.idx,]
#     label          P  Predicted 
# 24.0000000  0.3319259  0.0000000 
  
  y.predicted_optimistic <- y.labels[unrecognized["label"]]
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
  
    
  y.unrecognized <- y_ftest[unrecognized.idx]  
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

  y_ftest[unrecognized.idx] == y.predicted_optimistic 
  # TRUE
  
  
  dim(x_ftest)
  
  x.unrecognized <- x_ftest[unrecognized.idx,,]  
  dim(x.unrecognized)
  
  char.image(x.unrecognized)
  
  
  
  preds.combined <- apply(preds.mx, 1, function(r) {
    r.max <- max(r)
    c(label = as.character(y.labels)[which.max(r)], 
      P = r.max,
      Predicted = cnn.binclass.get_prediction_values(r.max))
  }) |> t()
  
  
  dim(preds.combined)
# [1] 4641    3
  
predictions <- preds.combined[, "label"]

head(predictions) 
# [1] 26 39  2 33 12 25

head(y_ftest)

head(predictions == as.integer(y_ftest))
#> [1]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE

accuracy <- mean(predictions == as.integer(y_ftest))  
# 0.8547727  

wrong.idx <- seq_len(length(y_ftest))[predictions != as.integer(y_ftest) ]
head(wrong.idx)
length(wrong.idx)
# [1] 674

 preds.wrong <- sapply(wrong.idx, function(i) {
   c(true_val = y_ftest[i], pred = predictioins[i], index = i)
 }) |> t()
 
 str(preds.wrong)
 dim(preds.wrong)
 
 head(preds.wrong)
 
 df.preds.wrong <- data.frame(true_val = y.labels[preds.wrong[,"true_val"]],
                              pred = y.labels[preds.wrong[,"pred"]],
                              index = preds.wrong[,"index"])
 head(df.preds.wrong)
#    true_val pred index
# 1        B    Z     2
# 2        W    N     7
# 3        H    A     9
# 4        N    U    16
# 5        F    P    22
# 6        Z    2    28

char.image(x_ftest[2,,]) 
char.image(x_ftest[7,,]) 
char.image(x_ftest[9,,]) 
char.image(x_ftest[16,,]) 
char.image(x_ftest[22,,]) 
char.image(x_ftest[28,,]) 

 
 
 
 
 
     
# ------------------------------  
  bin_preds.mx <- (preds.mx > 0.5) |> 
    as.integer() |> 
    matrix(nrow = nrow(preds.mx))
  
  colnames(bin_preds.mx) <- as.character(y.labels)
  
  class(bin_preds.mx)
  dim(bin_preds.mx)
  #> [1] 817379     39
  
  head(bin_preds.mx)
  #      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
  # [1,] 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # [2,] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # [3,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # [4,] 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # [6,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  
  
  df.model.acc_ordered <- df.models.acc |> arrange(accuracy)
  head(df.model.acc_ordered)
  # label  accuracy
  # T     T 0.9627761
  # 1     1 0.9708605
  # L     L 0.9711284
  # 5     5 0.9797157
  # I     I 0.9804546
  # U     U 0.9804766 
  
  lbl_T.idx <- df.model.acc_ordered$label == "T"
  lbl_T.acc <- df.model.acc_ordered[lbl_T.idx,]$accuracy
  lbl_T.acc
  # 0.9627761
  
  y.T_test <- (y_ftest == "T") |> as.integer()
  str(y.T_test)
  # int [1:817379] 0 0 0 0 0 0 0 0 0 0 ...
  
  bin.T_pred.mx <- bin_preds.mx[,"T"]
  str(bin.T_pred.mx)
  #  int [1:817379] 0 0 0 0 0 0 0 0 0 0 ..
  
  sum(bin.T_pred.mx)
  # 0
  
  sum(y.T_test)
  # Accuracy for binary classifier:
  sum(bin.T_pred.mx == y.T_test)
  # 786953
  
  mean(bin.T_pred.mx == y.T_test)
  #> [1] 0.9627761
  
  y.T_test.factor <- as.factor(y.T_test)
  str(y.T_test.factor)
  
  bin.T_pred.mx.factor <- factor(bin.T_pred.mx, levels = levels(y.T_test.factor))
  str(bin.T_pred.mx.factor)
  
  conf.mx <- confusionMatrix(y.T_test.factor, bin.T_pred.mx.factor)
  conf.mx
  # Confusion Matrix and Statistics
  # 
  #     Reference
  # Prediction      0      1
  # 0          786953      0
  # 1          30426      0
  # 
  # Accuracy : 0.9628          
  # 95% CI : (0.9624, 0.9632)
  # No Information Rate : 1               
  # P-Value [Acc > NIR] : 1               
  # 
  # Kappa : 0               
  # 
  # Mcnemar's Test P-Value : <2e-16          
  #                                         
  #           Sensitivity : 0.9628          
  #           Specificity :     NA          
  #        Pos Pred Value :     NA          
  #        Neg Pred Value :     NA          
  #            Prevalence : 1.0000          
  #        Detection Rate : 0.9628          
  #  Detection Prevalence : 0.9628          
  #     Balanced Accuracy :     NA          
  #                                         
  #      'Positive' Class : 0             
  
  lbl_1.idx <- df.model.acc_ordered$label == "1"
  lbl_1.acc <- df.model.acc_ordered[lbl_1.idx,]$accuracy
  lbl_1.acc
  # 0.9708605
  
  lbl_L.idx <- df.model.acc_ordered$label == "L"
  lbl_L.acc <- df.model.acc_ordered[lbl_L.idx,]$accuracy
  lbl_L.acc
  # 0.9711284
  
  names(evaluation.results) <- as.character(y.labels)
  
  avg.accuracy = mean(lbl_models.accuracies)
  avg.accuracy
  # 0.9888955
  
  head(preds.mx[,"T"])
  
  
  # rMaxs <- rowMaxs(preds.mx)
  # str(rMaxs)
  # head(rMaxs, 50)
  
  preds.optimistic <- apply(preds.mx, 1, function(r) {
    c(label = as.character(y.labels)[which.max(r)], 
      P = max(r))
  }) |> t()
  
  class(preds.optimistic)
  # "matrix" "array"
  dim(preds.optimistic)
  # 817379      2
  str(preds.optimistic)
  
  head(preds.optimistic)
  #       label P                   
  # [1,] "5"   "0.911750555038452" 
  # [2,] "6"   "0.989728391170502" 
  # [3,] "@"   "0.99999988079071"  
  # [4,] "&"   "0.999954521656036" 
  # [5,] "A"   "0.0807587429881096"
  # [6,] "@"   "0.999995708465576" 
  
  length(y_ftest)
  # 817379
  
  head(y_ftest)
  # [1] 5 6 @ & X @
  # 39 Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T ... Z
  
  acc.optimistic <- mean(y_ftest == preds.optimistic[,1]) 
  acc.optimistic
  #> 0.7836793
  
  y.labels.ext <- y.labels
  levels(y.labels.ext) <- c(levels(y.labels), "NA")
  y.labels.ext[40] <- "NA"
  y.labels.ext
  
  
  preds.norm <- apply(preds.mx, 1, function(r) {
    
    r.max = max(r)
    label <- as.character(y.labels)[which(r == r.max)]
    
    c(label = ifelse(r.max > 0.5, label, "NA"), 
      P = r.max)
  }) |> t()
  
  class(preds.norm)
  # "matrix" "array"
  dim(preds.norm)
  # 817379      2
  str(preds.norm)
  
  head(preds.norm)
  #        label P                   
  # [1,] "5"   "0.911750555038452" 
  # [2,] "6"   "0.989728391170502" 
  # [3,] "@"   "0.99999988079071"  
  # [4,] "&"   "0.999954521656036" 
  # [5,] "NA"  "0.0807587429881096"
  # [6,] "@"   "0.999995708465576" 
  
  acc.norm <- mean(y_ftest == preds.norm[,1]) 
  acc.norm
  # [1] 0.7429589
  
  
  
  preds.pessimistic <- apply(preds.mx, 1, function(r) {
    
    r.max = max(r)
    label <- as.character(y.labels)[which(r == r.max)]
    
    c(label = ifelse(r.max > 0.75, label, "NA"), 
      P = r.max)
  }) |> t()
  
  class(preds.pessimistic)
  # "matrix" "array"
  dim(preds.pessimistic)
  # 817379      2
  str(preds.pessimistic)
  
  head(preds.pessimistic)
  #        label P                   
  # [1,] "5"   "0.911750555038452" 
  # [2,] "6"   "0.989728391170502" 
  # [3,] "@"   "0.99999988079071"  
  # [4,] "&"   "0.999954521656036" 
  # [5,] "NA"  "0.0807587429881096"
  # [6,] "@"   "0.999995708465576" 
  
  acc.pessimistic <- mean(y_ftest == preds.pessimistic[,1]) 
  acc.pessimistic
  # [1] 0.6619133
  
  
  
  
  put_log("CNN: Caching the Ensemble Classifier Results...")
  save(cnn.ensemble,
       y_ftest,
       file = cnn_models.ensemble.cache_file.path)
  put_log("CNN: the Ensemble Classifier Results have been saved to the cache file:
%1", cnn_models.ensemble.cache_file.path)
  
  
}

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

err.head.img <- x_ftest[err.head.idx,,,1]
dim(err.head.img)

par(mfrow = c(6, 1))
for(i in err.head.idx) {
  char.image(x_ftest[i,,,1])
}
par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfronlbl.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/lateslbl.html
