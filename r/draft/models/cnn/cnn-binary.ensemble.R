#### Ensemble Classifier for all labels ----------------------------------------
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
  
  if(!exists("evaluation.results")) {
    stopifnot(!file.exists(cnn_models.eval.cache_file.path))
    
    put_log("CNN: Loading the Labeled Binary Classifier Models evalution results from cache...") 
    load(cnn_models.eval.cache_file.path)
    put_log("CNN: the Labeled Binary Classifier Models evalution results 
have been loaded from the cache file:
%1", cnn_models.eval.cache_file.path)
    
  }
  
  preds.mx <- sapply(y.labels, function(label) {
    #p <- 
    evaluation.results[[label]]$preds[,1]
    #str(p)
  })
  
  
  class(preds.mx)
  dim(preds.mx)
  
  colnames(preds.mx) <- as.character(y.labels)
  str(preds.mx)
  head(preds.mx)
  #               #            $            &            @            0            1
  # [1,] 1.369949e-12 2.396904e-11 4.014774e-10 3.622346e-09 1.100269e-06 3.533862e-05
  # [2,] 2.899649e-13 4.355252e-11 4.055812e-14 1.342694e-10 8.238061e-07 1.023313e-06
  # [3,] 1.046244e-07 6.048190e-11 2.396136e-08 9.999999e-01 3.843675e-13 2.751844e-09
  # [4,] 9.592652e-08 1.500224e-05 9.999545e-01 1.799757e-07 2.254854e-12 2.517688e-09
  # [5,] 6.245161e-10 4.554279e-12 1.529452e-06 4.037680e-11 2.591312e-10 1.986288e-03
  # [6,] 6.703245e-15 7.939808e-09 7.806473e-10 9.999957e-01 1.411460e-13 2.716332e-08
  #               2            3            4            5            6            7
  # [1,] 1.274050e-07 1.976297e-08 2.871119e-07 9.117506e-01 8.310264e-01 1.663653e-12
  # [2,] 1.036329e-04 5.842682e-08 7.924597e-06 2.290502e-03 9.897284e-01 2.419553e-16
  # [3,] 2.698742e-09 4.878437e-09 2.208877e-07 1.066727e-06 1.853692e-09 9.732599e-10
  # [4,] 2.360041e-07 3.376738e-05 4.101317e-08 6.470874e-04 5.833106e-08 2.433209e-13
  # [5,] 1.783007e-03 6.626777e-07 7.431519e-07 2.365470e-05 1.260650e-07 8.708229e-05
  # [6,] 3.107357e-10 2.704447e-08 7.753220e-05 3.603433e-06 3.583733e-05 1.783171e-09
  #               8            9            A            B            C            D
  # [1,] 5.372741e-05 1.092196e-07 5.041434e-05 2.950370e-04 4.660272e-07 1.776997e-05
  # [2,] 6.252207e-04 3.709382e-07 8.188316e-03 7.029655e-04 2.369297e-05 2.283386e-05
  # [3,] 7.717297e-07 1.682601e-07 5.947574e-06 2.228543e-05 2.070900e-07 2.059591e-08
  # [4,] 1.309585e-06 2.195027e-03 1.557932e-05 4.677896e-06 7.163868e-07 7.496845e-08
  # [5,] 4.989845e-03 3.979222e-07 8.075874e-02 5.097317e-04 9.486013e-08 3.500879e-03
  # [6,] 1.676496e-08 6.705029e-05 2.840243e-05 3.444430e-05 5.380749e-07 1.841350e-10
  #               E            F            G            H            I            J
  # [1,] 8.316667e-03 3.006915e-04 1.442758e-01 1.378821e-04 3.178597e-07 4.436253e-06
  # [2,] 1.033750e-02 8.659600e-03 1.065786e-01 3.527990e-05 9.302857e-08 2.817498e-10
  # [3,] 4.010222e-09 9.375635e-05 5.761859e-03 1.861690e-03 1.403905e-09 7.124655e-09
  # [4,] 1.788994e-08 1.207731e-06 1.175228e-03 5.415139e-06 4.145133e-10 2.589490e-09
  # [5,] 4.289760e-10 1.531458e-03 6.374015e-05 5.685989e-02 8.214690e-05 1.906513e-04
  # [6,] 1.403405e-08 3.572582e-05 5.061809e-03 6.416230e-03 4.529754e-08 2.157732e-12
  
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
  
  y.T_test <- (y_test == "T") |> as.integer()
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
  
  length(y_test)
  # 817379
  
  head(y_test)
  # [1] 5 6 @ & X @
  # 39 Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T ... Z
  
  acc.optimistic <- mean(y_test == preds.optimistic[,1]) 
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
  
  acc.norm <- mean(y_test == preds.norm[,1]) 
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
  
  acc.pessimistic <- mean(y_test == preds.pessimistic[,1]) 
  acc.pessimistic
  # [1] 0.6619133
  
  
  
  
  put_log("CNN: Caching the Ensemble Classifier Results...")
  save(cnn.ensemble,
       y_test,
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

err.head.img <- x_test[err.head.idx,,,1]
dim(err.head.img)

par(mfrow = c(6, 1))
for(i in err.head.idx) {
  char.image(x_test[i,,,1])
}
par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfronlbl.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/lateslbl.html
