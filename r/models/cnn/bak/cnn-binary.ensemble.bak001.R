#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Ensemble Classifier for all labels 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Preparing Validation Set ----------------------------------------------------

open_logfile(".cnn.ensemble.load-final-test-data")
stopifnot(exists("ft.x3d.test_set"))
start <- put_start_date()

put_log("Evaluating the pre-trained CNN-based Binary Classifier Models...")

put_log("Preparing a Test Set...")

put_log("The Final Test Set data is stored in the object `ft.x3d.test_set`, 
having the following structure:
%1", capture.output(str(ft.x3d.test_set)))

ft.class.groups <- ds.get_classIDs.grouped(ft.x3d.test_set$x.test)

y.test <- ft.class.groups$classID
length(y.test)
#> [1] 4641

y.test.cat <- to_categorical(y.test)
colnames(y.test.cat) <- y.labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y.test.cat)))
#> [1] 4641   39

# str(y.test.cat)
head(y.test.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0

put_log("Reshaping the Test Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x.test <- array_reshape(ft.x3d.test_set$x.test, 
                        c(nrow(ft.x3d.test_set$x.test), 
                          n.img_rows, 
                          n.img_cols, 
                          1))

x.test.files <- ft.x3d.test_set$x.files
str(x.test.files)
 # chr [1:4641(1d)] "data/raw/Vaibs.HW-Chars/Validation/L/38.jpg" "data/raw/Vaibs.HW-Chars/Validation/B/58.jpg" ...
 # - attr(*, "dimnames")=List of 1
 #  ..$ : chr [1:4641] "L" "B" "$" "T" ...


put_log("The Test Set has been reshaped as follows:
%1", capture.output(shape(x.test)))
# shape(4641, 28, 28, 1)

### class Identifies: Quick Analysis -------------------------------------------

y.test.chars <- class.groups$groupByClass
#str(y.test.chars)

char_n.max <- max(y.test.chars$n)
# 853
char_n.max == min(y.test.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Test Set is as follows:
%1", capture.output(print(y.test.chars, n = nrow(y.test.chars))))
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
}

log_close()

## Build & Test Ensemble Classifier --------------------------------------------
open_logfile(".cnn-model.ensemble-classifier")
start = put_start_date()


cnn_binary.ensemble.backup.path <- file.path(data.cnn.binary.dir, 
                                                 "cnn.lbl-models.ensemble.RData")



if (file.exists(cnn_binary.ensemble.backup.path)) {
  put_log("CNN: loading the Ensemble Classifier Results from cache file: 
%1", cnn_binary.ensemble.backup.path)
  load(cnn_binary.ensemble.backup.path)
  put_log("CNN: the Ensemble Classifier Results have been loaded from the cache file:
%1", cnn_binary.ensemble.backup.path)
} else {
  stopifnot(file.exists(cnn_bin.models.dat.backup.path))
  cnn_bin.models.dat <- readRDS(cnn_bin.models.dat.backup.path)
  
  put_log("The pre-trained CNN-based Binary Classifier (CNN BCC) Models associated data 
has been loaded from the backup file:
%1", cnn_bin.models.dat.backup.path)
  
  put_log("The CNN BCC Models associated is loaded into `cnn_bin.models.dat` object 
with the following structure: 
object structure:
%1", str(cnn_bin.models.dat))
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  start <- put_start_date()
  
  preds.mx <- sapply(y.labels, function(label) {
    plot(cnn_bin.models.dat[[label]]$train_history)
    bin_model.file_path <- cnn_bin.models.dat[[label]]$saved_model.filepath 
    stopifnot(file.exists(bin_model.file_path))

    put_log("Loading the pre-trained CNN-based Binary Classifier Model for `%1` Class...",
            label)
    bin_model <- load_model(bin_model.file_path)  
    put_log("The pre-trained CNN-based Binary Classifier model for `%1` Class
has been loaded from the following backup file:
%2", label, bin_model.file_path)
    
    put_log("The CNN BCC for Class `%1` Model Info:
%2",label, capture.output(summary(bin_model)))
    
    put_log("Making predictions for handwritten character '%1'...", 
            label)
    bin_model |> predict(x.test)    
  })

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)

  class(preds.mx)
  dim(preds.mx)
  
  colnames(preds.mx) <- as.character(y.labels)
  str(preds.mx)
  head(preds.mx)
{  
#                 #            $            &            @            0            1            2            3
# [1,] 1.464786e-17 1.509567e-13 3.079715e-13 7.177939e-11 8.461436e-06 8.660759e-01 6.768929e-04 3.817237e-08 2.614076e-04
# [2,] 1.861775e-08 2.512369e-12 5.933821e-19 7.540813e-12 5.338433e-01 2.124531e-06 3.459474e-03 6.356398e-01 1.476072e-09
# [3,] 3.476948e-04 9.999931e-01 8.101281e-11 1.664573e-08 2.128664e-04 2.342620e-07 1.558131e-05 1.856697e-05 4.627344e-04
# [4,] 1.012581e-08 3.263334e-14 2.187669e-12 2.318405e-11 7.923223e-10 4.016808e-02 3.031172e-03 5.368839e-04 9.638922e-01
# [5,] 7.281785e-12 2.871485e-14 8.823690e-24 4.816790e-14 1.638039e-06 7.047000e-08 3.610858e-02 8.431097e-07 2.555120e-06
# [6,] 1.472084e-14 3.908268e-15 4.192804e-13 5.499078e-11 8.873970e-12 5.543083e-08 1.227341e-04 9.119218e-08 4.523068e-04
#                 5            6            7            8            9            A            B            C            D
# [1,] 7.667530e-04 8.516050e-02 1.220847e-08 5.964331e-04 1.052870e-07 5.821041e-06 6.804146e-02 2.301725e-02 2.077815e-01
# [2,] 4.051482e-10 3.456025e-09 4.778238e-05 1.721914e-02 2.636404e-04 6.299572e-01 9.616938e-01 3.580939e-05 6.597655e-01
# [3,] 4.767201e-09 1.363848e-08 6.327665e-03 1.251704e-05 1.347518e-04 3.333718e-02 4.299765e-03 4.235611e-03 6.294344e-04
# [4,] 1.065112e-04 1.100860e-08 1.359132e-03 1.337727e-05 1.019289e-05 6.992164e-03 2.410672e-04 2.364573e-05 1.799050e-01
# [5,] 7.401734e-08 1.912607e-09 9.999986e-01 1.448860e-02 1.119607e-04 1.101027e-05 1.565752e-05 5.116930e-05 1.227967e-04
# [6,] 8.735183e-04 9.706657e-03 4.201246e-11 4.967719e-05 1.585338e-12 3.131644e-05 9.580175e-03 3.936167e-02 2.127756e-07
#                 E            F           G            H            I            J            K            L            M
# [1,] 1.307511e-02 0.0003513385 0.013077533 1.147409e-02 3.406847e-01 0.0049531823 0.4088266790 9.947804e-01 1.174876e-06
# [2,] 6.166366e-05 0.0000626726 0.168325335 9.598793e-05 5.110650e-03 0.0363538340 0.0003895060 1.687802e-03 1.066457e-03
# [3,] 1.208192e-01 0.0090885321 0.019006861 4.308836e-05 3.288939e-04 0.0028309680 0.0006415974 3.018585e-04 2.306932e-04
# [4,] 2.865517e-04 0.1636269987 0.002309387 3.078379e-01 5.850167e-04 0.1629280746 0.2812836170 1.651696e-03 4.009757e-05
# [5,] 1.229562e-05 0.0012818197 0.021010727 2.956275e-07 1.916056e-05 0.0014203266 0.0000656433 9.483139e-06 1.842921e-06
# [6,] 1.523830e-02 0.0115720220 0.039459635 1.325382e-03 5.192264e-05 0.0002774831 0.9999530315 4.883196e-04 9.172699e-05
#                 N            P           Q            R            S            T            U            V            W
# [1,] 1.152227e-09 0.0005821452 0.009532232 4.359295e-03 3.876949e-04 1.465944e-02 1.288445e-04 1.572705e-04 1.307910e-04
# [2,] 1.183885e-01 0.9439183474 0.515962243 9.137842e-03 2.635836e-06 1.110963e-04 4.627729e-06 1.718758e-02 1.661750e-04
# [3,] 6.079099e-04 0.0074645216 0.015756629 3.568921e-03 5.796201e-05 5.221503e-03 2.755409e-04 7.332277e-05 1.150830e-04
# [4,] 4.591564e-05 0.0001062945 0.003216897 1.411438e-04 3.680416e-03 9.999559e-01 1.280700e-06 1.061506e-03 5.434246e-04
# [5,] 9.855704e-10 0.0366729461 0.286233276 2.999031e-05 1.537278e-08 5.329107e-05 3.086797e-06 1.811162e-03 4.815557e-06
# [6,] 5.774745e-05 0.0004875380 0.007987122 7.183143e-03 2.450984e-05 7.260052e-03 2.695624e-05 8.324823e-06 1.871469e-02
#                 X            Y            Z
# [1,] 0.0527844951 0.0027798195 0.0827367827
# [2,] 0.0004318741 0.0024063005 0.9339944124
# [3,] 0.0028044707 0.0004571930 0.0070139300
# [4,] 0.0055124150 0.0789095163 0.0003901580
# [5,] 0.0057774796 0.0004329347 0.0945555866
# [6,] 0.0116087431 0.0144109828 0.0007007018
}  

  length(y.test)
  # 4641
  
  ensemble.preds <- apply(preds.mx, 1, function(r) {
    r.max <- max(r)
    c(class.id = y.labels[which.max(r)], 
      P = r.max
      #Predicted = cnn.binclass.get_prediction_values(r.max)
      )
  }) |> t()
  
  str(ensemble.preds)
 # num [1:4641, 1:3] 26 39 2 33 12 25 28 5 15 34 ...
 # - attr(*, "dimnames")=List of 2
 #  ..$ : NULL
 #  ..$ : chr [1:3] "label" "P" "Predicted"

  dim(ensemble.preds)
  # [1] 4641    3
  head(ensemble.preds)
#      class.id         P
# [1,]       26 0.9947804
# [2,]       16 0.9616938
# [3,]        2 0.9999931
# [4,]       33 0.9999559
# [5,]       12 0.9999986
# [6,]       25 0.9999530
>
  # sum(ensemble.preds[, "Predicted"])
  # # [1] 4637  
  # 
  # unrecognized.idx <- which(ensemble.preds[, "Predicted"] == 0) 
  # unrecognized <- ensemble.preds[unrecognized.idx,]
  # unrecognized
#      class.id         P Predicted
# [1,]       13 0.3356235         0
# [2,]       16 0.3841972         0
# [3,]       36 0.3208078         0
# [4,]       24 0.4083948         0
#     label          P  Predicted 
# 24.0000000  0.3319259  0.0000000 
  
  y.predicted_optimistic <- y.labels[unrecognized["label"]]
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
  
    
  y.unrecognized <- y.test[unrecognized.idx]  
# [1] J
# Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

  y.test[unrecognized.idx] == y.predicted_optimistic 
  # TRUE
  
  
  dim(x.test)
  
  x.unrecognized <- x.test[unrecognized.idx,,,1]  
  dim(x.unrecognized)
  
  char.image(x.unrecognized)
  
  
  
predictions <- ensemble.preds[, "label"]

head(predictions) 
# [1] 26 39  2 33 12 25

head(y.test)

head(predictions == as.integer(y.test))
#> [1]  FALSE  TRUE FALSE  TRUE  TRUE  TRUE

cnn.bin.accuracy <- mean(predictions == as.integer(y.test)) 
put_log("CNN-based Binary Classifier Models Ensemble accuracy: %1", cnn.bin.accuracy)

# 0.8547727 

head(predictions)

wrong.idx <- seq_len(length(y.test))[predictions != as.integer(y.test) ]
head(wrong.idx)
length(wrong.idx)
# [1] 674

 preds.wrong <- sapply(wrong.idx, function(i) {
   c(true_val = y.test[i], pred = predictioins[i], index = i)
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

# char.image(x.test[2,,,1]) 
# char.image(x.test[7,,,1]) 
# char.image(x.test[9,,,1]) 
# char.image(x.test[16,,,1]) 
# char.image(x.test[22,,,1]) 
# char.image(x.test[28,,,1]) 

}

put_end_date(start)
#### Close Log -----------------------------------------------------------------
log_close()
#----------------------------------
err.idx <- which(cnn.prediction.values != as.integer(y.test))
length(err.idx)
# 94951
err.head.idx <- head(err.idx)



err.pred.values <- cnn.prediction.values[err.idx]
head(err.pred.values)

err.teslbl.values <- y.test[err.idx]
head(err.teslbl.values)

err.head.img <- x.test[err.head.idx,,,1]
dim(err.head.img)

# par(mfrow = c(6, 1))
# for(i in err.head.idx) {
#   char.image(x.test[i,,,1])
# }
# par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfronlbl.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/lateslbl.html
