#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-Tuned CNN MCC Final Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".tuner.cnn-mcc.final-model.evaluation")
stopifnot(file.exists(tcnn_mcc.final_test.proc.path,
                      tcnn_mcc.final.file,
                      final_test.img28x28mx.array.file_path))

### Preparing a Test Set for the Best Model Final Test -------------------------
put_log("Loading the Test Binary Image 28x28 array set from the backup file...")
ftest_set <- readRDS(final_test.img28x28mx.array.file_path)
put_log("The Test Binary Image 28x28 array set has been loaded from the following file:
%1", final_test.img28x28mx.array.file_path)

x <- ftest_set$img28x28mx.array
x_test.files <- ftest_set$img28x28mx.fpath
rm(ftest_set)

x_test <- array_reshape(x, c(nrow(x), 28, 28, 1))
str(x_test)
shape(x_test)

y.test.groups <- ds.get_classIDs.grouped(x)
rm(x)

stopifnot(sum(as.character(y.test.groups$classID) != rownames(x_test)) == 0)

y_test <- as.array(as.integer(y.test.groups$classID) - 1)
str(y_test)
dim(y_test)

stopifnot(min(y_test) == 0,
          max(y_test) == 38,
          dim(y_test) == nrow(x_test))

y_test.cat <- to_categorical(y_test)
colnames(y_test.cat) <- Y.Labels

#### Size of the Final Test Set by Class ---------------------------------------
put_log("This imbalanced Test Set is prepared to use for the final testing of the best model:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  # 1 #        1300
  # 2 $        1350
  # 3 &         520
  # 4 @        1250
  # 5 0         368
  # 6 1        1675
  # 7 2        1267
  # 8 3        1301
  # 9 4        1277
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
  invisible(NULL)
}

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
tcnn_mcc.final.eval.result <- cnn_mcc.final |> evaluate(x_test, y_test.cat)
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

tcnn_mcc.final.eval.result$targets <- eval.targets

rm(cnn_preds.ts,
   cnn_mcc.final.predictions,
   cnn.prediction.values.idx,
   eval.targets)

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

metric <- metric_f1_score(average = 'macro', name = 'macro_f1', threshold = 0.5)

metric$update_state(y_test.cat, tcnn_mcc.final.eval.result$predicted.probs)
result <- metric$result()
result

f1_score <- result$numpy()
put_log("CNN-Based Multiclass Classifier Model F1 Score: %1", f1_score)



rm(x_test,
   y_test,
   y_test.cat)

log_close()







