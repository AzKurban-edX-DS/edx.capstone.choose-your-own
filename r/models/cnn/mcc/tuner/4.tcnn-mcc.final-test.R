#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-Tuned CNN MCC Final Model: Final Test
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".tuner.cnn-mcc.final-model.test")
stopifnot(file.exists(tcnn_mcc.final.file,
                      final_test.img28x28mx.array.file_path))

### Preparing a Test Set for the Best Model Final Test -------------------------
put_log("Loading the Test Binary Image 28x28 array set from the backup file...")
ftest_set <- readRDS(final_test.img28x28mx.array.file_path)
put_log("The Test Binary Image 28x28 array set has been loaded from the following file:
%1", final_test.img28x28mx.array.file_path)

ds.test <-shuffle.rows.x3d(ftest_set$img28x28mx.array,
                           ftest_set$img28x28mx.fpath)

x <- ds.test$x
x_test.files <- ds.test$x.files
rm(ftest_set, ds.test)

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
# [1] 0.9487213
# 
# $f1_macro
# [1] 0.9190166
# 
# $loss
# [1] 0.1859616

tcnn_mcc.final.eval.result$img_files <- x_test.files

put_log("CNN MCC Model: constructing predictions...")

tcnn_mcc.final.eval.result$predicted.probs <- cnn_mcc.final |> predict(x_test) 
put_log("CNN MCC Model: predictions have been constructed.")
put_end_date(start)

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

targets.idx <- y_test + 1
tcnn_mcc.final.eval.result$targets <- Y.Labels[targets.idx]

rm(cnn_preds.ts,
   cnn_mcc.final.predictions)

put_log("CNN MCC Model Final evaluation result structure:
%1", capture.output(str(tcnn_mcc.final.eval.result)))
# List of 6
# $ accuracy        : num 0.949
# $ f1_macro        : num 0.919
# $ loss            : num 0.186
# $ predicted.probs : num [1:22524, 1:39] 8.20e-13 1.77e-09 1.56e-19 1.05e-09 0.00 ...
# ..- attr(*, "dimnames")=List of 2
# .. ..$ : NULL
# .. ..$ : chr [1:39] "#" "$" "&" "@" ...
# $ predicted.values: Factor w/ 39 levels "#","$","&","@",..: 22 34 19 33 4 18 10 22 18 10 ...
# $ targets         : Factor w/ 39 levels "#","$","&","@",..: 22 34 19 33 4 18 10 22 18 10 ...

put_log("Saving the Multiclass Classifier model Evaluation Results...")
saveRDS(tcnn_mcc.final.eval.result,
        file = tcnn_mcc.final.eval_result.file)

put_log("The Evaluation Results data of the CNN-Based Multiclass Classifier Model 
have been backed up to the following file:
%1", tcnn_mcc.final.eval_result.file)

cnn_mcc.final.accuracy <- mean(cnn.prediction.values.idx == targets.idx)
put_log("CNN-Based Multiclass Classifier Model accuracy: %1", cnn_mcc.final.accuracy)
# 0.94872136387853
metric <- metric_f1_score(average = 'macro', name = 'macro_f1', threshold = 0.5)

metric$update_state(y_test.cat, tcnn_mcc.final.eval.result$predicted.probs)
result <- metric$result()
result
# tf.Tensor(0.9201387, shape=(), dtype=float32)

cnn_mcc.final.f1_macro <- result$numpy()
put_log("CNN-Based Multiclass Classifier Model, Macro F1 Score: %1", 
        cnn_mcc.final.f1_macro)
# 0.920138716697693


rm(x_test,
   y_test,
   y_test.cat)

log_close()
# =========================================================================
# Log End Time: 2026-09-26 09:17:44.095403
# Log Elapsed Time: 0 00:00:40
# =========================================================================







