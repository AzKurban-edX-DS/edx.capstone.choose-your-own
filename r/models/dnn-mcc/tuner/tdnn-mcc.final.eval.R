#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Tuned DNN-Based MCC (TDNN MCC) Final Model Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

open_logfile(".tdnn-mcc.final.evaluation")

stopifnot(file.exists(tdnn_mcc.final.file,
                      ds28x28.split.train_0.8.backup.file),
          dir.exists(dnn_mcc.tuner.dir))

## Prepare a Test Set for the TDNN MCC Final Model Evaluation ------------------
start <- put_start_date()

put_log("Loading the Test Set of 28x28-size image data...")
test_set <- load.test_set(ds28x28.split.train_0.8.backup.file)

put_log("The Test Set of 28x28-size image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)


put_log("The Test Set object structure is as follows:
%1", capture.output(str(test_set)))

x_test <- test_set$x
# storage.mode(x_test) <- "integer"

# x_test <- x_test[seq(1e4),,]
str(x_test)
dim(x_test)

y.test.groups <- test_set$class_groups
rm(test_set)

stopifnot(sum(as.character(y.test.groups$classID) != rownames(x_test)) == 0)

y_test <- as.array(as.integer(y.test.groups$classID) - 1)
str(y_test)
dim(y_test)

stopifnot(min(y_test) == 0,
          max(y_test) == 38,
          dim(y_test) == nrow(x_test))

### Size of the Test Set by Class ------------------------------------------

put_log("The Test Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
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
  invisible(NULL)
}

rm(y.test.groups)
log_close()

## TDNN MCC Final Model Evaluation ---------------------------------------------

put_log("Loading pre-trained TDNN MCC Final Model...")

tdnn_mcc.final <- keras3::load_model(tdnn_mcc.final.file)

put_log("The TDNN MCC Final Model has been loaded from the backup file:
%1", tdnn_mcc.final.file)



put_log("Evaluating the TDNN MCC Final Model...")
tdnn_mcc.final.eval.result <- tdnn_mcc.final |> evaluate(x_test, y_test)
put_log("The TDNN MCC Final Model evaluation result:
%1", capture.output(str(tdnn_mcc.final.eval.result)))
# List of 2
# $ accuracy: num 0.89
# $ loss    : num 0.381
# $ accuracy        : num 0.873
# $ loss            : num 0.419

put_end_date(start)
# Time difference of 1.668308 mins

tdnn_mcc.final.preds <- tdnn_mcc.final |> predict(x_test)
str(tdnn_mcc.final.preds)
# put_end_date(start)
# Time difference of  mins

colnames(tdnn_mcc.final.preds) <- Y.Labels
head(tdnn_mcc.final.preds[,1:5])
#                 #            $            &            @            0
# [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
# [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
# [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
# [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
# [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
# [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
dim(tdnn_mcc.final.preds)
#> [1] 33228    39

tdnn_mcc.final.eval.result$predicted.probs <- tdnn_mcc.final.preds
str(tdnn_mcc.final.eval.result)

tdnn_mcc.final.preds.ts <- as_tensor(tdnn_mcc.final.preds)
str(tdnn_mcc.final.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

tdnn_mcc.final.predictions <- tdnn_mcc.final.preds.ts |> op_argmax(2)
tdnn_mcc.final.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(tdnn_mcc.final.predictions)
#> [1] 33228
# tdnn_mcc.final.predictions$numpy()


# y_test
# as.integer(y_test)

tdnn_mcc.final.pred.values.idx <- tdnn_mcc.final.predictions$numpy()
# head(tdnn_mcc.final.pred.values.idx)
# min(tdnn_mcc.final.pred.values.idx)
# max(tdnn_mcc.final.pred.values.idx)

tdnn_mcc.final.pred.values <- Y.Labels[tdnn_mcc.final.pred.values.idx]
head(tdnn_mcc.final.pred.values)

tdnn_mcc.final.eval.result$predicted.values <- tdnn_mcc.final.pred.values
str(tdnn_mcc.final.eval.result)


y_test.idx <- y_test + 1
y_test.labels <- Y.Labels[y_test.idx]

tdnn_mcc.final.eval.result$targets <- y_test.labels

rm(y_test.labels,
   y_test.idx,
   tdnn_mcc.final.pred.values,
   tdnn_mcc.final.pred.values.idx,
   tdnn_mcc.final.predictions,
   tdnn_mcc.final.preds.ts,
   tdnn_mcc.final.preds)

put_log("Saving the TDNN MCC Final Model Evaluation Result object...")
saveRDS(tdnn_mcc.final.eval.result,
        file = tdnn_mcc.final.eval_result.file)

put_log("The TDNN MCC Final Model Evaluation Result object 
has been saved in the following file:
  %1", tdnn_mcc.final.eval_result.file)


rm(x_test,
   y_test)

# dnn_mcc.accuracy <- mean(tdnn_mcc.final.pred.values.idx == y_test.idx)
put_log("The overall TDNN MCC Final Model accuracy: %1", 
        tdnn_mcc.final.eval.result$accuracy)
# 0.898338750451427
# 0.872517168521881
# 2HL: 0.906163454055786
# 5HL: 0.906584799289703
