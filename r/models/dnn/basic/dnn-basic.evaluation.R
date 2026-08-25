#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DNN-Based Basic (DNNB) MCC Model Evaluation 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
open_logfile(".dnnb-mcc.model.evaluation")

stopifnot(file.exists(dnn_basic.model.file_path))
start <- put_start_date()

## Prepare a Test Set for DNN-Based Basic MCC Model ----------------------------

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

## DNNB MCC Model Evaluation ---------------------------------------------------

put_log("Loading pre-trained DNN-Based Basic MCC Model...")
dnn_basic.model <- keras3::load_model(dnn_basic.model.file_path)

put_log("The DNN-Based Basic MCC Model has been loaded from the backup file:
%1", dnn_basic.model.file_path)

if(file.exists(dnn_basic.model.train_history.file_path)){
  put_log("Loading the DNN-Based Basic MCC Model Train History...")
  
  dnn_basic.train_history <- readRDS(dnn_basic.model.train_history.file_path)
  
  put_log("The DNN-Based Basic MCC Model has been loaded from the backup file:
%1", dnn_basic.model.train_history.file_path)

  plot(dnn_basic.train_history) 
  # rm(dnn_basic.train_history)
} else {
  warning("The DNN-Based Basic MCC Model History backup file does not exist:
", dnn_basic.model.train_history.file_path)
}

put_log("Evaluating DNN-Based Basic MCC Model...")
# dnnb_mcc.eval.result <- dnn_basic.model |> evaluate(x_test, y_test.cat)
dnnb_mcc.eval.result <- dnn_basic.model |> evaluate(x_test, y_test)
put_log("DNN-Based Basic MCC Model evaluation result:
%1", capture.output(str(dnnb_mcc.eval.result)))
# List of 2
# $ accuracy: num 0.898
# $ loss    : num 0.319

put_log("The overall DNN-Based Basic MCC Model evaluation accuracy: %1",
        dnnb_mcc.eval.result$accuracy)
# 0.897375702857971


put_end_date(start)
# Time difference of 1.668308 mins

dnnb_mcc.eval.result$predicted.probs <- dnn_basic.model |> predict(x_test) 
put_end_date(start)
# Time difference of  mins

colnames(dnnb_mcc.eval.result$predicted.probs) <- Y.Labels
head(dnnb_mcc.eval.result$predicted.probs[,1:5])
#                 #            $            &            @            0
# [1,] 1.291058e-08 2.551113e-09 1.649115e-10 3.021582e-07 1.282524e-06
# [2,] 1.855945e-11 2.930238e-11 1.776997e-09 3.246364e-06 1.664781e-08
# [3,] 2.074071e-11 1.434007e-10 3.564378e-09 8.688334e-12 2.087918e-05
# [4,] 1.167588e-09 1.428053e-10 3.141675e-11 4.189771e-10 1.695369e-07
# [5,] 9.645254e-13 3.239760e-12 2.698784e-14 9.448751e-12 2.331821e-09
# [6,] 2.562272e-10 2.353311e-13 3.094632e-13 2.608500e-11 4.482589e-08
dim(dnnb_mcc.eval.result$predicted.probs)
#> [1] 33228    39

dnnb.preds.ts <- as_tensor(dnnb_mcc.eval.result$predicted.probs)
str(dnnb.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

dnnb.predictions <- dnnb.preds.ts |> op_argmax(2)
dnnb.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
shape(dnnb.predictions)
#> [1] 33228
# dnnb.predictions$numpy()


dnnb.pred.values.idx <- dnnb.predictions$numpy()
head(dnnb.pred.values.idx)

dnnb_mcc.eval.result$predicted.values <- Y.Labels[dnnb.pred.values.idx]
head(dnnb_mcc.eval.result$predicted.values)

dnnb_mcc.eval.result$targets <- y_test

dnn_basic.accuracy <- mean(dnnb.pred.values.idx == as.integer(y_test))
put_log("The overall DNN-Based Basic MCC Model accuracy: %1",dnn_basic.accuracy)
# 0.898158179848321
# 0.898489236831665

rm(dnnb.preds.ts,
   dnnb.predictions,
   # dnn_basic.train_history,
   dnnb.pred.values.idx)

put_log("Saving the DNN-Based Basic MCC Model  Evaluation Result object...")
saveRDS(dnnb_mcc.eval.result,
        file = dnnb_mcc.eval.result.file)

put_log("The DNN-Based Basic MCC Model  Evaluation Result object has been trained 
and saved in the following file:
  %1", dnnb_mcc.eval.result.file)
put_end_date(start)

# rm(x_test,
#    y_test,
#    y_test.cat)

log_close()

