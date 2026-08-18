#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DNN-Based Basic (DNNB) MCC Model Evaluation 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
open_logfile(".dnnb-mcc.model.evaluation")

stopifnot(file.exists(dnn_basic.model.file_path,
                      my_emnist.split.file_path))

## Preparing a Test Set ------------------------------------------------------ 
start <- put_start_date()

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.split.file_path)

put_log("The Split Flatten Dataset has the following structure:
%1", capture.output(str(ds)))

x_test <- ds$test$x

y_test <- ds$test$class_groups$classID

stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0)
stopifnot(nrow(x_test) == length(y_test))

### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/

y_test.cat <- to_categorical(y_test)
colnames(y_test.cat) <- Y.Labels
dim(y_test.cat)
str(y_test.cat)
head(y_test.cat)

### Size of the Test Set by Class --------------------------------------------
put_log("The Test set is balanced with respect to the set of classes:
%1", capture.output(print(ds$test$class_groups$groupByClass, n = N.classes)))
{
  # # A tibble: 39 × 2
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

rm(ds)

## DNNB MCC Model Evaluation ---------------------------------------------------

put_log("Loading pre-trained BDL MCC Model...")
dnn_basic.model <- keras3::load_model(dnn_basic.model.file_path)

put_log("The BDL MCC Model has been loaded from the backup file:
%1", dnn_basic.model.file_path)

if(file.exists(dnn_basic.model.train_history.file_path)){
  put_log("Loading the BDL MCC Model Train History...")
  
  dnn_basic.train_history <- readRDS(dnn_basic.model.train_history.file_path)
  
  put_log("The BDL MCC Model has been loaded from the backup file:
%1", dnn_basic.model.train_history.file_path)

  plot(dnn_basic.train_history) 
  rm(dnn_basic.train_history)
} else {
  warning("The BDL MCC Model backup does not exist:
", dnn_basic.model.train_history.file_path)
}

put_log("Evaluating DL Model...")
dnnb_mcc.eval.result <- dnn_basic.model |> evaluate(x_test, y_test.cat)
put_log("DL Model evaluation result:
%1", capture.output(str(dnnb_mcc.eval.result)))
# List of 2
#  $ accuracy: num 0.897
#  $ loss    : num 0.314

put_log("The overall Basic DL MCC Model evaluation accuracy: %1",
        dnnb_mcc.eval.result$accuracy)
# 0.898338750451427


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

rm(dnnb.preds.ts,
   dnnb.predictions,
   dnnb.pred.values.idx)

put_log("Saving the BDL MCC Model  Evaluation Result object...")
saveRDS(dnnb_mcc.eval.result,
        file = dnnb_mcc.eval.result.file)

put_log("The BDL MCC Model  Evaluation Result object has been trained 
and saved in the following file:
  %1", dnnb_mcc.eval.result.file)
put_end_date(start)

# dnn_basic.accuracy <- mean(dnnb.pred.values.idx == as.integer(y_test))
# put_log("The overall Basic `DL MCC` Model accuracy: %1",dnn_basic.accuracy)
# 0.898338750451427

rm(x_test,
   y_test,
   y_test.cat)

log_close()

