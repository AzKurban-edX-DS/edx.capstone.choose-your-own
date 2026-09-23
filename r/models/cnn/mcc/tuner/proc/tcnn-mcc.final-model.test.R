#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-Tuned CNN MCC Final Model: Test Procedure
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".tuner.cnn-mcc.final-model.evaluation")
stopifnot(exists("x_test"),
          exists("x_test.files"),
          exists("y_test"),
          file.exists(tcnn_mcc.final.file))

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
tcnn_mcc.final.eval.result <- cnn_mcc.final |> evaluate(x_test, y_test)
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

rm(x_test,
   y_test,
   y_test.cat)

log_close()



## Review Some Errors --------------------------------------------------------- 

recg.err.info <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                        tcnn_mcc.final.eval.result$targets,
                                        x_test.files)
put_log("First 30 prediction errors:
%1", capture.output(head(recg.err.info, n = 30)))
{
  #    predicted actual                                         file
  # 1          9      G  data/raw/Vaibs.HW-Chars/Train/G/_1_2079.jpg
  # 2          L      1    data/raw/Vaibs.HW-Chars/Train/1/21673.jpg
  # 3          U      V      data/raw/Vaibs.HW-Chars/Train/V/610.jpg
  # 4          S      5    data/raw/Vaibs.HW-Chars/Train/5/29573.jpg
  # 5          S      5    data/raw/Vaibs.HW-Chars/Train/5/21880.jpg
  # 6          U      4    data/raw/Vaibs.HW-Chars/Train/4/31650.jpg
  # 7          1      L     data/raw/Vaibs.HW-Chars/Train/L/5851.jpg
  # 8          V      Y       data/raw/Vaibs.HW-Chars/Train/Y/83.jpg
  # 9          0      D     data/raw/Vaibs.HW-Chars/Train/D/2412.jpg
  # 10         Q      G  data/raw/Vaibs.HW-Chars/Train/G/_1_3172.jpg
  # 11         1      I     data/raw/Vaibs.HW-Chars/Train/I/9487.jpg
  # 12         9      Q  data/raw/Vaibs.HW-Chars/Train/Q/_1_1842.jpg
  # 13         C      E data/raw/Vaibs.HW-Chars/Train/E/_1_14130.jpg
  # 14         1      I    data/raw/Vaibs.HW-Chars/Train/I/12125.jpg
  # 15         L      I    data/raw/Vaibs.HW-Chars/Train/I/12325.jpg
  # 16         N      H      data/raw/Vaibs.HW-Chars/Train/H/682.jpg
  # 17         I      L    data/raw/Vaibs.HW-Chars/Train/L/11802.jpg
  # 18         Q      A data/raw/Vaibs.HW-Chars/Train/A/_1_10481.jpg
  # 19         1      L    data/raw/Vaibs.HW-Chars/Train/L/16686.jpg
  # 20         I      L     data/raw/Vaibs.HW-Chars/Train/L/6876.jpg
  # 21         I      L     data/raw/Vaibs.HW-Chars/Train/L/8059.jpg
  # 22         L      1    data/raw/Vaibs.HW-Chars/Train/1/34280.jpg
  # 23         B      6    data/raw/Vaibs.HW-Chars/Train/6/18502.jpg
  # 24         3      2    data/raw/Vaibs.HW-Chars/Train/2/35718.jpg
  # 25         8      G   data/raw/Vaibs.HW-Chars/Train/G/_1_272.jpg
  # 26         I      1    data/raw/Vaibs.HW-Chars/Train/1/24956.jpg
  # 27         R      T      data/raw/Vaibs.HW-Chars/Train/T/789.jpg
  # 28         G      5     data/raw/Vaibs.HW-Chars/Train/5/3938.jpg
  # 29         6      B  data/raw/Vaibs.HW-Chars/Train/B/_1_4978.jpg
  # 30         V      U    data/raw/Vaibs.HW-Chars/Train/U/15093.jpg
  invisible()
}

wrong_pred.I <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                       tcnn_mcc.final.eval.result$targets,
                                       x_test.files,
                                       pred.char = 'I')
# dev.off()
print.image_grid(wrong_pred.I)


wrong_pred.L <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                      tcnn_mcc.final.eval.result$targets,
                                      x_test.files,
                                      pred.char = 'L')
# dev.off()
print.image_grid(wrong_pred.L)


wrong_pred.1 <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                      tcnn_mcc.final.eval.result$targets,
                                      x_test.files,
                                      pred.char = '1')
# dev.off()
print.image_grid(wrong_pred.1)


# dev.off()
# str(recg.err.info)
#rm(recg.err.info)

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

# rm(tcnn_mcc.final.eval.result)

log_close()
