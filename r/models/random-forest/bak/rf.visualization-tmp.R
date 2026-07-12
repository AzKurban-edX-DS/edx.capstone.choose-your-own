## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

data.models.plot_img.dir <- file.path(data.models.random_forest.dir, "plot.img")

if(!dir.exists(data.models.plot_img.dir))
  dir.create(data.models.plot_img.dir)

rf_best.eval.conf.mx.obj_file <- file.path(data.dir,
                                   "rf-final.eval.confusion-matrix.rds")
rf_best.eval.conf.mx.img_file <- file.path(data.models.plot_img.dir,
                                   "rf-final.eval.confusion-matrix.png")

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

### ROC Curves
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

put_log("Plotting ROC curves the Model Evaluation Results...")
eval.roc_curves <- plot.ROC.curves(y_test,
                                   fit_rf.mtry_best$test$votes)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")

fit_rf.mtry_best.accuracy.by_class <- 
  plot.per_class.accuracy.bars(y_test,
                               fit_rf.mtry_best$test$predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(fit_rf.mtry_best.accuracy.by_class))
{
  #' class  accuracy
  #'     # 1.0000000
  #'     $ 1.0000000
  #'     & 1.0000000
  #'     @ 1.0000000
  #'     0 0.9636150
  #'     1 0.7347418
  #'     2 0.8556338
  #'     3 0.9436620
  #'     4 0.9049296
  #'     5 0.8556338
  #'     6 0.9002347
  #'     7 0.9624413
  #'     8 0.8755869
  #'     9 0.9225352
  #'     A 0.8650235
  #'     B 0.8943662
  #'     C 0.9448357
  #'     D 0.8814554
  #'     E 0.9143192
  #'     F 0.9260563
  #'     G 0.6068075
  #'     H 0.9143192
  #'     I 0.6819249
  #'     J 0.9154930
  #'     K 0.9131455
  #'     L 0.5375587
  #'     M 0.9518779
  #'     N 0.9190141
  #'     P 0.9483568
  #'     Q 0.6572770
  #'     R 0.9072770
  #'     S 0.8685446
  #'     T 0.9154930
  #'     U 0.9260563
  #'     V 0.9166667
  #'     W 0.9647887
  #'     X 0.9260563
  #'     Y 0.8380282
  #'     Z 0.9014085
  invisible(NULL)
}

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

rf_best.eval.conf_mx.set <- 
  plot.confusion_matrix(y_test,
                        fit_rf.mtry_best$test$predicted,
                        # print.plot_object = T,
                        export.img_file = rf_best.eval.conf.mx.img_file,
                        backup.file = rf_best.eval.conf.mx.obj_file)
rm(y_test)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(rf_best.eval.conf_mx.set)))

# rf_best.eval.conf.mx.img <- magick::image_read(rf_best.eval.conf.mx.img_file)
# plot(rf_best.eval.conf.mx.img)

plot_image(rf_best.eval.conf.mx.img_file)

# print_confusioin_matrix(rf_best.eval.conf_mx.set$cm.chart)

put_end_date(start)
log_close()


