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
                                   fit_rf.mmtry_best$test$votes)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
k_best.nn_pca.acc_by_class <- plot.per_class.accuracy.bars(y_test,
                                                           k_best.nn_pca.predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(k_best.nn_pca.acc_by_class))
{
  
  
  
  invisible(NULL)
}

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

k_best.nn_pca.conf_mx.set <- 
  plot.confusion_matrix(y_test,
                        k_best.nn_pca.predicted,
                        # print.plot_object = T,
                        export.img_file = rf_best.eval.conf.mx.img_file,
                        backup.file = rf_best.eval.conf.mx.obj_file)
rm(y_test)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(k_best.nn_pca.conf_mx.set)))

# rf_best.eval.conf.mx.img <- magick::image_read(rf_best.eval.conf.mx.img_file)
# plot(rf_best.eval.conf.mx.img)

plot_image(rf_best.eval.conf.mx.img_file)

# print_confusioin_matrix(k_best.nn_pca.conf_mx.set$cm.chart)

put_end_date(start)
log_close()


