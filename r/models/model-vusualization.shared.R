#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared  Model Evaluation Results Visualization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

plot_img.dir <- file.path(data.dir, "plot.img")

if(!dir.exists(plot_img.dir))
  dir.create(plot_img.dir)

eval.conf.mx.obj_file <- file.path(data.dir,
                                   "eval.confusion-matrix.rds")
eval.conf.mx.img_file <- file.path(plot_img.dir,
                                   "eval.confusion-matrix.png")

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

put_log("Plotting ROC curves the Model Evaluation Results...")
eval.roc_curves <- plot.ROC.curves(y_test,
                                   k_best.nn_pca.probs)
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
                        export.img_file = eval.conf.mx.img_file,
                        backup.file = eval.conf.mx.obj_file)
rm(y_test)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(k_best.nn_pca.conf_mx.set)))

# eval.conf.mx.img <- magick::image_read(eval.conf.mx.img_file)
# plot(eval.conf.mx.img)

plot_image(eval.conf.mx.img_file)

# print_confusioin_matrix(k_best.nn_pca.conf_mx.set$cm.chart)

put_end_date(start)
log_close()



## Visualizing the Evaluation Results00 ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

conf.mx.img_file <- file.path(plot_img.dir,
                                           "eval.confusion-matrix.png")

eval.conf.mx.obj_file <- file.path(plot_img.dir,
                                           "eval.confusion-matrix.rds")

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

put_log("Plotting ROC curves the Model Evaluation Results...")
eval.roc_curves <- plot.ROC.curves(y.test.flatten,
                                            k_best.nn_pca.probs)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
k_best.nn_pca.acc_by_class <- plot.per_class.accuracy.bars(y.test.flatten,
                                                           k_best.nn_pca.predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(k_best.nn_pca.acc_by_class))
{
  invisible(NULL)
}

Sys.sleep(5)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

k_best.nn_pca.conf_mx.set <- 
  plot.confusion_matrix(y.test.flatten,
                        k_best.nn_pca.predicted,
                        # print.plot_object = T,
                        export.img_file = conf.mx.img_file,
                        backup.file = eval.conf.mx.obj_file)

# str(k_best.nn_pca.conf_mx.set)

# eval.conf.mx.img <- magick::image_read(conf.mx.img_file)
# plot(eval.conf.mx.img)

plot_image(conf.mx.img_file)

# print_confusioin_matrix(k_best.nn_pca.conf_mx.set$cm.chart)

put_end_date(start)
log_close()


## Evaluation Results: Visualization ------------------------------------------
start <- put_start_date()
put_log("Plotting ROC curves the Model Evaluation Results...")
CURRENT_MODEL.ROC.CURVES <- plot.ROC.curves(TEST_SET.TARGETS,
                                            EVAL.PREDICTED_PROBABILITIES)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
CURRENT_MODEL.ACCURACY_BY_CLASS <- plot.per_class.accuracy.bars(TEST_SET.TARGETS,
                                                                EVAL.PREDICTION_VALUES)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(CURRENT_MODEL.ACCURACY_BY_CLASS))
{

  invisible(NULL)
}

Sys.sleep(5)

# Confusion Matrix data suitable for Visualization using the `cvms` package:
# Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html

put_log("Creating a confusion matrix for Tuned BDL MCC Model in a format suitable for visualization 
using the `cvms` package...")

CURRENT_MODEL.CONFUSION_MATRIX <- create.confusion_matrix(TEST_SET.TARGETS,
                                                          EVAL.PREDICTION_VALUES)

put_log("The confusion matrix based on the `BDL MCC` Model evaluation results has been created:
%1", capture.output(CURRENT_MODEL.CONFUSION_MATRIX))

put_log("Plotting the confusion matrix, please wait...")
CONF.MX.CHART <- plot.confusion_matrix(CURRENT_MODEL.CONFUSION_MATRIX)

dev.off()
print(CONF.MX.CHART)

put_end_date(start)
log_close()
