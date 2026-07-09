#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared  Model Evaluation Results Visualization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Visualizing the Evaluation Results ------------------------------------------

open_logfile(".k(best)nn+pca.eval-results.visualization")

.eval.conf.mx.img_file <- file.path(.plot_img.dir,
                                    ".eval.confusion-matrix.png")

.eval.conf.mx.obj_file <- file.path(.plot_img.dir,
                                    ".eval.confusion-matrix.rds")

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

put_log("Plotting ROC curves the Model Evaluation Results...")
.eval.roc_curves <- plot.ROC.curves(y.test,
                                    .probs)
Sys.sleep(6)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
.acc_by_class <- plot.per_class.accuracy.bars(y.test,
                                              .predicted)

put_log("The following values of the BDL MCC Model Per-Class Accuracy have been plotted:
%1", capture.output(.acc_by_class))
{
  invisible(NULL)
}

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

.conf_mx.set <- 
  plot.confusion_matrix(y.test,
                        .predicted,
                        # print.plot_object = T,
                        export.img_file = .eval.conf.mx.img_file,
                        backup.file = .eval.conf.mx.obj_file)

put_log("Summary of the object containing computing results to plot the confusion matrix:
%1", capture.output(summary(k_best.nn_pca.conf_mx.set)))

# .eval.conf.mx.img <- magick::image_read(.eval.conf.mx.img_file)
# plot(.eval.conf.mx.img)

plot_image(.eval.conf.mx.img_file)

# print_confusioin_matrix(.conf_mx.set$cm.chart)

put_end_date(start)
log_close()

