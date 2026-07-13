#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared  Model Evaluation Results Visualization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot.args <- list(ROC = list(targets = y_test,
                             pred_probs = fit_rf.mtry_best$test$votes),
                  acc_by_class = list(targets = y_test,
                                      pred_values = fit_rf.mtry_best$test$predicted),
                  cm = list(targets = y_test,
                            pred_values = fit_rf.mtry_best$test$predicted,
                            # print.plot = T,
                            export.img_file = rf_best.eval.conf.mx.img_file))

plot.args <- list(targets = y_test,
                  pred_probs = fit_rf.mtry_best$test$votes,
                  pred_values = fit_rf.mtry_best$test$predicted,
                  cm.export_img.file = rf_best.eval.conf.mx.img_file)

str(plot.args)
class(plot.args)

## Visualizing the Evaluation Results ------------------------------------------
stopifnot(exists("plot.args"),
          exists("model.eval.plots_dat.file"))

start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

plots.dat <- structure(list(), class = "plotsDat")


### Plotting ROC Curves --------------------------------------------------------
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

put_log("Plotting ROC curves of the Multiclass Classifier Evaluation Results...")

plots.dat$ROC <- plot_ROC(plot.args)
plots.dat$ROC <- plot_ROC(plots.dat)
class(plots.dat$ROC)

Sys.sleep(6)

put_log("Plotting the Model Predictions Per-Class Accuracy...")

plots.dat$PCA <- barPlot.accuracy_by_class(plot.args)
plots.dat$PCA <- barPlot.accuracy_by_class(plots.dat)
class(plots.dat$PCA)

put_log("The following values of the Model Predictions Per-Class Accuracy have been plotted:
%1", capture.output(plots.dat$PCA))

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


if(!file.exists(model.evaluation.plots.input_args.file)) {
  put_log("Saving the model-related plots input data object to file...")
  
  saveRDS(plots.input_arg.list,
          file = model.evaluation.plots.input_args.file)

  put_log("The model-related plots input data object has been saved to the following file:
%1", model.evaluation.plots.input_args.file)
}

put_end_date(start)

