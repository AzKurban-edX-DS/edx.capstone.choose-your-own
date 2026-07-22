#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared  Model Evaluation Results Visualization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Visualizing the Evaluation Results ------------------------------------------
stopifnot(exists("plots.args"),
          exists("model.eval.plots_dat.file"))

summary(plots.args)
class(plots.args)


start <- put_start_date()
while(!is.null(dev.list())) dev.off()
gc()

plots.dat <- structure(list(), class = "plotsDat")


### Plotting ROC Curves --------------------------------------------------------
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

put_log("Plotting ROC curves of the Multiclass Classifier Evaluation Results...")

plots.dat$ROC <- plot.ROC_curves(plots.args)
# plots.dat$ROC <- plot.ROC_curves(plots.dat)
class(plots.dat$ROC)
summary(plots.dat$ROC)

Sys.sleep(6)

put_log("Plotting the Model Predictions Per-Class Accuracy...")

plots.dat$PCA <- barPlot.accuracy_by_class(plots.args)
# plots.dat$PCA <- barPlot.accuracy_by_class(plots.dat)
class(plots.dat$PCA)
summary(plots.dat$PCA)

put_log("The following values of the Model Predictions Per-Class Accuracy have been plotted:
%1", capture.output(plots.dat$PCA))

Sys.sleep(6)

put_log("Plotting the confusion matrix based on the `BDL MCC` Model evaluation results, 
please wait...")

plots.dat$CM <- plot.confusion_matrix(plots.args)
# plots.dat$CM <- plot.confusion_matrix(plots.dat)
class(plots.dat$CM)
summary(plots.dat$CM)

if(!file.exists(model.eval.plots_dat.file)) {
  put_log("Saving the model-related plots input data object to file...")
  
  saveRDS(plots.dat,
          file = model.eval.plots_dat.file)

  put_log("The model-related plots input data object has been saved to the following file:
%1", model.eval.plots_dat.file)
}

put_end_date(start)

