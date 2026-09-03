#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RF MCC Model: Re-Train with the Best `mtry` Parameter Value
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Performing Evaluation --------------------------------------------------------

open_logfile(".rf-tuned.eval-results.visualization")

stopifnot(file.exists(fit_rf.final.backup.path,
                      model_visualization.shared.script.path))

put_log("Loading data of the fine-tuned `RF MCC` Model by the `mtry` parameter...")

fit_rf.final <- readRDS(fit_rf.final.backup.path)

put_log("The data of the fine-tuned `RF MCC` Model, 
trained with the best `mtry` parameter value, has been loaded from the following backup file:
%1", fit_rf.final.backup.path)

# put_log("The results of the fine-tuning `RF MCC` Model (after being trained with the best `mtry` parameter value
# on an 80% sample of the`Training Set` dataset and tested on the remaining 20% of the `Training Set`) 
# are as follows:
# %1", capture.output(fit_rf.final))
# put_end_date(start)
# # Time difference of 6.260901 hours
# 
# plot(fit_rf.final,
#      main = "Fine-tuning Results of the `RF MCC` Model by the `mtry` Parameter")
# 
# put_log("Prediction accuracy of the fine-tuned 'RF MCC' Model, 
# trained with the best `mtry` parameter value, is as follows:
# %1", fit_rf.final$test$accuracy)
# # 0.886029854339713
# 

#--------------------------------------------------
#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(rf_tuned.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(rf_tuned.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", rf_tuned.eval.plots_dat.file)
} else {
  plots.args <- init.plots_args(targets = fit_rf.final$test$targets,
                                predicted.probabilities = fit_rf.final$test$votes,
                                predicted.values = fit_rf.final$test$predicted,
                                model_type = "MCC",
                                alg_name = "RF",
                                pca.export_img.file_name = "rf-mcc.final.eval.pca.png",
                                pca.export_img.dir = data.models.rf.plots.dat.dir,
                                plots_dat.file = rf_tuned.eval.plots_dat.file,
                                cm.export.img_file = rf_tuned.eval.conf.mx.img_file,
                                cm.print.image = T)
}

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args,
   fit_rf.final)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

put_log("Saving the model-related plots input data object to file...")

saveRDS(plots.dat,
        file = rf_tuned.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", rf_tuned.eval.plots_dat.file)

rm(plots.dat)
log_close()


