#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualizing the Tuned DNN-Based MCC (TDNN MCC) Final Model Evaluation Results
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Plotting the TDNN MCC Final Model Evaluation Results -----------------------------

open_logfile(".tdnn-fmcc.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path,
                      tdnn_mcc.final.eval.result.file))

if(!dir.exists(dnn_mcc.tuner.plots.dat.dir))
  dir.create(dnn_mcc.tuner.plots.dat.dir)

tdnn_mcc.final.eval.conf.mx.img_file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                                 "dnnb-final.eval.confusion-matrix.png")

tdnn_mcc.final.eval.plots_dat.file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                               "dnnb-final.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(tdnn_mcc.final.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(tdnn_mcc.final.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", tdnn_mcc.final.eval.plots_dat.file)
} else {
  put_log("Loading the BDL MCC Final Model Evaluation Result object...")
  tdnn_mcc.final.eval.result <- readRDS(tdnn_mcc.final.eval.result.file)
  
  put_log("The BDL MCC Final Model Evaluation Result object has been loaded 
from the following file:
%1", tdnn_mcc.final.eval.result.file)
  
  plots.args <- init.plots_args(targets = tdnn_mcc.final.eval.result$targets,
                                predicted.probabilities = tdnn_mcc.final.eval.result$predicted.probs,
                                predicted.values = tdnn_mcc.final.eval.result$predicted.values,
                                alg_name = "DL Basic Tuned",
                                plots_dat.file = tdnn_mcc.final.eval.plots_dat.file,
                                cm.export.img_file = tdnn_mcc.final.eval.conf.mx.img_file,
                                cm.print.image = T)
}

rm(tdnn_mcc.final.eval.result)

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args,
   fit_rf.mtry_best)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

put_log("Saving the model-related plots input data object to file...")

saveRDS(plots.dat,
        file = tdnn_mcc.final.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", tdnn_mcc.final.eval.plots_dat.file)

# put_log("The Basic DL Model per-class accuracy:,
# %1", capture.output(plots.dat$PCA$acc.by_class))
{
  
  
  
  
  invisible(NULL)
}

rm(plots.dat)
log_close()











