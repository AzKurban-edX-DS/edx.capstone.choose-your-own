#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualizing the Tuned DNN-Based MCC (TDNN MCC) Final Model Evaluation Results
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Plotting the TDNN MCC Final Model Evaluation Results -----------------------------

open_logfile(".tdnn-fmcc.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path,
                      tdnn_mcc.final.eval_result.file))

if(!dir.exists(dnn_mcc.tuner.plots.dat.dir))
  dir.create(dnn_mcc.tuner.plots.dat.dir)

tdnn_mcc.final.eval.conf.mx.img_file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                                 "tdnn-mcc.final.eval.confusion-matrix.png")

tdnn_mcc.final.eval.plots_dat.file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                               "tdnn-mcc.final.eval.plots_dat.rds")

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
  tdnn_mcc.final.eval.result <- readRDS(tdnn_mcc.final.eval_result.file)
  
  put_log("The BDL MCC Final Model Evaluation Result object has been loaded 
from the following file:
%1", tdnn_mcc.final.eval_result.file)

  plots.args <- init.plots_args(targets = tdnn_mcc.final.eval.result$targets,
                                predicted.probabilities = tdnn_mcc.final.eval.result$predicted.probs,
                                predicted.values = tdnn_mcc.final.eval.result$predicted.values,
                                model_type = "Tuned MCC",
                                alg_name = "DNN",
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
#' class  accuracy
#'     # 0.9988263
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 1.0000000
#'     0 0.9260563
#'     1 0.8438967
#'     2 0.8896714
#'     3 0.9765258
#'     4 0.9330986
#'     5 0.8544601
#'     6 0.9084507
#'     7 0.9624413
#'     8 0.9025822
#'     9 0.9436620
#'     A 0.8638498
#'     B 0.9107981
#'     C 0.9295775
#'     D 0.9143192
#'     E 0.9237089
#'     F 0.9084507
#'     G 0.5974178
#'     H 0.9143192
#'     I 0.7312207
#'     J 0.9307512
#'     K 0.9237089
#'     L 0.2335681
#'     M 0.9683099
#'     N 0.9061033
#'     P 0.9600939
#'     Q 0.6314554
#'     R 0.9272300
#'     S 0.8697183
#'     T 0.9389671
#'     U 0.9507042
#'     V 0.9072770
#'     W 0.9612676
#'     X 0.9319249
#'     Y 0.8544601
#'     Z 0.9061033
  invisible(NULL)
}

rm(plots.dat)
log_close()











