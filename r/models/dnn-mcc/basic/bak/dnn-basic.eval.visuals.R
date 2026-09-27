#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualizing the DNN-Based Basic (DNNB) MCC Model Evaluation Results
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Plotting the DNNB MCC Model Evaluation Results ------------------------------
open_logfile(".dl-basic.eval-results.visualization")

stopifnot(file.exists(model_visualization.shared.script.path,
                      dnnb_mcc.eval.result.file))

if(!dir.exists(dnn_mcc.basic.plots.dat.dir))
  dir.create(dnn_mcc.basic.plots.dat.dir)

dnnb_mcc.eval.conf.mx.img_file <- file.path(dnn_mcc.basic.plots.dat.dir,
                                             "dnn-basic.mcc.eval.confusion-matrix.png")

dnnb_mcc.eval.plots_dat.file <- file.path(dnn_mcc.basic.plots.dat.dir,
                                           "dnn-basic,mcc.eval.plots_dat.rds")

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
if(file.exists(dnnb_mcc.eval.plots_dat.file)) {
  put_log("Function `init.plots_args`:
Loading the model-related plots input data object from the backup file...")
  plots.args <- init.plots_args(dnnb_mcc.eval.plots_dat.file)
  
  put_log("Function `init.plots_args`:
The model-related plots input data object has been loaded from the following file:
%1", dnnb_mcc.eval.plots_dat.file)
} else {
  put_log("Loading the BDL MCC Model Evaluation Result object...")
  dnnb_mcc.eval.result <- readRDS(dnnb_mcc.eval.result.file)
  
  put_log("The BDL MCC Model Evaluation Result object has been loaded 
from the following file:
%1", dnnb_mcc.eval.result.file)
  
  plots.args <- init.plots_args(targets = dnnb_mcc.eval.result$targets,
                                predicted.probabilities = dnnb_mcc.eval.result$predicted.probs,
                                predicted.values = dnnb_mcc.eval.result$predicted.values,
                                model_type = "Basic MCC",
                                alg_name = "DNN",
                                plots_dat.file = dnnb_mcc.eval.plots_dat.file,
                                cm.export.img_file = dnnb_mcc.eval.conf.mx.img_file,
                                cm.print.image = T)
}

rm(dnnb_mcc.eval.result)

#'Run the helper script specifically designed to visualize 
#'the model evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

rm(plots.args)

stopifnot(exists("plots.dat"),
          !is.null(plots.dat$ROC),
          !is.null(plots.dat$PCA),
          !is.null(plots.dat$CM))

put_log("Saving the model-related plots input data object to file...")

saveRDS(plots.dat,
        file = dnnb_mcc.eval.plots_dat.file)

put_log("The model-related plots input data object has been saved to the following file:
%1", dnnb_mcc.eval.plots_dat.file)

## The DNNB MCC Model Per-Class Accuracy ---------------------------------------
put_log("The Basic DL Model per-class accuracy:,
%1", capture.output(plots.dat$PCA$acc.by_class))
#' class  accuracy
#'     # 1.0000000
#'     $ 1.0000000
#'     & 1.0000000
#'     @ 1.0000000
#'     0 0.9577465
#'     1 0.6502347
#'     2 0.8673709
#'     3 0.9577465
#'     4 0.9295775
#'     5 0.8767606
#'     6 0.9213615
#'     7 0.9776995
#'     8 0.9225352
#'     9 0.8356808
#'     A 0.8685446
#'     B 0.9025822
#'     C 0.9366197
#'     D 0.9295775
#'     E 0.9284038
#'     F 0.9354460
#'     G 0.6913146
#'     H 0.9225352
#'     I 0.7453052
#'     J 0.9166667
#'     K 0.9237089
#'     L 0.5258216
#'     M 0.9565728
#'     N 0.9284038
#'     P 0.9589202
#'     Q 0.7746479
#'     R 0.9107981
#'     S 0.8826291
#'     T 0.9342723
#'     U 0.9589202
#'     V 0.8990610
#'     W 0.9671362
#'     X 0.9377934
#'     Y 0.8767606
#'     Z 0.9260563

rm(plots.dat)
log_close()


