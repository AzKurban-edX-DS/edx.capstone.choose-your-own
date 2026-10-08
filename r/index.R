#%%%%%%%%%%%%%%%%%%%%
# Main (Index) Script
#%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------

r_scripts.dir <- "r"
stopifnot(dir.exists(r_scripts.dir))

support_scripts.dir <-  file.path(r_scripts.dir, "support-scripts")
stopifnot(dir.exists(support_scripts.dir))

support_functions.dir <- file.path(r_scripts.dir, "support-functions")
stopifnot(dir.exists(support_functions.dir))

setup_script.file_path <- file.path(support_scripts.dir, "setup.R")
stopifnot(file.exists(setup_script.file_path))

source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Download the Kaggle Dataset ------------------------------------------------

stopifnot(file.exists(ds.kaggle.download.script.path))

source(ds.kaggle.download.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Prepare Input Datasets -----------------------------------------------------
stopifnot(file.exists(ds.train2trimmed_img.dat.script.path,
                      prepare_ds.script.path))

#### Train Set: Create Trimmed Image Data List ---------------------------------

source(ds.train2trimmed_img.dat.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


#### Prepare Input Datasets ----------------------------------------------------

source(prepare_ds.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

#### Prepare Flatten Datasets --------------------------------------------------
ds.prepare_flattened.script.path <- file.path(support_scripts.dir, 
                                         "prepare-flattened-datasets.R")

ds.load_flattened.script.path <- file.path(support_scripts.dir, 
                                         "load-flattened-dataset.R")

stopifnot(file.exists(ds.prepare_flattened.script.path,
                      ds.load_flattened.script.path))

source(ds.prepare_flattened.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## kNN+PCA MCC Model -----------------------------------------------------------
stopifnot(file.exists(knn_pca.tune.script.path,
                      knn_pca.retrain.best_k.script.path,
                      knn_pca.best.eval.script.path))
#### Init Paths ----------------------------------------------------------------

k1_8nn_pca.model.backup.path <-
  file.path(knn_pca.data.dir, "k1-8nn+pca(0.1train-set).rds")

k_best.nn_pca.model.backup.path <-
  file.path(knn_pca.data.dir, "k_best.nn+pca.rds")

knn_pca.eval.results.backup <-
  file.path(knn_pca.data.dir, "knn+pca.eval-results.rds")


knn_pca.eval.conf.mx.img_file <- file.path(knn_pca.data.plots.dat.dir,
                                           "knn+pca-tuned.eval.confusion-matrix.png")

knn_pca.eval.plots_dat.file <- file.path(knn_pca.data.plots.dat.dir,
                                         "knn+pca-tuned.eval.plots_dat.rds")

if(!dir.exists(knn_pca.data.plots.dat.dir))
  dir.create(knn_pca.data.plots.dat.dir)

#### Run Scripts ---------------------------------------------------------------

if(!file.exists(knn_pca.eval.results.backup)) {
  if(!file.exists(k_best.nn_pca.model.backup.path)) {
    if(!file.exists(k1_8nn_pca.model.backup.path)) {
      # Build & Tune the kNN+PCA MCC Model
      source(knn_pca.tune.script.path, 
             catch.aborts = TRUE,
             echo = TRUE,
             spaced = TRUE,
             verbose = TRUE,
             keep.source = TRUE)
    }
    
    # Re-Train kNN+PCA MCC Model with the Best `k` Value
    source(knn_pca.retrain.best_k.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }
  
  # Evaluate the best kNN+PCA MCC Model
  source(knn_pca.best.eval.script.path, 
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### The Best Model Evaluation Results Visualization ---------------------------

open_logfile(".visual.eval-results.k(best)nn+pca")

put_log("Loading Predicted Data of the Fine-Tuned kNN+PCA Model...") 

knn_pca.eval.results <- readRDS(knn_pca.eval.results.backup)
put_end_date(start)
# Time difference of 

put_log("The Predicted Data of the Fine-Tuned kNN+PCA Model has been loaded from the following file:
%1...", knn_pca.eval.results.backup)

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = knn_pca.eval.results$targets,
                              predicted.probabilities = knn_pca.eval.results$predicted.probs,
                              predicted.values = knn_pca.eval.results$predicted,
                              model_type = "MCC",
                              alg_name = "kNN+PCA",
                              pca.export_img.file_name = "knn+pca-mcc.best.eval.pca.png",
                              pca.export_img.dir = knn_pca.data.plots.dat.dir,
                              plots_dat.file = knn_pca.eval.plots_dat.file,
                              cm.export.img_file = knn_pca.eval.conf.mx.img_file,
                              cm.print.image = T)

put_log("The `plots.args` object of class `%1` has been created for use to generate 
a visual representation of the DNNB MCC model evaluation results.",
        class(plots.args))

# rm(knn_pca.eval.results)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()

## Random Forest (RF) MCC Model ------------------------------------------------

stopifnot(file.exists(rf_tuning.script.path,
                      rf_retraining.best_par.script.path))

### Init Paths ----------------------------------------------------------------

data.models.rf.tuning.dir <- file.path(data.models.rf.dir, "tuning")

fit_rf.fine_tuned.backup.path <- file.path(data.models.rf.tuning.dir, 
                                           "fit_rf.fine-tuned.ntree200.back.rds")

fit_rf.final.backup.path <- file.path(data.models.rf.dir, 
                                      "fit_rf.final.ntree400.back.rds")

rf_tuned.eval.conf.mx.img_file <- file.path(data.models.rf.plots.dat.dir,
                                            "rf-tuned.eval.confusion-matrix.png")

rf_tuned.eval.plots_dat.file <- file.path(data.models.rf.plots.dat.dir,
                                          "rf-tuned.eval.plots_dat.rds")

if(!dir.exists(data.models.rf.tuning.dir))
  dir.create(data.models.rf.tuning.dir)

if(!dir.exists(data.models.rf.plots.dat.dir))
  dir.create(data.models.rf.plots.dat.dir)

### Run Scripts ----------------------------------------------------------------

if(!file.exists(fit_rf.final.backup.path)) {
  if(!file.exists(fit_rf.fine_tuned.backup.path)) {
    # Build & Tune the RF MCC Model
    source(rf_tuning.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }
  
  # Re-Train RF MCC Model with the Best `k` Value
  source(rf_retraining.best_par.script.path, 
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### The Best Model Evaluation Results Visualization ---------------------------

open_logfile(".visual.eval-results.rf-final")

put_log("Loading data of the fine-tuned `RF MCC` Model by the `mtry` parameter...")
fit_rf.final <- readRDS(fit_rf.final.backup.path)

put_log("The data of the fine-tuned `RF MCC` Model, 
trained with the best `mtry` parameter value, has been loaded from the following backup file:
%1", fit_rf.final.backup.path)

put_log("The results of the fine-tuning `RF MCC` Model (after being trained with the best `mtry` parameter value
on an 80% sample of the`Training Set` dataset and tested on the remaining 20% of the `Training Set`) 
are as follows:
%1", capture.output(fit_rf.final))
put_end_date(start)
# Time difference of 6.260901 hours

plot(fit_rf.final,
     main = "Fine-tuning Results of the `RF MCC` Model by the `mtry` Parameter")

put_log("Prediction accuracy of the fine-tuned 'RF MCC' Model, 
trained with the best `mtry` parameter value, is as follows:
%1", fit_rf.final$test$accuracy)
# 0.886029854339713

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = fit_rf.final$test$targets,
                              predicted.probabilities = fit_rf.final$test$votes,
                              predicted.values = fit_rf.final$test$predicted,
                              model_type = "MCC",
                              alg_name = "Random Forest",
                              pca.export_img.file_name = "rf-mcc.final.eval.pca.png",
                              pca.export_img.dir = data.models.rf.plots.dat.dir,
                              plots_dat.file = rf_tuned.eval.plots_dat.file,
                              cm.export.img_file = rf_tuned.eval.conf.mx.img_file,
                              cm.print.image = T)

put_log("The `plots.args` object of class `%1` has been created for use to generate 
a visual representation of the DNNB MCC model evaluation results.",
        class(plots.args))

# rm(fit_rf.final)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()

## DNN-Based MCC Model ---------------------------------------------------------
### DNN-Based Basic MCC Model --------------------------------------------------
stopifnot(file.exists(dnnb_mcc.script.path,
                      dnnb_mcc.eval.script.path,
                      my_emnist.split.file_path))
#### Init Paths ----------------------------------------------------------------
dnnb_mcc.file <- file.path(data.dnn_mcc.basic.dir, 
                                       "dnnb_mcc.pre-trained.keras")

dnnb_mcc.train_history.file <- file.path(data.dnn_mcc.basic.dir, 
                                                     "dnnb_mcc.train_history.rds")

dnnb_mcc.eval.result.file <- file.path(data.dnn_mcc.basic.dir,
                                       "dnnb_mcc.eval.result.rds")

dnnb_mcc.plot_img.file <- file.path(dnnb_mcc.plots.dat.dir,
                                    "dnnb_mcc.model.png")


dnnb_mcc.eval.cm_img.file <- file.path(dnnb_mcc.plots.dat.dir,
                                            "dnnb_mcc.eval.cm.png")

dnnb_mcc.eval.plots_dat.file <- file.path(dnnb_mcc.plots.dat.dir,
                                          "dnnb_mcc.eval.plots_dat.rds")

if(!dir.exists(data.dnn_mcc.basic.dir))
  dir.create(data.dnn_mcc.basic.dir)

if(!dir.exists(dnnb_mcc.plots.dat.dir))
  dir.create(dnnb_mcc.plots.dat.dir)

#### Run Scripts ---------------------------------------------------------------
if(!file.exists(dnnb_mcc.eval.result.file)) {
  if(!file.exists(dnnb_mcc.file)) {
    source(dnnb_mcc.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }
  
  source(dnnb_mcc.eval.script.path,
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### Model Evaluation Results Visualization ------------------------------------

open_logfile(".dnnb-mcc.visual.eval-results")
stopifnot(file.exists(dnnb_mcc.file,
                      dnnb_mcc.eval.result.file,
                      model_visualization.shared.script.path),
          exists("dnnb_mcc.plot_img.file"))

put_log("Loading pre-trained DNN-Based Basic MCC Model...")
dnnb_mcc <- keras3::load_model(dnnb_mcc.file)

put_log("The DNN-Based Basic MCC Model has been loaded from the backup file:
%1", dnnb_mcc.file)

dnnb_mcc |> plot_keras_model(to_file = dnnb_mcc.plot_img.file,
                             show_shapes = T)
# rm(dnnb_mcc)

if(file.exists(dnnb_mcc.train_history.file)){
  put_log("Loading the DNN-Based Basic MCC Model Train History...")
  
  dnnb_mcc.train_history <- readRDS(dnnb_mcc.train_history.file)
  
  put_log("The DNN-Based Basic MCC Model has been loaded from the backup file:
%1", dnnb_mcc.train_history.file)
  
  plot(dnnb_mcc.train_history) 
  
  best_metrics <- model.train_history.get_best_metrics(dnnb_mcc.train_history)
  put_log("The best values of the DNN-Based Basic MCC Model training result are as follows:
%1", capture.output(best_metrics))
#  accuracy         loss val_accuracy     val_loss 
# 0.9301116    0.2064773    0.9013734    0.2976910  

  # rm(dnnb_mcc.train_history)
} else {
  warning("The DNN-Based Basic MCC Model History backup file does not exist:
", dnnb_mcc.train_history.file)
}

put_log("Loading the DNNB MCC Model Evaluation Result object...")
dnnb_mcc.eval.result <- readRDS(dnnb_mcc.eval.result.file)

put_log("The DNNB MCC Model Evaluation Result object has been loaded 
from the following file:
%1", dnnb_mcc.eval.result.file)

put_log("The DNNB MCC Model Evaluation Result accuracy: %1", 
        dnnb_mcc.eval.result$accuracy)
# 0.8973456

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = dnnb_mcc.eval.result$targets,
                              predicted.probabilities = dnnb_mcc.eval.result$predicted.probs,
                              predicted.values = dnnb_mcc.eval.result$predicted.values,
                              plots_dat.file = dnnb_mcc.eval.plots_dat.file,
                              model_type = "Basic MCC",
                              alg_name = "DNN",
                              pca.export_img.file_name = "dnnb-mcc.eval.pca.png",
                              pca.export_img.dir = dnnb_mcc.plots.dat.dir,
                              cm.export.img_file = dnnb_mcc.eval.cm_img.file,
                              cm.print.image = T)

put_log("The `plots.args` object of class `%1` has been created for use to generate 
a visual representation of the DNNB MCC model evaluation results.",
        class(plots.args))

rm(dnnb_mcc.eval.result)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()

### DNN-Based MCC Model Tuning -------------------------------------------------

stopifnot(file.exists(dnn_mcc.tuner.script.path,
                      tdnn_mcc.final.retrain.script.path,
                      tdnn_mcc.final.eval.script.path))

#### Init Paths ----------------------------------------------------------------
tdnn_mcc.best_hp.config.file <- file.path(dnn_mcc.tuner.dir,
                                               "tdnn-mcc.best-hp.config.rds")

tdnn_mcc.final.file <- file.path(dnn_mcc.tuner.dir, 
                                     "tdnn-mcc.final-model.keras")

tdnn_mcc.final.train_history.file <- file.path(dnn_mcc.tuner.dir, 
                                                      "tdnn-mcc.final-train-history.rds")

tdnn_mcc.final.eval_result.file <- file.path(dnn_mcc.tuner.dir,
                                        "tdnn-mcc.final.eval-result.rds")

tdnn_mcc.final.plot_img.file <- file.path(dnn_mcc.tuner.plots.dat.dir, 
                                          "tdnn_mcc.final-model.png")


tdnn_mcc.final.eval.plots_dat.file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                          "tdnn-mcc.final.eval.plots_dat.rds")

tdnn_mcc.final.eval.conf.mx.img_file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                                  "tdnn-mcc.final.eval.confusion-matrix.png")

tdnn_mcc.final.eval.plots_dat.file <- file.path(dnn_mcc.tuner.plots.dat.dir,
                                                "tdnn-mcc.final.eval.plots_dat.rds")

if(!dir.exists(dnn_mcc.tuner.dir))
  dir.create(dnn_mcc.tuner.dir)

if(!dir.exists(dnn_mcc.tuner.plots.dat.dir))
  dir.create(dnn_mcc.tuner.plots.dat.dir)

#### Run Scripts ---------------------------------------------------------------

if(!file.exists(tdnn_mcc.final.eval_result.file)) {
  if(!file.exists(tdnn_mcc.final.file)) {
    if(!file.exists(tdnn_mcc.best_hp.config.file)) {
      source(dnn_mcc.tuner.script.path, 
             catch.aborts = TRUE,
             echo = TRUE,
             spaced = TRUE,
             verbose = TRUE,
             keep.source = TRUE)
    }

    source(tdnn_mcc.final.retrain.script.path,
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }

  source(tdnn_mcc.final.eval.script.path,
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### Final Model Evaluation Results Visualization ------------------------------

open_logfile(".tdnn-mcc.visual.eval-results")
stopifnot(file.exists(tdnn_mcc.final.file,
                      tdnn_mcc.final.eval_result.file,
                      model_visualization.shared.script.path),
          exists("tdnn_mcc.final.plot_img.file"))

put_log("Loading pre-trained TDNN MCC Final Model...")
tdnn_mcc.final <- keras3::load_model(tdnn_mcc.final.file)

put_log("The TDNN MCC Final Model has been loaded from the backup file:
%1", tdnn_mcc.final.file)

tdnn_mcc.final |> plot_keras_model(to_file = tdnn_mcc.final.plot_img.file,
                             show_shapes = T)
# rm(tdnn_mcc.final)

if(file.exists(tdnn_mcc.final.train_history.file)){
  put_log("Loading the Tuned DNN MCC Final Model Train History...")
  tdnn_mcc.final.train_history <- readRDS(tdnn_mcc.final.train_history.file)
  
  put_log("The Tuned DNN MCC Final Model Train History has been loaded 
from the following file:
%1", tdnn_mcc.final.train_history.file)
  
  plot(tdnn_mcc.final.train_history)
  
  best_metrics <- model.train_history.get_best_metrics(tdnn_mcc.final.train_history)
  put_log("The best values of the Tuned DNN MCC Final Model training result are as follows:
%1", capture.output(best_metrics))
#  accuracy         loss val_accuracy     val_loss 
# 0.9337805    0.1783210    0.9070931    0.2982191   
  
  # rm(tdnn_mcc.final.train_history)
} else {
  warning("The Tuned DNN MCC Final Model History backup file does not exist:
%1", tdnn_mcc.final.train_history.file)
}

put_log("Loading the Tuned DNN-Based MCC Final Model Evaluation Result object...")
tdnn_mcc.final.eval.result <- readRDS(tdnn_mcc.final.eval_result.file)

put_log("The Tuned DNN-Based MCC Final Model Evaluation Result object 
has been loaded from the following file:
%1", tdnn_mcc.final.eval_result.file)

put_log("The Tuned DNN-Based MCC Final Model Evaluation Result accuracy: %1", 
        tdnn_mcc.final.eval.result$accuracy)
# 0.9035151

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = tdnn_mcc.final.eval.result$targets,
                              predicted.probabilities = tdnn_mcc.final.eval.result$predicted.probs,
                              predicted.values = tdnn_mcc.final.eval.result$predicted.values,
                              model_type = "Tuned MCC",
                              alg_name = "DNN",
                              plots_dat.file = tdnn_mcc.final.eval.plots_dat.file,
                              pca.export_img.file_name = "tdnn.final.eval.pca.png",
                              pca.export_img.dir = dnn_mcc.tuner.plots.dat.dir,
                              cm.export.img_file = tdnn_mcc.final.eval.conf.mx.img_file,
                              cm.print.image = T)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()
## CNN-based MCC Model ---------------------------------------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

### CNN-Based Basic MCC Model --------------------------------------------------
stopifnot(file.exists(cnnb_mcc.script.path,
                      cnnb_mcc.eval.script.path))

#### Init Paths ----------------------------------------------------------------

# cnn_mcc.x3d.test_set.bakup <- file.path(data.cnn_mcc.dir,
#                                         "x3d.test_set.rds")

cnnb_mcc.file <- file.path(data.cnnb_mcc.dir, 
                           "cnnb_mcc.pre-trained.keras")

cnnb_mcc.train_history.file <- file.path(data.cnnb_mcc.dir, 
                                         "cnnb_mcc.train_history.rds")

cnnb_mcc.eval.result.file <- file.path(data.cnnb_mcc.dir,
                                       "cnnb_mcc.eval.result.rds")

cnnb_mcc.plot_img.file <- file.path(cnnb_mcc.plots.dat.dir,
                                    "cnnb_mcc.model.png")

cnnb_mcc.eval.cm_img.file <- file.path(cnnb_mcc.plots.dat.dir,
                                       "cnnb_mcc.eval.cm.png")

cnnb_mcc.eval.plots_dat.file <- file.path(cnnb_mcc.plots.dat.dir,
                                          "cnnb_mcc.eval.plots_dat.rds")

if(!dir.exists(data.cnnb_mcc.dir))
  dir.create(data.cnnb_mcc.dir)

if(!dir.exists(cnnb_mcc.plots.dat.dir))
  dir.create(cnnb_mcc.plots.dat.dir)

#### Run Scripts ---------------------------------------------------------------

if(!file.exists(cnnb_mcc.eval.result.file)) {
  if(!file.exists(cnnb_mcc.file)) {
    source(cnnb_mcc.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }
  
  source(cnnb_mcc.eval.script.path,
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### Model Evaluation Results Visualization ------------------------------------

open_logfile(".cnnb-mcc.visual.eval-results")

stopifnot(file.exists(cnnb_mcc.file,
                      cnnb_mcc.eval.result.file,
                      model_visualization.shared.script.path))

put_log("Loading the pre-trained CNN-based Multiclass Classifier model...")
cnnb_mcc <- keras3::load_model(cnnb_mcc.file)
put_log("The pre-trained CNN-based Multiclass Classifier model 
has been loaded from the following backup file:
%1", cnnb_mcc.file)

cnnb_mcc |> plot_keras_model(to_file = cnnb_mcc.plot_img.file,
                             show_shapes = T)
# rm(cnnb_mcc)

if(file.exists(cnnb_mcc.train_history.file)){
  put_log("Loading the model training history...")
  cnn_mcc.train_history <- readRDS(cnnb_mcc.train_history.file)
  put_log("The model training history has been loaded from the backup file:
%1", cnnb_mcc.train_history.file)
  
  plot(cnn_mcc.train_history)
  
  best_metrics <- model.train_history.get_best_metrics(cnn_mcc.train_history)
  put_log("The best values of the CNNB MCC model training result are as follows:
%1", capture.output(best_metrics))
#  accuracy         loss val_accuracy     val_loss 
# 0.9070390    0.2745264    0.9250632    0.2219673 

  rm(cnn_mcc.train_history)
} else {
  warning("The CNNB MCC model history backup does not exist:
", cnnb_mcc.train_history.file)
}

put_log("Loading the CNNB MCC Model Evaluation Result object...")
cnnb_mcc.eval.result <- readRDS(cnnb_mcc.eval.result.file)

put_log("The CNNB MCC Model Evaluation Result object has been loaded 
from the following file:
%1", cnnb_mcc.eval.result.file)

put_log("The CNNB MCC Model Evaluation Result accuracy: %1", 
        cnnb_mcc.eval.result$accuracy)
# 0.92193329334259

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = cnnb_mcc.eval.result$targets,
                              predicted.probabilities = cnnb_mcc.eval.result$predicted.probs,
                              predicted.values = cnnb_mcc.eval.result$predicted.values,
                              plots_dat.file = cnnb_mcc.eval.plots_dat.file,
                              model_type = "Basic MCC",
                              alg_name = "CNN",
                              pca.export_img.file_name = "cnn-mcc.eval.pca.png",
                              pca.export_img.dir = cnnb_mcc.plots.dat.dir,
                              cm.export.img_file = cnnb_mcc.eval.cm_img.file,
                              cm.print.image = T)

put_log("The `plots.args` object of class `%1` has been created for use to generate 
a visual representation of the CNNB MCC model evaluation results.",
        class(plots.args))

# rm(cnnb_mcc.eval.result)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

log_close()

### CNN-Based MCC Model Tuning -------------------------------------------------
stopifnot(file.exists(cnn_mcc.arch_tuner.script.path,
                      cnn_mcc.lr_tuner.script.path,
                      tcnn_mcc.final.retrain.script.path,
                      tcnn_mcc.final_test.script.path))

#### Init Paths ----------------------------------------------------------------

tcnn_mcc.arch.best_hp.config.file <- file.path(cnn_mcc.tuner.dir, 
                                         paste0('arch-tuned.best-hp.config', 
                                                '.rds'))

tcnn_mcc.best_hp.config.file <- file.path(cnn_mcc.tuner.dir,
                                          "tcnn-mcc.best-hp.config.rds")

tcnn_mcc.final.file <- file.path(cnn_mcc.tuner.dir, 
                                 "tcnn-mcc.final-model.keras")

tcnn_mcc.final.train_history.file <- file.path(cnn_mcc.tuner.dir, 
                                               "tcnn-mcc.final-train-history.rds")

tcnn_mcc.final.eval_result.file <- file.path(cnn_mcc.tuner.dir,
                                             "tcnn-mcc.final.eval-result.rds")

tcnn_mcc.final.plot_img.file <- file.path(cnn_mcc.tuner.plots.dat.dir, 
                                          "tcnn_mcc.final-model.png")

tcnn_mcc.final.eval.plots_dat.file <- file.path(cnn_mcc.tuner.plots.dat.dir,
                                                "tcnn-mcc.final.eval.plots_dat.rds")

tcnn_mcc.final.eval.conf.mx.img_file <- file.path(cnn_mcc.tuner.plots.dat.dir,
                                                  "tcnn-mcc.final.eval.confusion-matrix.png")

tcnn_mcc.final.eval.plots_dat.file <- file.path(cnn_mcc.tuner.plots.dat.dir,
                                                "tcnn-mcc.final.eval.plots_dat.rds")

if(!dir.exists(cnn_mcc.tuner.dir))
  dir.create(cnn_mcc.tuner.dir)

if(!dir.exists(cnn_mcc.tuner.plots.dat.dir))
  dir.create(cnn_mcc.tuner.plots.dat.dir)

#### Run Scripts ---------------------------------------------------------------

if(!file.exists(tcnn_mcc.final.eval_result.file)) {
  if(!file.exists(tcnn_mcc.final.file)) {
    if(!file.exists(tcnn_mcc.best_hp.config.file)) {
      if(!file.exists(tcnn_mcc.arch.best_hp.config.file)) {
        source(cnn_mcc.arch_tuner.script.path, 
               catch.aborts = TRUE,
               echo = TRUE,
               spaced = TRUE,
               verbose = TRUE,
               keep.source = TRUE)
      }
      
      source(cnn_mcc.lr_tuner.script.path,
             catch.aborts = TRUE,
             echo = TRUE,
             spaced = TRUE,
             verbose = TRUE,
             keep.source = TRUE)
    }
    
    source(tcnn_mcc.final.retrain.script.path,
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
  }
  
  source(tcnn_mcc.final_test.script.path,
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
}

#### Final Test Results Visualization ------------------------------------------

open_logfile(".tcnn-mcc.visual.eval-results")

stopifnot(file.exists(tcnn_mcc.final.file,
                      tcnn_mcc.final.eval_result.file,
                      model_visualization.shared.script.path),
          exists("tcnn_mcc.final.plot_img.file"))

put_log("Loading pre-trained Tuned CNN-Based MCC Final Model...")
tcnn_mcc.final <- keras3::load_model(tcnn_mcc.final.file)

put_log("The Tuned CNN-Based MCC Final Model has been loaded from the backup file:
%1", tcnn_mcc.final.file)

tcnn_mcc.final |> plot_keras_model(to_file = tcnn_mcc.final.plot_img.file,
                                   show_shapes = T)
# rm(tcnn_mcc.final)

if(file.exists(tcnn_mcc.final.train_history.file)){
  put_log("Loading the Tuned CNN-Based MCC Final Model Train History...")
  tcnn_mcc.final.train_history <- readRDS(tcnn_mcc.final.train_history.file)
  
  put_log("The Tuned CNN-Based MCC Final Model Train History has been loaded 
from the following file:
%1", tcnn_mcc.final.train_history.file)
  
  plot(tcnn_mcc.final.train_history)
  # rm(tcnn_mcc.final.train_history)
  
  best_metrics <- model.train_history.get_best_metrics(tcnn_mcc.final.train_history)
  put_log("The best values of the Tuned CNN-Based MCC Final Model training result are as follows:
%1", capture.output(best_metrics))
#  accuracy     f1_macro         loss val_accuracy val_f1_macro     val_loss 
# 0.9406126    0.9207393    0.1823343    0.9454636    0.9272678    0.1663086 
} else {
  warning("The Tuned CNN-Based MCC Final Model History backup file does not exist:
%1", tcnn_mcc.final.train_history.file)
}

put_log("Loading the Tuned CNN-Based MCC Final Model Evaluation Result object...")
tcnn_mcc.final.eval.result <- readRDS(tcnn_mcc.final.eval_result.file)

put_log("The Tuned CNN-Based MCC Final Model Evaluation Result object 
has been loaded from the following file:
%1", tcnn_mcc.final.eval_result.file)

put_log("The Tuned CNN-Based MCC Final Model Evaluation Result metrics: 
%1", capture.output(c('accuracy' = tcnn_mcc.final.eval.result$accuracy,
                         'f1_macro' = tcnn_mcc.final.eval.result$f1_macro,
                         'loss' = tcnn_mcc.final.eval.result$loss)))
#  accuracy  f1_macro      loss 
# 0.9487213 0.9190166 0.1859616 

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = tcnn_mcc.final.eval.result$targets,
                              predicted.probabilities = tcnn_mcc.final.eval.result$predicted.probs,
                              predicted.values = tcnn_mcc.final.eval.result$predicted.values,
                              model_type = "Fine-Tuned MCC",
                              alg_name = "CNN",
                              plots_dat.file = tcnn_mcc.final.eval.plots_dat.file,
                              pca.export_img.file_name = "tcnn.final.eval.pca.png",
                              pca.export_img.dir = cnn_mcc.tuner.plots.dat.dir,
                              cm.export.img_file = tcnn_mcc.final.eval.conf.mx.img_file,
                              cm.print.image = T)

#'Run the helper script specifically designed to visualize 
#'the MCC models evaluation results:
source(model_visualization.shared.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

##### Reviewing the Most Confusion Error Results --------------------------------
###### Target Value `1`--------------------------------------------------------

# Mistakenly predicted as `I` (47 cases of misclassification)
missclass.1_as_I <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                      tcnn_mcc.final.eval.result$targets,
                                      x_test.files,
                                      pred.char = 'I',
                                      target.char = '1')
str(missclass.1_as_I)
# dev.off()
print.missclass_image.grid(missclass.1_as_I,
                 tile = '6x100')
                 # geometry = "x160+5+5",
                 # image.annotate = F)

# Mistakenly predicted as `L` (52 cases of misclassification)
missclass.1_as_L <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                           tcnn_mcc.final.eval.result$targets,
                                           x_test.files,
                                           pred.char = 'L',
                                           target.char = '1')
str(missclass.1_as_L)
# dev.off()
print.missclass_image.grid(missclass.1_as_L,
                           tile = '6x100')

###### Target Value `I`-------------------------------------------------------------

# Mistakenly predicted as `1` (147 cases of misclassification)
missclass.I_as_1 <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                          tcnn_mcc.final.eval.result$targets,
                                          x_test.files,
                                          pred.char = '1',
                                          target.char = 'I')
str(missclass.I_as_1)
# dev.off()
print.missclass_image.grid(missclass.I_as_1,
                           tile = '6x100')
# geometry = "x160+5+5",
# image.annotate = F)

# Mistakenly predicted as `L` (33 cases of misclassification)
missclass.I_as_L <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                          tcnn_mcc.final.eval.result$targets,
                                          x_test.files,
                                          pred.char = 'L',
                                          target.char = 'I')
str(missclass.I_as_L)
# dev.off()
print.missclass_image.grid(missclass.I_as_L,
                           tile = '6x100')

###### Target Value `Q`-------------------------------------------------------------

# Mistakenly predicted as `9` (100 cases of misclassification)
missclass.Q_as_9 <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                          tcnn_mcc.final.eval.result$targets,
                                          x_test.files,
                                          pred.char = '9',
                                          target.char = 'Q')
str(missclass.Q_as_9)
# dev.off()
print.missclass_image.grid(missclass.Q_as_9,
                           tile = '6x100')

###### Target Value `G`-------------------------------------------------------------

# Mistakenly predicted as `9` (53 cases of misclassification)
missclass.G_as_9 <- recognition_err.table(tcnn_mcc.final.eval.result$predicted.values,
                                          tcnn_mcc.final.eval.result$targets,
                                          x_test.files,
                                          pred.char = '9',
                                          target.char = 'G')
str(missclass.G_as_9)
# dev.off()
print.missclass_image.grid(missclass.G_as_9,
                           tile = '6x100')

# Appendix: The Device (laptop) Info Where the Project was Build & Tested ---------------

# Processor	13th Gen Intel(R) Core(TM) i7-13620H (2.40 GHz)
# Installed RAM	32.0 GB (31.7 GB usable)
# System type	64-bit operating system, x64-based processor

# Edition	Windows 11 Pro
# Version	25H2
# Installed on	‎12/‎14/‎2024
# OS build	26200.8973
# Experience	Windows Feature Experience Pack 1000.26100.344.0


