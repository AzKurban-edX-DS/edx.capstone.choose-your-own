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

### Prepare Input Datasets -----------------------------------------------------
stopifnot(file.exists(prepare_ds.script.path))

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
                                       "dnnb_mcc.pre-trained.model.keras")

dnnb_mcc.train_history.file <- file.path(data.dnn_mcc.basic.dir, 
                                                     "dnnb_mcc.train_history.bak.rds")

dnnb_mcc.eval.result.file <- file.path(data.dnn_mcc.basic.dir,
                                       "dnnb_mcc.eval.result.rds")

dnnb_mcc.eval.conf.mx.img_file <- file.path(dnn_mcc.basic.plots.dat.dir,
                                            "dnn-basic.mcc.eval.confusion-matrix.png")

dnnb_mcc.eval.plots_dat.file <- file.path(dnn_mcc.basic.plots.dat.dir,
                                          "dnn-basic.mcc.eval.plots_dat.rds")

if(!dir.exists(dnn_mcc.basic.plots.dat.dir))
  dir.create(dnn_mcc.basic.plots.dat.dir)

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

open_logfile(".dnnb-mcc.visual.eval-results")

stopifnot(file.exists(model_visualization.shared.script.path))

put_log("Loading the BDL MCC Model Evaluation Result object...")
dnnb_mcc.eval.result <- readRDS(dnnb_mcc.eval.result.file)

put_log("The BDL MCC Model Evaluation Result object has been loaded 
from the following file:
%1", dnnb_mcc.eval.result.file)

#' Initialize the `plots.args` object containing argument values 
#' for the visualization helper functions being called in the following script 
#' about to launch:
plots.args <- init.plots_args(targets = dnnb_mcc.eval.result$targets,
                              predicted.probabilities = dnnb_mcc.eval.result$predicted.probs,
                              predicted.values = dnnb_mcc.eval.result$predicted.values,
                              plots_dat.file = dnnb_mcc.eval.plots_dat.file,
                              model_type = "Basic MCC",
                              alg_name = "DNN",
                              pca.export_img.file_name = "dnn-mcc.basic.eval.pca.png",
                              pca.export_img.dir = dnn_mcc.basic.plots.dat.dir,
                              cm.export.img_file = dnnb_mcc.eval.conf.mx.img_file,
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
                                               "dnn-mcc.tuner.best-hp.config.rds")

tdnn_mcc.final.file <- file.path(dnn_mcc.tuner.dir, 
                                     "tdnn-mcc.final-model.keras")

tdnn_mcc.final.train_history.file <- file.path(dnn_mcc.tuner.dir, 
                                                      "tdnn-mcc.final-train-history.rds")

tdnn_mcc.final.eval_result.file <- file.path(dnn_mcc.tuner.dir,
                                        "tdnn-mcc.final.eval-result.rds")

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

open_logfile(".tdnn-mcc.visual.eval-results")

put_log("Loading the BDL MCC Final Model Evaluation Result object...")
tdnn_mcc.final.eval.result <- readRDS(tdnn_mcc.final.eval_result.file)

put_log("The BDL MCC Final Model Evaluation Result object has been loaded 
from the following file:
%1", tdnn_mcc.final.eval_result.file)

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
## CNN-based Multiclass Classifier (CNN MCC) Model ----------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

#### Initial Paths -------------------------------------------------------------

open_logfile(".ds.prepare.train&test.balanced_sets")

cnn_mcc.x3d.test_set.bakup <- file.path(data.cnn_mcc.dir,
                                        "x3d.test_set.rds")

cnn_mcc.basic.file_path <- file.path(data.cnn_mcc.basic.dir, 
                                     "cnn.pre-trained.multiclass.model.keras")
cnn_mcc.basic.train_history.file_path <- file.path(data.cnn_mcc.basic.dir,
                                             "cnn_mcc.train_history.backup.rds")

cnn_mcc.final.file <- file.path(data.cnn_mcc.tuner.best.dir, 
                                     "cnn_mcc.final-model.keras")

cnn_mcc.final.train_history.file <- file.path(data.cnn_mcc.tuner.best.dir, 
                                              "cnn.mcc-final.train-history.rds")

#### Build Basic CNN MCC Model -------------------------------------------------
stopifnot(file.exists(cnn_mcc.basic.script.path))

put_log("Defining and training a CNN-based Multiclass Classifier Model...")

source(cnn_mcc.basic.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

# Evaluate the pre-trained Basic CNN MCC Model 
source(cnn_mcc.basic.eval.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


#### Tuning CNN MCC Model ------------------------------------------------------

if(file.exists(cnn_mcc.final.file)) {
  if(file.exists(cnn_mcc.final.train_history.file)){
    put_log("Loading the tuned Final MCC Model Train History...")
    
    cnn_mcc.final.train_history <- readRDS(cnn_mcc.final.train_history.file)
    
    put_log("The Tuned Final MCC Model Training History has been loaded from the backup file:
%1", cnn_mcc.final.train_history.file)
    
    put_log("The Tuned Final MCC Model Training History Summary:
%1", capture.output(cnn_mcc.final.train_history))
    plot(cnn_mcc.final.train_history)
  } else {
    warning("The tuned Final MCC Model backup does not exist:
", cnn_mcc.final.train_history.file)
  }
  
  put_log("Evaluating the tuned and re-trained Final CNN MCC Model")
  stopifnot(file.exists(cnn_mcc_final.eval.script.path))
  
  # source(cnn_mcc_final.eval.script.path, 
  #        catch.aborts = TRUE,
  #        echo = TRUE,
  #        spaced = TRUE,
  #        verbose = TRUE,
  #        keep.source = TRUE)
} else {
    stopifnot(file.exists(cnn_mcc.model_tuner.script.path))
    
    put_log("Tuning the CNN-based Multiclass Classifier Model...")
    source(cnn_mcc.model_tuner.script.path, 
           catch.aborts = TRUE,
           echo = TRUE,
           spaced = TRUE,
           verbose = TRUE,
           keep.source = TRUE)
}

log_close()

## CNN-based Binary Classifier Models -----------------------------------------
stopifnot(file.exists(cnn_binary.r_scripts.dir))

data.cnn.binary.dir <- file.path(data.dl.cnn.dir, "binary")

if(!dir.exists(data.cnn.binary.dir))
  dir.create(data.cnn.binary.dir)


data.cnn.binary.models.dir <- file.path(data.cnn.binary.dir, "models")

if(!dir.exists(data.cnn.binary.models.dir))
  dir.create(data.cnn.binary.models.dir)

data.cnn.binary.models.checkpoints.dir <- file.path(data.cnn.binary.models.dir,
                                                    "checkpoints")
if(!dir.exists(data.cnn.binary.models.checkpoints.dir))
  dir.create(data.cnn.binary.models.checkpoints.dir)

data.cnn.binary.models.evaluation.dir <- file.path(data.cnn.binary.models.dir,
                                                   "evaluation")
if(!dir.exists(data.cnn.binary.models.evaluation.dir))
  dir.create(data.cnn.binary.models.evaluation.dir)

# source(cnn_binary.r_scripts.dir,
#        catch.aborts = TRUE,
#        echo = TRUE,
#        spaced = TRUE,
#        verbose = TRUE,
#        keep.source = TRUE)

## Final Test for the Best Models ----------------------------------------------
### Preparing the Final Test Data ----------------------------------------------

open_logfile(".ds.prepare.final-test.balanced_sets")

put_log("Preparing a Final Test Set for validating the CNN-based Models...")
start <- put_start_date()
stopifnot(file.exists(final_test.img28x28mx.array.file_path))

put_log("Loading the Final Test 28x28 Image Data Array Set from the backup file...")
ft.img_mx.set <- readRDS(final_test.img28x28mx.array.file_path)

put_log("The Final Test 28x28 Image Data Array Set has been loading from the following file:
%1", final_test.img28x28mx.array.file_path)

put_log("The Final Test 28x28 Image Data Set structure:
%1", capture.output(str(ft.img_mx.set)))

##### Creating Final Test Dataset -----------------------------------------------
put_log("Making a balanced sample from the Validation 28x28 Image Data Array...")

set.seed(N.classes)
ft.sample_set <- sample_train_test_sets.x3d(ft.img_mx.set$img28x28mx.array,
                                           ft.img_mx.set$img28x28mx.fpath,
                                           test.ratio = 1)

put_log("The Final Test Set sample has been made from the Validation 28x28 Image Data Array,
which is returned in an object with the following structure:
%1", capture.output(str(ft.sample_set)))
put_end_date(start)

ft.x3d.test_set <- ft.sample_set$test_set
put_log("The Test Set has been saved in the object `ft.x3d.test_set`, 
which contains a testing sample stored in the `x.test` variable having the following shape:
%1", capture.output(shape(ft.x3d.test_set$x.test)))
# shape(4641, 28, 28)

# rm(ft.sample_set)

log_close()

### Final Testing of CNN BCC-Based Ensemble ------------------------------------
stopifnot(file.exists(cnn_binary.ensemble.script.path))


source(cnn_binary.ensemble.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

### Final Testing of the CNN-Based Multiclass Classifier Model -----------------
stopifnot(file.exists(cnn_mcc.model.final_test.file_path))

x3d.test_set <- ft.x3d.test_set
rm(ft.x3d.test_set)

cnn_mcc.model.final_test.file_path <- file.path(data.cnn_mcc.dir, 
                                                "cnn.multiclass.model.final-test.RData")
stopifnot(file.exists(cnn_mcc.model.final_test.file_path))
saaaaaaaaaaaaaaaaaaaaaaaaaasssssssss
source(cnn_mcc.model.final_test.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

# Appendix: The Device (laptop) Info Where the Project was Build & Tested ---------------

# Processor	13th Gen Intel(R) Core(TM) i7-13620H (2.40 GHz)
# Installed RAM	32.0 GB (31.7 GB usable)
# System type	64-bit operating system, x64-based processor

# Edition	Windows 11 Pro
# Version	25H2
# Installed on	‎12/‎14/‎2024
# OS build	26200.8973
# Experience	Windows Feature Experience Pack 1000.26100.344.0


