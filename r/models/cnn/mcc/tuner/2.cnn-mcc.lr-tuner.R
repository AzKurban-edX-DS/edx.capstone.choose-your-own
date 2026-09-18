## Setup -----------------------------------------------------------------------
open_logfile(".cnn_mcc.model-tuning.learning-rate")
stopifnot(exists("cnn_mcc.tuner.dir"),
          exists("tcnn_mcc.arch.best_hp.config.file"))

start <- put_start_date()


### Tuning the `learning rate` parameter ---------------------------------------------


cnn_mcc.lr_tuner.proj.dir <- file.path(cnn_mcc.tuner.dir, 
                            '2.lr.fine-tuning.prj')


cnn_mcc.lr_tuner.checkpoints.dir <- file.path(cnn_mcc.lr_tuner.proj.dir, 
                                   "checkpoints")


cnn_mcc.lr_tuner.best_model.plot.img_file <- file.path(cnn_mcc.tuner.proj.arch.dir,
                                            paste0('lr-tuned.best-model.plot', 
                                                   '.png'))

put_log("Loading the Best Hyperparameter Configuration of the tuned architecture...")
best_hp.config <- readRDS(tcnn_mcc.arch.best_hp.config.file)

put_log("The best Hyperparameters of the tuned architecture have been loaded from the following file:
%1", tcnn_mcc.arch.best_hp.config.file)

# Build the HyperParameters object from the configuration
kt <- import("keras_tuner")
hp <- kt$HyperParameters$from_config(best_hp.config)
rm(best_hp.config)

put_log("The best Hyperparameters values:
%1", capture.output(hp$values))
{
  # $conv_blocks
  # [1] 2
  # 
  # $conv_padding
  # [1] "same"
  # 
  # $dense_units
  # [1] 320
  # 
  # $dropout1
  # [1] 0.3
  # 
  # $dropout2
  # [1] 0.2
  # 
  # $learning_rate
  # [1] 1e-04
  # 
  # $conv1_filters
  # [1] 224
  # 
  # $conv1_kernel.size
  # [1] 5
  # 
  # $conv2_filters
  # [1] 224
  # 
  # $conv2_kernel.size
  # [1] 5
  # 
  # $conv3_filters
  # [1] 256
  # 
  # $conv3_kernel.size
  # [1] 4
  # 
  # $`tuner/epochs`
  # [1] 10
  # 
  # $`tuner/initial_epoch`
  # [1] 4
  # 
  # $`tuner/bracket`
  # [1] 1
  # 
  # $`tuner/round`
  # [1] 1
  # 
  # $`tuner/trial_id`
  # [1] "0052"  
  
  invisible()
}

hp$Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

tuner.result <- 
  cnn_mcc.Hyperband.fit_tuner(hp,
                              x_train,
                              y_train,
                              validation.data = tuple(x_test, y_test),
                              project.dir = cnn_mcc.lr_tuner.proj.dir,
                              project.name = 'tuner.dat',
                              checkpoints.dir = cnn_mcc.lr_tuner.checkpoints.dir,
                              max_epochs = 15L)


if(!is.null(tuner.result$error) ||
   !is.null(tuner.result$hypermodel$error)) {
  put_log("Some error(s) occurred while tuning.")
  
  if(!is.null(tuner.result$error))
    put_log(tuner.result$error)
  
  if(!is.null(tuner.result$hypermodel$error))
    put_log(tuner.result$hypermodel$error)
}

tuner <- tuner.result$tuner

# This prints a summary of the search space and lists the top trial results
tuning_result <- kerastuneR::plot_tuner(cnn_mcc.arch_tuner)
# the list will show the plot and the data.frame of tuning results

put_log("The CNN MCC Tuning Results:
%1", capture.output(tuning_result))
rm(tuning_result)

tuner$results_summary()
{
  invisible()
}

tuner.best_hp <- 
  tuner$get_best_hyperparameters(num_trials = 1L)[[1]]

put_log("The best Hyperparameters values:
%1", capture.output(tuner.best_hp$values))
{
  # $conv_blocks
  # [1] 2
  # 
  # $conv_padding
  # [1] "same"
  # 
  # $dense_units
  # [1] 320
  # 
  # $dropout1
  # [1] 0.3
  # 
  # $dropout2
  # [1] 0.2
  # 
  # $learning_rate
  # [1] 1e-04
  # 
  # $conv1_filters
  # [1] 224
  # 
  # $conv1_kernel.size
  # [1] 5
  # 
  # $conv2_filters
  # [1] 224
  # 
  # $conv2_kernel.size
  # [1] 5
  # 
  # $conv3_filters
  # [1] 256
  # 
  # $conv3_kernel.size
  # [1] 4
  # 
  # $`tuner/epochs`
  # [1] 10
  # 
  # $`tuner/initial_epoch`
  # [1] 4
  # 
  # $`tuner/bracket`
  # [1] 1
  # 
  # $`tuner/round`
  # [1] 1
  # 
  # $`tuner/trial_id`
  # [1] "0052"  
  
  invisible()
}

best_hp.config <- tuner.best_hp$get_config()
put_log("The best Hyperparameters configuration:
%1", capture.output(best_hp.config))


put_log("Saving the Best Hyper-parameter Configuration...")
saveRDS(best_hp.config,
        file = tcnn_mcc.best_hp.config.file)

rm(best_hp.config)

put_log("The Best Hyperparameters Configuration have been saved to the following file:
%1", tcnn_mcc.best_hp.config.file)


best_trial <- 
  tuner$oracle$get_best_trials(num_trials = 1L)[[1]]

put_log("The best step of the best trial: %1", best_trial$best_step)

best_trial$summary()

rm(best_trial)

#### Retrieving the Best Model ------------------------------------------------

#best_model <- tcnn_mcc.best_models[[1]]
# rm(tcnn_mcc.best_models)


best_model <- 
  kerastuneR::get_best_models(tuner = tuner, 
                              num_models = 1L)[[1]]

best_model |> plot_keras_model(to_file = cnn_mcc.lr_tuner.best_model.plot.img_file,
                                             show_shapes = T)

# put_log("Saving the CNN MCC Best Model...")
# keras3::save_model(tcnn_mcc.best_model,
#                    file = best_model.file,
#                    overwrite = TRUE)

# put_log("The CNN MCC Best Model object has been saved in the following file:
#   %1", best_model.file)

rm(best_mode)
#### (Alternatively) Building the model from the Best Hyper-parameters ---------

best_hp.model <- 
  tuner.result$hypermodel$build(tuner.best_hp)

put_log("Summary of the Tuned Model built from the best hyper-parameters:
%1", capture.output(best_hp.model))

rm(best_hp.model)

log_close()
