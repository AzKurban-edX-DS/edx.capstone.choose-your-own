#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#> kNN+PCA Multiclass Classifier (MCC) Model: 
#> Building & tuning based on the `k` parameter value
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> k-Nearest Neighbors with Principal Component Analysis (kNN+PCA) and 
#> Random Forest (RF) Multiclass Classifier Models

# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro

# options(timeout = max(1000, getOption("timeout")))
# options(expressions = 50000) # Increases nesting limit if needed


## Prepare Input Datasets ------------------------------------------------------

stopifnot(file.exists(my_emnist.0.1split.file_path))

open_logfile(".split.10%train.balanced_subset")
start <- put_start_date()

### Loading Split Flattened Dataset allocated 10% for the Training Set ------------

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.0.1split.file_path)
str(ds)


log_close()
## Model Building & Tuning -----------------------------------------------------
open_logfile(".pre-train-model.k1-8nn+pca")

# if(!is.null(dev.list())) dev.off()
graphics.off()

knn_pca.path = file.path(models.path, "knn-pca")

if(!dir.exists(knn_pca.path))
  dir.create(knn_pca.path)

k1_8nn_pca.model.backup.path <-
  file.path(knn_pca.path, "k1-8nn+pca(0.1train-set).rds")

knn_pca.plot_img.dir <- file.path(knn_pca.path, "plot.img")

if(!dir.exists(knn_pca.plot_img.dir))
  dir.create(knn_pca.plot_img.dir)


if (file.exists(k1_8nn_pca.model.backup.path)) {
  put_log("Loading pre-trained `kNN+PCA MCC` Model 
(tuned for `k` values ranged from 1 to 8) from the backup file...")
  
  k1_8nn_pca.model <- readRDS(k1_8nn_pca.model.backup.path)
  put_end_date(start)
  # Time difference of 
  
  put_log("The pre-trained Model has been loaded from the following file:
%1", k1_8nn_pca.model.backup.path)
} else {
  put_log("Training Model `kNN+PCA` on the 10% size Training Set..." )
  
  start <- put_start_date()
  #flush.console()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
k.values <- seq_len(8)

# The model will be tuned by *k* parameter ranging from 1 to 8 on 10% size sample of the Training Set.

# Reference:
# Dimension reduction with PCA
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#dimension-reduction-with-pca

  k1_8nn_pca.model <- caret::train(x.train, y.train, method = "knn", 
                                preProcess = "pca",
                                trControl = trainControl("cv", 
                                                         number = 5, 
                                                         p = 0.95,
                                                         preProcOptions = list(thresh = 0.9),
                                                         verboseIter = TRUE),
                                tuneGrid = data.frame(k = k.values))
  stopCluster(cl)
  stopImplicitCluster()
  
  # Aggregating results
  # Selecting tuning parameters
  # Fitting k = 5 on full training set
  # Warning in pre_process_options(method, column_types) :
  #   The following pre-processing methods were eliminated: 'pca', 'center', 'scale'  

    put_end_date(start)
  # Time difference of 27.84693 mins
  
  put_log("The Model `kNN+PCA` has been trained on the 10% size Training Set")

  put_log("Saving the pre-trained model in the backup file...")

    saveRDS(k1_8nn_pca.model, 
          file = k1_8nn_pca.model.backup.path)
  
  put_log("The Model `kNN+PCA` pre-trained on the 10% size Training Set 
for *k* values ranged from 1 to 8 has been backed up in the following file:
`%1`", k1_8nn_pca.model.backup.path)

  put_end_date(start)
  # Time difference of 27.88649 mins
}

## The Tuning Results Visualization & Analysis -------------------------------

put_log("The pre-trained `kNN+PCA MCC` Model tuned result:
%1", capture.output(k1_8nn_pca.model))

# k-Nearest Neighbors 

# 16575 samples
# 784 predictor
# 39 classes: '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
# 
# Pre-processing: ignore (784) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 13260, 13260, 13260, 13260, 13260 
# Resampling results across tuning parameters:
#   
#   k  Accuracy   Kappa    
# 1  0.7838914  0.7782043
# 2  0.7677828  0.7616718
# 3  0.7923379  0.7868731
# 4  0.7930015  0.7875542
# 5  0.7955958  0.7902167
# 6  0.7931222  0.7876780
# 7  0.7937255  0.7882972
# 8  0.7924585  0.7869969
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 5.


# The Model tuning visualzation:
trellis.par.set(caretTheme())
plot(k1_8nn_pca.model, 
     main = "`kNN+PCA` Multiclass Classifier Model Tuning Results")

acc.max.idx <- which.max(k1_8nn_pca.model$results$Accuracy)
acc.max.idx
# 5

k1_8nn_pca.max_accuracy <- k1_8nn_pca.model$results$Accuracy[acc.max.idx]
k1_8nn_pca.max_accuracy
# 0.7955958

k1_8nn.best <- k1_8nn_pca.model$results$k[acc.max.idx]
k1_8nn.best
# 5
# 
# k-Nearest Neighbors 

# 75032 samples
# 784 predictor
# 39 classes: '#', '$', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' 
# 
# Pre-processing: principal component signal extraction (743), centered (743), scaled (743), remove (41) 
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 60022, 60023, 60030, 60025, 60028 
# Resampling results across tuning parameters:

# k  Accuracy   Kappa    
# 1  0.8520766  0.8461487
# 3  0.8610591  0.8554102
# 5  0.8632982  0.8576989
# 7  0.8625919  0.8569392

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 5.

log_close()

