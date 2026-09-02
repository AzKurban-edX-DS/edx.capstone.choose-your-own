#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# kNN+PCA MCC: The Best Model Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stopifnot(file.exists(my_emnist.split.file_path,
                      k_best.nn_pca.model.backup.path))

open_logfile(".eval.k(best)nn+pca")
start <- put_start_date()

## Prepare Test Datasets -------------------------------------------------------

start <- put_start_date()

### Loading Split Flattened Dataset allocated 10% for the Training Set ---------

put_log("Loading the Split Flattened Dataset from the backup file...")

ds <- load_datasets(my_emnist.split.file_path)
str(ds)

x_test <- ds$test$x

put_log("The Test set is balanced with respect to the set of classes:
%1", capture.output(print(ds$test$class_groups$groupByClass, n = N.classes)))
{
  # # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         852
  #  2 $         852
  #  3 &         852
  #  4 @         852
  #  5 0         852
  #  6 1         852
  #  7 2         852
  #  8 3         852
  #  9 4         852
  # 10 5         852
  # 11 6         852
  # 12 7         852
  # 13 8         852
  # 14 9         852
  # 15 A         852
  # 16 B         852
  # 17 C         852
  # 18 D         852
  # 19 E         852
  # 20 F         852
  # 21 G         852
  # 22 H         852
  # 23 I         852
  # 24 J         852
  # 25 K         852
  # 26 L         852
  # 27 M         852
  # 28 N         852
  # 29 P         852
  # 30 Q         852
  # 31 R         852
  # 32 S         852
  # 33 T         852
  # 34 U         852
  # 35 V         852
  # 36 W         852
  # 37 X         852
  # 38 Y         852
  # 39 Z         852
  invisible(NULL)
}

y_test <- ds$test$class_groups$classID

stopifnot(sum(as.character(y_test) != rownames(x_test)) == 0)
stopifnot(nrow(x_test) == length(y_test))

rm(ds)

## Evaluating the kNN+PCA MCC Model (for the best *k* Parameter value) ---------

put_log("Loading the `kNN+PCA MCC` Model (trained for the best `k` value) from the backup file...")

k_best.nn_pca.model -> readRDS(k_best.nn_pca.model.backup.path)
put_end_date(start)
# Time difference of 

put_log("The `kNN+PCA MCC` Model trained for the best `k` value 
has been loaded from the following backup file:
%1", k_best.nn_pca.model.backup.path)

knn_pca.eval.results <- list()

put_log("Constructing predictions using the `kNN+PCA MCC` Model trained for the best *k* value...")

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

knn_pca.eval.results$predicted.probs <- caret::predict.train(k_best.nn_pca.model, 
                                                             newdata = x_test,
                                                             type = "prob",
                                                             verbose = TRUE)
put_end_date(start)

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)
# Time difference of 2.354987 hours

knn_pca.eval.results$predicted <- 
  predicted_probs2classes(as.matrix(knn_pca.eval.results$predicted.probs), 
                          Y.Labels)

put_end_date(start)

knn_pca.eval.results$accuracy <- mean(k_best.nn_pca.predicted == y_test)
# [1] 0.8625557

knn_pca.eval.results$targets <- y_test


put_log("Saving the Tuned `kNN+PCA MCC` Model evaluation results...")
saveRDS(knn_pca.eval.results,
        file = knn_pca.eval.results.backup)

put_log("The evaluation results of the tuned `kNN+PCA MCC` Model has been saved 
to the following file:
%1", knn_pca.eval.results.backup)

rm(x_test,
   k_best.nn_pca.model,
   y_test)

put_log("Accuracy of the predicted data for the `kNN+PCA MCC` Model tuned by *k* parameter:
%1", knn_pca.eval.results$accuracy)
#> [1] 0.862555675935958

log_close()
# Log Elapsed Time: 0 02:15:36

