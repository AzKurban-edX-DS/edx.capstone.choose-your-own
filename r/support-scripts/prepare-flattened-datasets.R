#%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare Flatten Dataset
#%%%%%%%%%%%%%%%%%%%%%%%%

open_logfile(".load-input-data")

stopifnot(exists("my_emnist.split.file_path"),
          exists("my_emnist.0.1split.file_path"),
          file.exists(ds.load_flattened.script.path))

if(!file.exists(my_emnist.split.file_path) ||
   !file.exists(my_emnist.0.1split.file_path)) {
  
  source(ds.load_flattened.script.path, 
         catch.aborts = TRUE,
         echo = TRUE,
         spaced = TRUE,
         verbose = TRUE,
         keep.source = TRUE)
  
  stopifnot(exists("my_emnist.set"))
  my_emnist.seed <- nrow(my_emnist.set$img.mx)
  
  #* Split the Dataset into Sets with 20% Test Ratio (default) ***************
  if(!file.exists(my_emnist.split.file_path)) {
    
    put_log("Splitting the Flattened Training Dataset into a Train and Test Sets with 20% Test Ratio...")
    
    set.seed(N.classes)
    ds_flatten.split_list <- sample_train_test_sets.mx(my_emnist.set$img.mx, 
                                                       my_emnist.set$img.file_path,
                                                       seed = my_emnist.seed)
    
    put_log("The Flattened Training Dataset has been split into a Train and Test Sets:
%1", capture.output(str(ds_flatten.split_list)))
    
    put_log("Saving a backup copy of a split dataset...")
    
    saveRDS(ds_flatten.split_list, 
            my_emnist.split.file_path)
    
    rm(ds_flatten.split_list)
    
    put_log("The backup copy of a split dataset has been saved in the following file:
%1", my_emnist.split.file_path)
  } else {
    put_log("A Training Split Flattened Dataset with 20% Test Ratio has already been created
and saved tp following file:
%1", my_emnist.split.file_path)
  }
  
  #* Split the Dataset into Sets with 90% Test Ratio (10% for Training Set) *****
  if(!file.exists(my_emnist.0.1split.file_path)) {
    put_log("Splitting the Flattened Training Dataset into a Train and Test Sets 
with 90% Test Ratio (10% for Training Set)...")
    
    set.seed(N.classes)
    ds_flatten.0.1split_list <- sample_train_test_sets.mx(my_emnist.set$img.mx, 
                                                          my_emnist.set$img.file_path,
                                                          test.ratio = 0.9,
                                                          seed = my_emnist.seed)
    rm(x)
    rm(my_emnist.set)
    
    put_log("The Flattened Training Dataset has been split into a Train and Test Sets 
with 90% Test Ration (10% for Training Set):
%1", capture.output(str(ds_flatten.0.1split_list)))
    
    put_log("Saving a backup copy of a split dataset (10% for Training Set)...")
    
    saveRDS(ds_flatten.0.1split_list, 
            my_emnist.0.1split.file_path)
    
    rm(ds_flatten.0.1split_list)
    
    put_log("The backup copy of a split dataset (10% for Training Set) has been saved in the following file:
%1", my_emnist.0.1split.file_path)
    
  } else {
    put_log("A Training Split Flattened Dataset with 90% Test Ratio has already been created
and saved tp following file:
%1", my_emnist.0.1split.file_path)
  }
  
  rm(my_emnist.set)
} else {
  put_log("A Training Split Flattened Datasets has already been created
and saved tp following files:
  - with 20% Test Ratio (default): 
    %1;
  - with 90% Test Ratio ('lite' Train Set of 10% size): 
    %2.", my_emnist.split.file_path,
          my_emnist.0.1split.file_path)
}

log_close()
