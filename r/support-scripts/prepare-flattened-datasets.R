#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare Flattened Training Split Dataset
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stopifnot(exists("my_emnist.split.file"),
          exists("my_emnist.0.1split.file"),
          file.exists(ds.load_flattened.script.path))

## Split the Dataset into Sets with 20% Test Ratio (default) -------------------
if(!file.exists(my_emnist.split.file)) {
  
  put_log("Splitting the Flattened Training Dataset into a Train and Test Sets with 20% Test Ratio...")
  
  set.seed(N.classes)
  ds_flatten.split_list <- sample_train_test_sets.mx(my_emnist.set$img.mx, 
                                                     my_emnist.set$img.file,
                                                     seed = my_emnist.seed)
  
  put_log("The Flattened Training Dataset has been split into a Train and Test Sets:
%1", capture.output(str(ds_flatten.split_list)))
  
  put_log("Saving a backup copy of a split dataset...")
  
  saveRDS(ds_flatten.split_list, 
          my_emnist.split.file)
  
  rm(ds_flatten.split_list)
  
  put_log("The backup copy of a split dataset has been saved in the following file:
%1", my_emnist.split.file)
} else {
  put_log("A Training Split Flattened Dataset with 20% Test Ratio has already been created
and saved tp following file:
%1", my_emnist.split.file)
}

## Split the Dataset into Sets with 90% Test Ratio (10% for Training Set) ------
if(!file.exists(my_emnist.0.1split.file)) {
  put_log("Splitting the Flattened Training Dataset into a Train and Test Sets 
with 90% Test Ratio (10% for Training Set)...")
  
  set.seed(N.classes)
  ds_flatten.0.1split_list <- sample_train_test_sets.mx(my_emnist.set$img.mx, 
                                                        my_emnist.set$img.file,
                                                        test.ratio = 0.9,
                                                        seed = my_emnist.seed)
  rm(x)
  rm(my_emnist.set)
  
  put_log("The Flattened Training Dataset has been split into a Train and Test Sets 
with 90% Test Ration (10% for Training Set):
%1", capture.output(str(ds_flatten.0.1split_list)))
  
  put_log("Saving a backup copy of a split dataset (10% for Training Set)...")
  
  saveRDS(ds_flatten.0.1split_list, 
          my_emnist.0.1split.file)
  
  rm(ds_flatten.0.1split_list)
  
  put_log("The backup copy of a split dataset (10% for Training Set) has been saved in the following file:
%1", my_emnist.0.1split.file)
  
} else {
  put_log("A Training Split Flattened Dataset with 90% Test Ratio has already been created
and saved tp following file:
%1", my_emnist.0.1split.file)
}

rm(my_emnist.set)


