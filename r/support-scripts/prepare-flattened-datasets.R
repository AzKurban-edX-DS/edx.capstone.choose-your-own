#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare Flattened Training Split Dataset
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stopifnot(exists("ds.flatten.split.file"),
          exists("ds.flatten.0.1split.file"),
          file.exists(ds.train.flatten.file))

# Load Flatten Dataset 
put_log("Loading the Flattened Training Dataset from the backup file...")
ds.train.flatten <- readRDS(ds.train.flatten.file)

put_log("The Flattened Training Dataset has been loaded from the following backup:
%1", ds.train.flatten.file)

put_log("The Flattened Dataset Set (containing the  EMNIST-like dataset) has the following structure:
  %1", capture.output(str(ds.train.flatten)))


ds.flatten.seed <- nrow(ds.train.flatten$img.mx)

## Split the Dataset into Sets with 20% Test Ratio (default) -------------------
if(!file.exists(ds.flatten.split.file)) {
  stopifnot()
  
  put_log("Splitting the Flattened Training Dataset into a Train and Test Sets with 20% Test Ratio...")
  
  ds_flatten.split_list <- ds.flatten.sample(ds.train.flatten$img.mx, 
                                                     ds.train.flatten$img.file,
                                                     seed = ds.flatten.seed)
  
  put_log("The Flattened Training Dataset has been split into a Train and Test Sets:
%1", capture.output(str(ds_flatten.split_list)))
  
  put_log("Saving a backup copy of a split dataset...")
  
  saveRDS(ds_flatten.split_list, 
          ds.flatten.split.file)
  
  rm(ds_flatten.split_list)
  
  put_log("The backup copy of a split dataset has been saved in the following file:
%1", ds.flatten.split.file)
} else {
  put_log("A Training Split Flattened Dataset with 20% Test Ratio has already been created
and saved tp following file:
%1", ds.flatten.split.file)
}

## Split the Dataset into Sets with 90% Test Ratio (10% for Training Set) ------
if(!file.exists(ds.flatten.0.1split.file)) {
  put_log("Splitting the Flattened Training Dataset into a Train and Test Sets 
with 90% Test Ratio (10% for Training Set)...")
  
  set.seed(N.classes)
  ds_flatten.0.1split_list <- ds.flatten.sample(ds.train.flatten$img.mx, 
                                                        ds.train.flatten$img.file,
                                                        test.ratio = 0.9,
                                                        seed = ds.flatten.seed)
  
  put_log("The Flattened Training Dataset has been split into a Train and Test Sets 
with 90% Test Ration (10% for Training Set):
%1", capture.output(str(ds_flatten.0.1split_list)))
  
  put_log("Saving a backup copy of a split dataset (10% for Training Set)...")
  
  saveRDS(ds_flatten.0.1split_list, 
          file = ds.flatten.0.1split.file)
  
  rm(ds_flatten.0.1split_list)
  
  put_log("The backup copy of a split dataset (10% for Training Set) has been saved in the following file:
%1", ds.flatten.0.1split.file)
  
} else {
  put_log("A Training Split Flattened Dataset with 90% Test Ratio has already been created
and saved tp following file:
%1", ds.flatten.0.1split.file)
}

rm(ds.train.flatten)


