#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Final Test Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

open_logfile(".prepare-final_test-data")
start <- put_start_date()

## Pre-process the Final Test Image Data ---------------------------------------

final_test.img28x28bin.list.file_path <- file.path(final_test.data.dir, 
                                                   "final_test.img28x28bin.list.rds")

put_log("The path for the backup file to save the Final Test
List of Binary 28x28-size Image Objects:
%1", final_test.img28x28bin.list.file_path)

final_test.img28x28mx.list.file_path <- file.path(final_test.data.dir, 
                                                  "final_test.img28x28mx.list.rds")

put_log("The path for the backup file to save the Final Test 
List of Binary 28x28-size Image  Matrix:
%1", final_test.img28x28mx.list.file_path)

final_test.img28x28mx.array.file_path <- file.path(final_test.data.dir, 
                                                   "final_test.img28x28mx.array.rds")

put_log("The path for the backup file to save the Final Test 
Array of the Binary 28x28-size Image  Matrix:
%1", final_test.img28x28mx.array.file_path)

if (!file.exists(final_test.img28x28bin.list.file_path)) {
  put_log("Creating the Final Test Binary Image 28x28 list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.validation_root.dir)
  #label_folder.list <- c("0","1","2","3","7", "A", "B", "C", "D") 
  ft.img28x28bin.list <- ds.load2bin28x28mx.list(img.validation_root.dir)
  put_end_date(start)
  
  put_log("The Final Test Binary Image 28x28 list has been created with the following structure:
%1", capture.output(str(ft.img28x28bin.list)))
  
  put_log("Saving the Final Test Binary Image 28x28 list...")
  
  saveRDS(ft.img28x28bin.list,
          file = final_test.img28x28bin.list.file_path)
  
  put_log("The Final Test Binary Image 28x28 list has been saved to the following file:
%1", final_test.img28x28bin.list.file_path)
} else {
  put_log("The Binary Image 28x28 list has already been constructed 
and backed up in the following file:
%1", final_test.img28x28bin.list.file_path)
}
put_end_date(start)

if(!file.exists(final_test.img28x28mx.list.file_path)){
  if(!exists("ft.img28x28bin.list")) {
    stopifnot(file.exists(final_test.img28x28bin.list.file_path))
    ft.img28x28bin.list <- readRDS(final_test.img28x28bin.list.file_path)
  }
  
  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- ft.img28x28bin.list$img.list
  rm(ft.img28x28bin.list)
  
  length(img.nested_list)
  names(img.nested_list)
  
  
  ft.img28x28mx.list <- lapply(names(img.nested_list), function(label){
    item <- img.nested_list[[label]]
    fpath_list.size = length(item$fpath.list)
    
    stopifnot(length(item$img.list) == fpath_list.size)
    
    img.array <- abind(item$img.list, rev.along = 3)
    dimnames(img.array) <- list(base::rep(label, 
                                          times = length(item$img.list)),
                                NULL,
                                NULL)
    
    stopifnot(dim(img.array)[1] == fpath_list.size)
    put_log("The array of images of character `%1` has been created.", label)
    
    list(img.array = img.array,
         file.path = item$fpath.list)
  })
  rm(img.nested_list)
  
  names(ft.img28x28mx.list) <- as.character(Y.Labels)
  
  put_log("The Final Test Binary Image 28x28 Matrix list has been created with the following structure:
  %1", capture.output(str(ft.img28x28mx.list)))
  put_end_date(start)
  
  put_log("Saving Binary Image 28x28 Matrix list to the backup file...")
  saveRDS(ft.img28x28mx.list,
          file = final_test.img28x28mx.list.file_path)
  
  put_log("Binary Image 28x28 Matrix list has been saved to the following file:
%1", final_test.img28x28mx.list.file_path)
  put_end_date(start)
} else {
  put_log("The Binary Image 28x28 Matrix list has already been constructed 
and backed up to the following file:
%1", final_test.img28x28mx.list.file_path)
}

## Prepare a Dataset for the Final Test of the Best Model ----------------------

if(!file.exists(final_test.img28x28mx.array.file_path)){
  if (!exists("ft.img28x28mx.list")) {
    stopifnot(file.exists(final_test.img28x28mx.list.file_path))
    
    put_log("Loading the Final Binary Image 28x28 Matrix list...")
    ft.img28x28mx.list <- readRDS(final_test.img28x28mx.list.file_path)
    
    put_log("The Final Test Binary Image 28x28 Matrix list has been loaded from the following file:
%1", final_test.img28x28mx.list.file_path)
  }
  
  put_log("Combining Binary image 28x28x matrix list to array...")
  
  img28x28mx.array.list <- lapply(ft.img28x28mx.list, function(item) {
    item$img.array
  })
  img28x28mx.file.list <- lapply(ft.img28x28mx.list, function(item) {
    item$file.path
  })
  
  img28x28mx.array <- abind(img28x28mx.array.list, along = 1)
  rm(img28x28mx.array.list)
  
  img28x28mx.fpath <- abind(img28x28mx.file.list)
  rm(img28x28mx.file.list)
  
  names(img28x28mx.fpath) <- rownames(img28x28mx.array)
  
  stopifnot(length(img28x28mx.fpath) == nrow(img28x28mx.array))
  
  put_log("Combined Final Test Binary image matrix 28x28 array has the following structure:
  %1", capture.output(str(img28x28mx.array)))
  
  put_log("Combined Final Test Binary image matrix 28x28 array has the following dimentions:
  %1", capture.output(dim(img28x28mx.array)))
  
  put_log("Saving Final Test Binary Image 28x28 array to the backup file...")
  saveRDS(list(img28x28mx.array = img28x28mx.array,
               img28x28mx.fpath = img28x28mx.fpath),
          file = final_test.img28x28mx.array.file_path)
  
  put_log("The Final Test Binary Image 28x28 array has been saved to the following file:
%1", final_test.img28x28mx.array.file_path)
  
  rm(img28x28mx.array)
  rm(img28x28mx.fpath)
} else {
  put_log("The Binary Image 28x28 array has already been constructed 
and backed up to the following file:
%1", final_test.img28x28mx.array.file_path)
}

put_end_date(start)
log_close()
# =========================================================================
# Log End Time: 2026-10-14 01:36:32.252927
# Log Elapsed Time: 0 00:02:09
# =========================================================================

