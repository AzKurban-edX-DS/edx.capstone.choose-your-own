#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Final Test Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Transforming into a List of Standardized Binary Image Data Objects ----------

open_logfile(".prepare-bin.train-dat")
start <- put_start_date()


put_log("Preparing a List of Standardized Binary Image Data Objects...")

ft.img_bin.mx_list.file <- file.path(train.data.dir, 
                                        "final-test.img_bin.mx_list.rds")

put_log("The path for the backup file to save the Standardized Binary Image Object list:
%1", ft.img_bin.mx_list.file)

if (!file.exists(ft.img_bin.mx_list.file)) {
  stopifnot(file.exists(final_test.raw_img.dat.file))
  
  put_log("Loading list of Image objects...")
  raw_img.dat.ls <- readRDS(final_test.raw_img.dat.file)
  put_log("The list of Image objects has been loading from the following file:
%1", final_test.raw_img.dat.file)
  
  ### Standardize the Image Data -------------------------------------------------    
  
  put_log("Creating a Standardized Image Binary Data list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train_root.dir)
  
  start <- put_start_date()
  
  ft.img_bin.mx_list <- 
    ds.standardize.img_dat(raw_img.dat.ls,
                           function(image.dat) {
                             image.dat2bin_mx(image.dat,
                                              ds.img_min.geometry) # '28x28!'
                           })
  
  put_log("The Standardized Image Binary Data list has been created 
with the following structure:
%1", capture.output(str(ft.img_bin.mx_list)))
  put_end_date(start)
  
  put_log("Saving Standardized Image Binary Data list to the backup file...")
  saveRDS(ft.img_bin.mx_list,
          file = ft.img_bin.mx_list.file)
  put_log("Standardized Image Binary Data list has been saved to the following file:
%1", ft.img_bin.mx_list.file)
  put_end_date(start)
  
} else {
  put_log("The Standardized Image Binary Data list has already been constructed 
and backed up to the following file:
%1", ft.img_bin.mx_list.file)
}

log_close()

## Pre-process the Final Test Image Data ---------------------------------------
### Preparing a List of Standardized Binary Image  Matrix ----------------------

final_test.img_bin.mx_list.file <- file.path(final_test.data.dir, 
                                                  "final_test.img_bin.mx_list.rds")

put_log("The path for the backup file to save the Final Test
List of Standardized Binary Image Objects:
%1", final_test.img_bin.mx_list.file)

final_test.img.std_bin.array.file <- file.path(final_test.data.dir, 
                                                   "final_test.img.std_bin.array..rds")

put_log("The path for the backup file to save the Final Test 
Array of the Standardized Binary Image  Matrix:
%1", final_test.img.std_bin.array.file)


img.std_bin.array.ls.file <- file.path(final_test.data.dir, 
                                  "final_test.img.std_bin.array.ls.rds")

put_log("The path for the backup file to save the Final Test 
List of Standardized Binary Image  Matrix:
%1", img.std_bin.array.ls.file)

if(!file.exists(img.std_bin.array.ls.file)){
  if(!exists("ft.img_bin.mx_list")) {
    stopifnot(file.exists(final_test.img_bin.mx_list.file))
    ft.img_bin.mx_list <- readRDS(final_test.img_bin.mx_list.file)
  }
  
  put_log("Combining nested list of images to list of arrays...")
  
  img.nested_list <- ft.img_bin.mx_list$img.list
  rm(ft.img_bin.mx_list)
  
  length(img.nested_list)
  names(img.nested_list)
  
  
  ft.img.std_bin.array.ls <- lapply(names(img.nested_list), function(label){
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
  
  names(ft.img.std_bin.array.ls) <- as.character(Y.Labels)
  
  put_log("The Final Test Standardized Image Binary Data Matrix list has been created with the following structure:
  %1", capture.output(str(ft.img.std_bin.array.ls)))
  put_end_date(start)
  
  put_log("Saving Standardized Image Binary Data Matrix list to the backup file...")
  saveRDS(ft.img.std_bin.array.ls,
          file = img.std_bin.array.ls.file)
  
  put_log("Standardized Image Binary Data Matrix list has been saved to the following file:
%1", img.std_bin.array.ls.file)
  put_end_date(start)
} else {
  put_log("The Standardized Image Binary Data Matrix list has already been constructed 
and backed up to the following file:
%1", img.std_bin.array.ls.file)
}

## Prepare a Dataset for the Final Test of the Best Model ----------------------

if(!file.exists(final_test.img.std_bin.array.file)){
  if (!exists("ft.img.std_bin.array.ls")) {
    stopifnot(file.exists(img.std_bin.array.ls.file))
    
    put_log("Loading the Final Standardized Image Binary Data Matrix list...")
    ft.img.std_bin.array.ls <- readRDS(img.std_bin.array.ls.file)
    
    put_log("The Final Test Standardized Image Binary Data Matrix list has been loaded from the following file:
%1", img.std_bin.array.ls.file)
  }
  
  put_log("Combining Binary image 28x28x matrix list to array...")
  
  array.ls <- lapply(ft.img.std_bin.array.ls, function(item) {
    item$img.array
  })
  file.ls <- lapply(ft.img.std_bin.array.ls, function(item) {
    item$file.path
  })
  
  img_dat.array <- abind(array.ls, along = 1)
  rm(array.ls)
  
  img.file_path.array <- abind(file.ls)
  rm(file.ls)
  
  names(img.file_path.array) <- rownames(img_dat.array)
  
  stopifnot(length(img.file_path.array) == nrow(img_dat.array))
  
  put_log("Combined Final Test Binary image matrix 28x28 array has the following structure:
  %1", capture.output(str(img_dat.array)))
  
  put_log("Combined Final Test Binary image matrix 28x28 array has the following dimentions:
  %1", capture.output(dim(img_dat.array)))
  
  put_log("Saving Final Test Standardized Image Binary Data array to the backup file...")
  saveRDS(list(img_dat.array = img_dat.array,
               img.file_path.array = img.file_path.array),
          file = final_test.img.std_bin.array.file)
  
  put_log("The Final Test Standardized Image Binary Data array has been saved to the following file:
%1", final_test.img.std_bin.array.file)
  
  rm(img_dat.array)
  rm(img.file_path.array)
} else {
  put_log("The Standardized Image Binary Data array has already been constructed 
and backed up to the following file:
%1", final_test.img.std_bin.array.file)
}

put_end_date(start)
log_close()
# =========================================================================
# Log End Time: 2026-10-14 01:36:32.252927
# Log Elapsed Time: 0 00:02:09
# =========================================================================

