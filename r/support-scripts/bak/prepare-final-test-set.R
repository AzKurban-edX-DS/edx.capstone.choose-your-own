#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preparing Final Test Datasets 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

