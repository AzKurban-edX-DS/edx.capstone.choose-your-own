#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Image Data Preparing Script: Creating List of Trimmed Image Objects 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Processing Data --------------------------------------------------------------
start <- put_start_date()

put_log("Preparing a List of the Trimmed Image Data objects...")

if (!file.exists(img_dat.list.file)) {
  put_log("Creating an Image list from the raw data files 
stored in the following root directory: %1,
Please wait...", root_dir)
  img_dat.list <- load_image.list(root_dir,
                                        load.img_trimmed.dat)
  
  put_log("The Trimmed Image list has been created with the following structure:
%1", capture.output(str(img_dat.list)))
  put_end_date(start)
  
  put_log("Saving Trimmed Image list to the backup file...")
  saveRDS(img_dat.list,
          file = img_dat.list.file)
  put_log("Trimmed Image list has been saved to the following file:
%1", img_dat.list.file)

  rm(img_dat.list)  
  gc()
  
} else {
  put_log("The list of the Trimmed Image objects has already been constructed 
and backed up to the following file:
%1", img_dat.list.file)
  
}

rm(root_dir, img_dat.list.file)
