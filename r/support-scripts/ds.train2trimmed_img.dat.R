#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training Data Preparing Script: Creating List of Trimmed Image Objects 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Processing Data --------------------------------------------------------------
open_logfile(".ds.train2trimmed_img.dat")
start <- put_start_date()

put_log("Preparing a List of the Trimmed Image Data objects...")

if (!file.exists(train.img_dat.list.file)) {
  put_log("Creating an Image list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train_root.dir)
  train.img_dat.list <- load_image.list(img.train_root.dir,
                                            load.img_trimmed)
  
  put_log("The Trimmed Image list has been created with the following structure:
%1", capture.output(str(train.img_dat.list)))
  put_end_date(start)
  
  put_log("Saving Trimmed Image list to the backup file...")
  saveRDS(train.img_dat.list,
          file = train.img_dat.list.file)
  put_log("Trimmed Image list has been saved to the following file:
%1", train.img_dat.list.file)

  rm(train.img_dat.list)  
  gc()
  
} else {
  put_log("The list of the Trained Trimmed Image objects has already been constructed 
and backed up to the following file:
%1", train.img_dat.list.file)
  
}

log_close()
# =========================================================================
# Log End Time: 2026-10-07 05:55:29.738928
# Log Elapsed Time: 0 00:53:48
# =========================================================================


