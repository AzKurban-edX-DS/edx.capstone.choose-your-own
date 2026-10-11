#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training Data Preparing Script: Creating List of Trimmed Image Objects 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Processing Data --------------------------------------------------------------
open_logfile(".load-train-data")
start <- put_start_date()

put_log("Preparing a List of the Trimmed Image objects...")

train.trimmed_img.file <- file.path(train.data.dir, 
                                    "train.trimmed-img.list.rds")

put_log("The path for the backup file to save the Trimmed Image Object list:
%1", train.trimmed_img.file)

if (!file.exists(train.trimmed_img.file)) {
  put_log("Creating an Image list from the raw data files 
stored in the following root directory: %1,
Please wait...", img.train_root.dir)
  train.trimmed_img.list <- load_image.list(img.train_root.dir,
                                            load.img_trimmed)
  
  put_log("The Trimmed Image list has been created with the following structure:
%1", capture.output(str(train.trimmed_img.list)))
  put_end_date(start)
  
  put_log("Saving Trimmed Image list to the backup file...")
  saveRDS(train.trimmed_img.list,
          file = train.trimmed_img.file)
  put_log("Trimmed Image list has been saved to the following file:
%1", train.trimmed_img.file)

  rm(train.trimmed_img.list)  
  gc()

} else {
  put_log("The list of the Trained Trimmed Image objects has already been constructed 
and backed up to the following file:
%1", train.trimmed_img.file)
  
}

log_close()
# =========================================================================
# Log End Time: 2026-10-07 05:55:29.738928
# Log Elapsed Time: 0 00:53:48
# =========================================================================


