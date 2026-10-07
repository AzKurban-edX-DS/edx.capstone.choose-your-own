#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Download the Kaggle Dataset 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters

kaggle_dataset <- "vaibhao/handwritten-characters"

open_logfile(".download-kaggle-dataset")
start.download <- put_start_date()

if(!dir.exists(raw_data.chars.dir)) {
  put_log("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.dir, unzip = TRUE)
  put_log("The Kaggle image files have been downloaded 
and unziped to the following directory: 
`%1`", raw_data.chars.dir)
} else {
  warning(str.build("The Kaggle image files have already been downloaded 
and saved to the following directory: 
`%1`.
If you need to rerun the download, delete the root folder and rerun this script.", 
                    raw_data.chars.dir))
}

# Remove duplicate files:
dir.to_remove <- file.path(raw_data.chars.dir, "dataset")
dir.to_remove

if (dir.exists(dir.to_remove)) {
  put_log("Deleting the folder with duplicate files: `%1`...", dir.to_remove)
  unlink(dir.to_remove, recursive = TRUE, force = TRUE)
  put_log("Directory removed: `%1`", dir.to_remove)
} else {
  warning(str.build("Couldn't delete the folder:
`%1`  
It has already been deleted or moved.", 
                    dir.to_remove))
}

put_end_date(start.download)
log_close()
# =========================================================================
# Log End Time: 2026-10-07 03:39:40.263557
# Log Elapsed Time: 0 00:54:11
# =========================================================================

