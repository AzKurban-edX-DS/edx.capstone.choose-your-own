#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Image Data Preparing Script: Creating List of Image Objects 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Processing Data --------------------------------------------------------------
start <- put_start_date()

put_log("Preparing a List of the Image Data objects...")

if (!file.exists(img_dat.list.file)) {
  put_log("Creating an Image list from the raw data files 
stored in the following root directory: %1,
Please wait...", root_dir)
  # img_dat.list <- load_image.list(root_dir,
  #                                 load.img_trimmed.dat)
  img_dat.list <- load_image.list(root_dir,
                                  function(file_path) {
                                    #' Load `magick` image, extracts, and returns 
                                    #' the underlying raw pixel buffer data.
                                    image_data(image_read(file_path))
                                  })
  
  ## Logging the summary & structure of the Image List -----------------    
  put_log("Summary of the Image list:
`summary(img_dat.list)`
%1", capture.output(summary(img_dat.list)))
  put_end_date(start)
  
  
  put_log("Label List:
`img_dat.list$label.list`
%1", capture.output(img_dat.list$label.list))
  
  put_log("Summary of the nested image list: 
`summary(img_dat.list$img.list)`:
%1", capture.output(summary(img_dat.list$img.list)))
  
  put_log("Example of the summary of the nested image list, label `W`: 
`summary(img_dat.list$img.list$W)`:
%1", capture.output(summary(img_dat.list$img.list$W)))
  
  put_log("Example of the structure of the first 3 items of the image list of the label `W`: 
`summary(img_dat.list$img.list$Wmg.list[1:3])`:
%1", capture.output(str(img_dat.list$img.list$W$img.list[1:3])))

  ## Saving the Image List Object --------------------------------------  
  put_log("Saving Image list to the backup file...")
  saveRDS(img_dat.list,
          file = img_dat.list.file)
  put_log("Image list has been saved to the following file:
%1", img_dat.list.file)

  rm(img_dat.list)  
  gc()
  
} else {
  put_log("The list of the Image objects has already been constructed 
and backed up to the following file:
%1", img_dat.list.file)
  
}

rm(root_dir, img_dat.list.file)
