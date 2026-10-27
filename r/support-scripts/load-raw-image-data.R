#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Image Data Preparing Script: Creating List of Image Objects 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Processing Data --------------------------------------------------------------
start <- put_start_date()

put_log("Preparing a List of the Image Data objects...")

if (!file.exists(raw_img.dat.file)) {
  put_log("Creating an Image list from the raw data files 
stored in the following root directory: %1,
Please wait...", root_dir)
  # raw_img.dat.ls <- load_image.list(root_dir,
  #                                 load.img_trimmed.dat)
  raw_img.dat.ls <- load_image.list(root_dir,
                                  function(file_path) {
                                    #' Load `magick` image, extracts, and returns 
                                    #' the underlying raw pixel buffer data.
                                    image_data(image_read(file_path))
                                  })
  
  ## Logging the summary & structure of the Image List -----------------    
  put_log("Summary of the Image list:
`summary(raw_img.dat.ls)`
%1", capture.output(summary(raw_img.dat.ls)))
  put_end_date(start)
  
  
  put_log("Label List:
`raw_img.dat.ls$label.list`
%1", capture.output(raw_img.dat.ls$label.list))
  
  put_log("Summary of the nested image list: 
`summary(raw_img.dat.ls$img.list)`:
%1", capture.output(summary(raw_img.dat.ls$img.list)))
  
  put_log("Example of the summary of the nested image list, label `W`: 
`summary(raw_img.dat.ls$img.list$W)`:
%1", capture.output(summary(raw_img.dat.ls$img.list$W)))
  
  put_log("Example of the structure of the first 3 items of the image list of the label `W`: 
`summary(raw_img.dat.ls$img.list$Wmg.list[1:3])`:
%1", capture.output(str(raw_img.dat.ls$img.list$W$img.list[1:3])))

  ## Saving the Image Data List Object -----------------------------------------  
  put_log("Saving Image list to the backup file...")
  saveRDS(raw_img.dat.ls,
          file = raw_img.dat.file)
  put_log("Image list has been saved to the following file:
%1", raw_img.dat.file)

  rm(raw_img.dat.ls)
  gc()
  
} else {
  put_log("The list of the Image objects has already been constructed 
and backed up to the following file:
%1", raw_img.dat.file)
  
}

