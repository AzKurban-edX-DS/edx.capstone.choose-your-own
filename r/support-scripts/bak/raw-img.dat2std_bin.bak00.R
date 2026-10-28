## Transforming into a List of Binary standardized Image Data Objects ----------

open_logfile(".prepare,std-bin.mx-dat")
start <- put_start_date()


put_log("Preparing a List of Binary standardized Image Data Objects...")

put_log("The path for the backup file to save the Binary standardized Image Object list:
%1", img.std_bin.mx.ls.file)

if (!file.exists(img.std_bin.mx.ls.file)) {
  stopifnot(file.exists(raw_img.dat.file))
  
  put_log("Loading list of Image objects...")
  raw_img.dat.ls <- readRDS(raw_img.dat.file)
  put_log("The list of Image objects has been loading from the following file:
%1", raw_img.dat.file)
  
  ### Define the Standard Image Size Geometry ----------------------------------
  
  if(is.null(ds.img_min.geometry)) {
    # Find the smallest existing image in the dataset. 
    x <- lapply(raw_img.dat.ls$img.list, function(ch.class) {
      img_dat.ls <- ch.class$img.list
      x <- lapply(img_dat.ls, function(img_dat) {
        dim(img_dat[1,,])
      })
    }) |> unlist() |>
      matrix(ncol = 2, 
             byrow = T, 
             dimnames = list(NULL, 
                             c('width', 'height')))
    
    put_log("Structure of the Image Size Array:
%1", capture.output(str(x)))
    
    y <- as.data.frame(x)
    
    y.grouped <- y |> group_by(width) |>
      summarise(min.height = min(height))
    
    print(y.grouped)
    
    y.size <- y.grouped$width * y.grouped$min.height
    img.min_size <- y.grouped[which.min(y.size),]
    
    # Define the geometry based on the smallest image found in the dataset.
    ds.img_min.geometry  <- paste0(img.min_size$width,
                                   'x',
                                   img.min_size$min.height,
                                   '!')[1]
    rm(x,y)
    
  }
  
  
  put_log("The geometry for standardizing images to a uniform size, 
based on the smallest existing image in the Training Set, is as follows: 
%1", ds.img_min.geometry)
  
  ### Standardize the Image Data -------------------------------------------------    
  
  put_log("Creating a Standardized Image Binary Data list from the raw pixel buffer data...")
  
  start <- put_start_date()
  
  img.std_bin.mx.ls <- 
    ds.standardize.img_dat(raw_img.dat.ls,
                           function(image.dat) {
                             image.dat2bin_mx(image.dat,
                                              ds.img_min.geometry, # '28x28!'
                                              trim_image = TRUE)
                           })
  
  put_log("The Standardized Image Binary Data list has been created 
with the following structure:
%1", capture.output(str(img.std_bin.mx.ls)))
  put_end_date(start)
  
  put_log("Saving Standardized Image Binary Data list to the backup file...")
  saveRDS(img.std_bin.mx.ls,
          file = img.std_bin.mx.ls.file)
  put_log("Standardized Image Binary Data list has been saved to the following file:
%1", img.std_bin.mx.ls.file)
  put_end_date(start)
  
} else {
  put_log("The Standardized Image Binary Data list has already been constructed 
and backed up to the following file:
%1", img.std_bin.mx.ls.file)
}

log_close()
# =========================================================================
# Log End Time: 2026-10-26 02:52:02.624323
# Log Elapsed Time: 0 00:21:09
# =========================================================================

