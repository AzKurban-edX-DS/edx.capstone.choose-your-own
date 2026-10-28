#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Standardize & Binarize Image Data  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start <- put_start_date()


put_log("Preparing a List of Standardized Binary Image Data Objects...")

put_log("The path for the backup file to save the Standardized Binary Image Object list:
%1", img.std_bin.ls.nested.file)

## Initialize Class Labels Global Variable ------------------------------------

raw_img.dat.ls <- NULL

if(is.null(Y.Labels) ||
   !file.exists(img.std_bin.ls.nested.file)) {
  
  stopifnot(file.exists(raw_img.dat.file))
  
  put_log("Loading list of Image objects...")
  raw_img.dat.ls <- readRDS(raw_img.dat.file)
  put_log("The list of Image objects has been loading from the following file:
%1", raw_img.dat.file)
  
  Y.Labels <- raw_img.dat.ls$label.list
}

## Define the Standard Image Size Geometry -------------------------------------

if(is.null(ds.img_min.geometry)) {
  stopifnot(!is.null(raw_img.dat.ls))  
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
  
  std_img.side <- min(img.min_size$width, img.min_size$min.height)
  
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

## Transform into a Nested List of Standardized Binary Image Data Objects ------

if (!file.exists(img.std_bin.ls.nested.file)) {
stopifnot(!is.null(raw_img.dat.ls))  

  # Standardize the Image Data -----------------------------------------------    
  
  put_log("Creating a Standardized Image Binary Data list from the raw pixel buffer data...")
  
  start <- put_start_date()
  
  img.std_bin.ls.nested <- 
    ds.standardize.img_dat(raw_img.dat.ls,
                           function(image.dat) {
                             image.dat2bin_mx(image.dat,
                                              ds.img_min.geometry, # '28x28!'
                                              trim_image = TRUE)
                           })
  rm(raw_img.dat.ls)
  gc()
  
  # Logging the summary & structure of the Image Data Array Nested List -------    
  put_log("Summary of the Image list:
`summary(img.std_bin.ls.nested)`
%1", capture.output(summary(img.std_bin.ls.nested)))
  put_end_date(start)
  
  
  put_log("Label List:
`img.std_bin.ls.nested$label.list`
%1", capture.output(img.std_bin.ls.nested$label.list))
  
  put_log("Summary of the nested image list: 
`summary(img.std_bin.ls.nested$img.list)`:
%1", capture.output(summary(img.std_bin.ls.nested$img.list)))
  
  put_log("Example of the summary of the nested image list, label `W`: 
`summary(img.std_bin.ls.nested$img.list$W)`:
%1", capture.output(summary(img.std_bin.ls.nested$img.list$W)))
  
  put_log("Example of the structure of the first 3 items of the image list of the label `W`: 
`summary(img.std_bin.ls.nested$img.list$Wmg.list[1:3])`:
%1", capture.output(str(img.std_bin.ls.nested$img.list$W$img.list[1:3])))
  
  # Saving the Image Data Array Nested List Object ----------------------------  
  
  put_log("Saving Standardized Image Binary Data list to the backup file...")
  saveRDS(img.std_bin.ls.nested,
          file = img.std_bin.ls.nested.file)
  put_log("Standardized Image Binary Data list has been saved to the following file:
%1", img.std_bin.ls.nested.file)
  put_end_date(start)
  
  rm(img.std_bin.ls.nested)
  gc()
  
} else {
  put_log("The Standardized Image Binary Data list has already been constructed 
and backed up to the following file:
%1", img.std_bin.ls.nested.file)
}

# =========================================================================
# Log End Time: 2026-10-26 02:52:02.624323
# Log Elapsed Time: 0 00:21:09
# =========================================================================

### Preparing a List of Standardized Image Binary Data Matrices ----------------

put_log("Preparing a List of the Standardized Binary Image  Matrix...")

put_log("The path for the backup file to save the Standardized Binary Image  Matrix object:
%1", img.std_bin.array.ls.file)

if(!file.exists(img.std_bin.array.ls.file)){
  stopifnot(file.exists(img.std_bin.ls.nested.file))
  
  put_log("Loading Standardized Image Binary Data list...")
  img.std_bin.ls.nested <- readRDS(img.std_bin.ls.nested.file)
  put_log("The Standardized Image Binary Data list has been loaded from the following file:
%1", img.std_bin.ls.nested.file)
  
  put_log("Combining nested list of images to a list of arrays...")
  
  img.std_bin.array.ls <- lapply(names(img.std_bin.ls.nested$img.list), 
                                function(label){
    put_log("Creating array of images of character `%1`", label)
    item <- img.std_bin.ls.nested$img.list[[label]]
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
  names(img.std_bin.array.ls) <- as.character(Y.Labels)
  rm(img.std_bin.ls.nested)
  
  put_log("A Standardized Image Binary Data Matrix list has been created with the following structure:
  %1", capture.output(str(img.std_bin.array.ls)))
  put_end_date(start)
  
  ## Saving the Image Data Array List Object -----------------------------------  
  
  
  put_log("Saving Standardized Image Binary Data Matrix list to the backup file...")
  saveRDS(img.std_bin.array.ls,
          file = img.std_bin.array.ls.file)
  
  rm(img.std_bin.array.ls)
  
  put_log("Standardized Image Binary Data Matrix list has been saved to the following file:
%1", img.std_bin.array.ls.file)
  
} else {
  put_log("The Standardized Image Binary Data Matrix list has already been constructed 
and backed up to the following file:
%1", img.std_bin.array.ls.file)
}

### Preparing an Array of the Standardized Binary Image  Matrices --------------


put_log("Preparing an Array of the Standardized Binary Image  Matrices...")

put_log("The path for the backup file to save the Array of the Binary Image  Matrix:
%1", img.std_bin.array.file)

if(!file.exists(img.std_bin.array.file)){
  stopifnot(file.exists(img.std_bin.array.ls.file))
  
  put_log("Loading the Standardized Binary Data Matrix list...")
  img.std_bin.array.ls <- readRDS(img.std_bin.array.ls.file)
  
  put_log("The Standardized Image Binary Data Matrix list has been loaded from the following file:
%1", img.std_bin.array.ls.file)
  
  put_log("Combining Binary image matrix list to array...")
  
  img.array.ls <- lapply(img.std_bin.array.ls, function(item) {
    item$img.array
  })
  img.file.ls <- lapply(img.std_bin.array.ls, function(item) {
    item$file.path
  })
  
  img_dat.array <- abind(img.array.ls, along = 1)
  rm(img.array.ls)
  
  img.file_path.array <- abind(img.file.ls)
  rm(img.file.ls)
  
  names(img.file_path.array) <- rownames(img_dat.array)
  
  rm(img.std_bin.array.ls)
  
  stopifnot(length(img.file_path.array) == nrow(img_dat.array))
  
  put_log("images File Path list has the following structure:
  %1", capture.output(str(img.file_path.array)))
  
  put_log("Combined Binary image matrix array has the following structure:
  %1", capture.output(str(img_dat.array)))
  
  put_log("Combined Binary image matrix array has the following dimentions:
  %1", capture.output(dim(img_dat.array)))
  
  put_log("Saving Standardized Image Binary Data array set to the backup file...")
  saveRDS(list(img_dat.array = img_dat.array,
               file_path.array = img.file_path.array),
          file = img.std_bin.array.file)
  
  put_log("The Standardized Image Binary Data array set has been saved to the following file:
%1", img.std_bin.array.file)
  
  rm(img_dat.array,
     img.file_path.array)
  
  put_end_date(start)
} else {
  put_log("The Standardized Image Binary Data array has already been constructed 
and backed up to the following file:
%1", img.std_bin.array.file)
}

if (exists("img.std_bin.array.ls")) rm(img.std_bin.array.ls)

