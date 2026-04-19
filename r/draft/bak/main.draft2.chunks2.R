library(data.table)

str(my_emnist.train)
data <- my_emnist.train
seed <- 1

# sample_train_test_sets.mx <- function(data, seed, test.ratio = 0.2){ -----
  put_log("Function: `sample_train_test_sets.mx`: Sampling 20% of the `data` data...")
  row_indices <- seq_len(nrow(data))
  row_names <- rownames(data)
  #names(row_indices) <- row_names
  
  char_group.IDs <- as.factor(row_names)
  str(char_group.IDs)
  

  char_group.list <- split(row_indices, char_group.IDs)
  str(char_group.list)
  
  set.seed(seed)
  
  test.idx <-
    sapply(char_group.list,
           function(group) 
             sample(group, 
                    ceiling(length(group)*test.ratio))) |>
  unlist(use.names = FALSE) |>
  sort()
    
  str(test.idx)
  
  
  # df.row_index = data.frame(index = row_indices,
  #                           char = char_group.IDs)
  # str(df.row_index)
  # head(df.row_index)
  
  
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 80% of the original index set of `data` not used for the test Set.")

  # train.idx <- row_indices[-test.idx]
  # str(train.idx)
  
  # train_set <- data[-test_ind,]
  # df.train.rows<- df.row_index[-test.idx,]
  # str(df.train.rows)
  
  train.set <- data[-test.idx,]
  str(train.set)
  dim(train.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: train_set")
  # put(summary(train_set))
  
#   put_log("Function: `sample_train_test_sets.mx`: 
# To make sure we don’t include grooups the Training Set that should not be there, 
# we exclude entries using the semi_join function from the test Set.")
#   
  # test.rows.tmp <- df.row_index[test.idx,]
  # str(test.rows.tmp)
  # 
  # df.test.rows <- test.rows.tmp |>
  #   semi_join(df.train.rows, by = "char")
  # str(df.test.rows)
  
  
  
  
  
#   put_log1("Function: `sample_train_test_sets.mx`:
# Extracting %1% of the original `data` not used for the Validation Set,
# excluding data for users who provided no more than a specified number of ratings: {min_nratings}.",
#            (1 - test.ratio)*100)
#   
#   #  test_set <- data[test_ind,]
#   tmp.data <- data[test_ind,]
#   
#   test_set <- tmp.data |> 
#     semi_join(train_set, by = "movieId") |> 
#     semi_join(train_set, by = "userId") |>
#     as.data.frame()
#   
#   # Add rows excluded from `test_set` into `train_set`
#   tmp.excluded <- anti_join(tmp.data, test_set)
#   train_set <- rbind(train_set, tmp.excluded)
  
  put_log("Function: `sample_train_test_sets.mx`: 
Extracting 20% of the original index set of `data` used for the test Set.")
  
  test.set <- data[test.idx,]
  str(test.set)
  dim(test.set)
  
  put_log("Function: `sample_train_test_sets.mx`: Dataset created: test_set")
  put(summary(test_set))
  
  # Return result datassets
  list(train_set = train_set,
       test_set = test_set)
# } ---------
  ### Open log: Load Dataset -----------------------------------------------------
  open_logfile(".load-dataset")
  
#  hwChar_data.load <- function(root_path, 
#                               folder.list = NULL, 
#                               char_files.max = NA){ --------------------------
    
  root_path <- img.train.root_path
  root_path
  char_files.max
  folder.list <- NULL
  folder.list
  
  start <- put_start_date()
    put_log("Getting file path lists...")
    img.file_list <- img.file_path.get_list(root_path, 
                                            folder.list,
                                            char_files.max)
    put_end_date(start)
    put_log("File path lists have been created")
    put(str(img.file_list))
    
    start <- put_start_date()
    put_log("Loading image files...")
    img_list <- lapply(img.file_list, function(img_f){
      list(cimg.list = map_il(img_f$file_path.list, image_load.cimg),
           fpath.list = img_f$file_path.list)
    })
    put_end_date(start)
    put_log("Image files have been loaded.")
    put(str(img_list))
    
    start <- put_start_date()
    put_log("Converting image lists to matrices...")
    char_matrix.list <- lapply(names(img_list), function(label){
      img_list[[label]]$cimg.list |> 
        as.matrix.cimg(label)
    })
    put_end_date(start)
    put_log("Image matrix list has been created.")
    put(str(char_matrix.list))
    
    start <- put_start_date()
    put_log("Combining image data to single matrix...")
    img.mx <- do.call(rbind, char_matrix.list)
    put_end_date(start)
    put_log("Image dataset matrix has been created.")
    put(dim(img.mx))
    
    label_list <- as.factor(names(img.file_list))
    # str(label_list)
    
    
train.dat.subset64 <- list(img.files = img.file_list,
                           label.list = label_list,
                           img.list = char_matrix.list,
                           my_emnist = img.mx)
#  } ------------------------
    
rm(root_path)
rm(folder.list)
rm(char_files.max)
    
    
    
  
  
  
