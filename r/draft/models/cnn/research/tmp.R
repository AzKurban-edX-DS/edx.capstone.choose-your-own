binClass.sample_train_test_sets.x3d <- function(x,
                                       x.files = NULL,
                                       seed = NA, 
                                       test.ratio = 0.2,
                                       shuffle_rows = TRUE,
                                       balanced = TRUE,
                                       bootstap.sample = FALSE) { 
  dim.x <- dim(x)
  dim.x
  stopifnot(length(dim.x) == 3)
  
  sample.idx <- binClass.get_sample.idx(x,
                                        class.label,
                                        seed, 
                                        test.ratio,
                                        bootstap.sample)
  
  if(!is.null(x.files)) {
    train.files <- x.files[sample.idx$train.index]
    test.files <- x.files[sample.idx$test.index]
  } else {
    train.files <- NULL
    test.files <- NULL
  }
  
  train.size <- (1 - test.ratio) * 100
  test.size <- test.ratio * 100
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
Generating a training sample of size %1% from the original dataset...",
          train.size)
  
  train.set <- list(x.train = x[sample.idx$train.index,,],
                    x.files = train.files)
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
A training sample of size %1% has been made with the following structure:
%2", train.size, capture.output(str(train.set)))
  
  put_log("Function: `sample_train_test_sets.mx`: 
Generating a testing sample of size %1% from the original dataset...",
          test.size)
  
  test.set <- list(x.test = x[sample.idx$test.index,,],
                   x.files = test.files)
  
  put_log("Function: `binClass.sample_train_test_sets.x3d`: 
A testing sample of size %1% has been made with the following structure:
%2", test.size, capture.output(str(test.set)))
  
  
  list(train_set = train.set,
       test_set = test.set)
}

