#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# `HyperModel` Subclass for CNN MCC  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> `KerasTuner` `HyperModel` subclass for Convolutional Neural Network 
#> Multiclass Classifier (BDL MCC)  Model Tuning

# References:
# [1] R interface to Keras Tuner
# https://eagerai.github.io/kerastuneR/#r-interface-to-keras-tuner [1]

# Basic CNN Architecture: The 5 Key Layers Simplified for 2026
# https://amityonline.com/blog/basic-cnn-architecture

# [2] Hyperparameter tuning with Keras Tuner
# https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html

# [3] Easy Hyperparameter Tuning with Keras Tuner and TensorFlow
# https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/

# library(keras3)
# library(tensorflow)
# library(magrittr)
# library(kerastuneR)

CNN_MCC.HyperModel <- reticulate::PyClass(
  'HyperModel',
  inherit = kerastuneR::HyperModel_class(),
  list(
    
    `__init__` = function(self, 
                          num_classes,
                          start_date = NULL) {
      
      self$num_classes = num_classes
      self$start_date = ifelse(is.null(start_date), 
                               Sys.time(), 
                               start_date)
      self$error = NULL
      
      NULL
    },
    
    build = function(self, hp) { # [2]
      
      if(is.null(self$start_date)) {
        start <- put_start_date()
      } else {
        start <- self$start_date
      }
      
      
      input_layer <- layer_input(shape = shape(28L, 28L, 1L))
      layer <- input_layer
      
      # conv_padding <- hp$get('conv_padding') # same | valid
      
      for (i in 1:hp$get('conv_blocks')) {
        
        # conv_filters <- hp$get(paste0('conv', i, '_filters'))
        # kernel_size <- hp$get(paste0('conv', i,'_kernel.size'))  
        
        conv_filters <- hp$Int(paste0('conv', i, '_filters'),
                               min_value = 32L,
                               max_value = 256L,
                               step = 32L)
        
        kernel_size <- hp$Choice(paste0('conv', i,'_kernel.size'), 
                                 c(2L, 3L, 4L, 5L))

        add_block.result <- try(
          {
            layer <- layer |>
              layer_conv_2d(filters = conv_filters,
                            kernel_size = kernel_size,
                            padding = 'same',
                            # strides = list(1L, 1L),
                            activation = "relu") |>
              layer_max_pooling_2d()
          }, silent = TRUE)

        if("try-error" %in% class(add_block.result)) {
          put_log("Failed to add Convolution Block %1", i)
          put_log(add_block.result)

          put_log("The Convolution Block %1 HAS NOT BEEN ADDED to the CNN MCC Model.
Building the model with %2 Conv Blocks", i, i -1)
          
          self$conv_blocks.max <- (i - 1)
          self$error <- add_block.result
          
          break
        }
        
        rm(add_block.result)
        
        put_log("The Convolution Block has been added with the following parameters
Block #: %1, filters number: %2, kernel size: %3
has been added to the CNN MCC Model.",
                i,
                conv_filters,
                kernel_size)
      }
      
      dense_units <- hp$get('dense_units')
      dropout1 <- hp$get('dropout1')
      dropout2 <- hp$get('dropout2')
      

      layer <- layer |>
        layer_dropout(dropout1) |>
        layer_flatten() |>
        layer_dense(dense_units,
                    activation = 'relu') |>
        layer_dropout(dropout2)

      put_log("Dense layers block added with the following parameters:
`dense_units`: %1, `dropout1 rate:`%2, `dropout2 rate` %3",
              dense_units,
              dropout1,
              dropout2)

      output_layer <- layer |>
        layer_dense(as.integer(self$num_classes),
                    activation = 'softmax')

      learning_rate <- hp$get('learning_rate')

      model <- keras_model(input_layer, output_layer) |>
        compile(
          optimizer = keras3::optimizer_adamax(learning_rate),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      put_log("A new model has been compiled with `learning_rate`: %1.", learning_rate)
      put_end_date(start)
      
      return(model)
    }
  )
)
