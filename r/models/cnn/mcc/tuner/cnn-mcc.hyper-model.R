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
                          start_time,
                          conv_blocks = 5,
                          learning_rate = NULL) {
      
      self$num_classes = num_classes
      self$start_time = start_time
      self$learning_rate = learning_rate
      self$conv_blocks = conv_blocks
      self$error = NULL
      
      NULL
    },
    
    build = function(self, hp) { # [2]
      input_layer <- layer_input(shape = shape(28L, 28L, 1L))
      layer <- input_layer

      # max_blocks <- ifelse(is.null(self$conv_blocks), 5, self$conv_blocks)
      
      put_log("Maximum number of Convolution Blocks: %1", self$conv_blocks)
      
      # kernel_size <- hp$Choice('kernel_size',
      #                 c(2L, 3L))
      
      kernel_size <- hp$Choice('kernel_size', 
                               values = c(3L, 5L))
      
      dense_units <- hp$Int('dense_units',
                            min_value = 128L,
                            max_value = 512L,
                            step = 64L,
                            default = 128L)
      
      drop1_rate <- hp$Float('dropout1',
                             min_value = 0.0,
                             max_value = 0.3,
                             step = 0.05,
                             default = 0.25)  
      
      drop2_rate <- hp$Float('dropout2',
                             min_value = 0.0,
                             max_value = 0.5,
                             step = 0.1,
                             default = 0.5)  
      
      ln_rate <- self$learning_rate
      
      if(is.null(ln_rate))
        ln_rate <- hp$Float('learning_rate',
                            min_value = 1e-4,
                            max_value = 1e-2,
                            sampling = "log")
      
      conv_filters.min_value <- 32L
      
      put_log("Tuning the model of %1 Convolution Block with the following hype-parameters: 
%2 min number of filters for the Convolution Layes,
Learning Rate: %3.
dropout layer 1 rate: %4,
dense layer units: %5,
dropout layer 2 rate: %6",
              self$conv_blocks,
              conv_filters.min_value,
              ln_rate,
              drop1_rate,
              dense_units,
              drop2_rate)

      for (i in 1:self$conv_blocks) {
        
        conv_filters <- hp$Int('conv_filter',
                               min_value = conv_filters.min_value,
                               max_value = 256L,
                               step = 32L)

        put_log("Adding the Conv Block %1 with %2 filters & kernel size = %3 
for the Conv_2d layer...", 
                i, 
                conv_filters,
                kernel_size)

        add_block.result <- try(
          {
            layer <- layer |>
              layer_conv_2d(filters = conv_filters,
                            kernel_size = kernel_size,
                            # padding = 'same',
                            # strides = list(1L, 1L),
                            activation = "relu") |>
              layer_max_pooling_2d(pool_size = c(2L, 2L))
          }, silent = TRUE)

        if("try-error" %in% class(add_block.result)) {
          put_log("Failed to add Convolution Block %1", i)
          put_log(add_block.result)

          put_log("The Convolution Block %1 HAS NOT BEEN ADDED to the CNN MCC Model.
Building the model with %2 Conv Blocks", i, i -1)
          
          self$conv_blocks <- (i - 1)
          self$error <- add_block.result
          
          break
        }
        
        conv_filters.min_value <- conv_filters
        
        rm(add_block.result)
        put_log("The Convolution Block %1 with %2 filters & kernel size = %3
has been added to the CNN MCC Model.", 
                i,
                conv_filters,
                kernel_size)
      }

      put_log("Adding the final dense hidden layers block...")
      
      
      layer <- layer |>
        layer_dropout(drop1_rate) |>
        layer_flatten() |>
        layer_dense(dense_units,
                    activation = 'relu') |>
        layer_dropout(drop2_rate)

      put_log("Creating an output layer...")
      
      output_layer <- layer |>
        layer_dense(as.integer(self$num_classes),
                    activation = 'softmax')

      put_log("Compiling the next model being tuned...")
      
      model <- keras_model(input_layer, output_layer) |>
        compile(
          optimizer = keras3::optimizer_adamax(learning_rate = ln_rate),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      put_log("The next model for tuning has been compiled.")
      put_end_date(self$start_time)
      
      return(model)
    }
  )
)
