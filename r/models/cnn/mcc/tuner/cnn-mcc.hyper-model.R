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
                          kernal_size.min = 3L,
                          kernal_size.max = 5L,
                          cnvFilters.min = 32L,
                          cnvFilters.max = 128L,
                          learning_rate.min = 1e-4,
                          learning_rate.max = 1e-2,
                          start_date = NULL,
                          conv_blocks) {
      
      self$num_classes = num_classes
      self$kernal_size.min = kernal_size.min
      self$kernal_size.max = kernal_size.max
      
      self$cnvFilters.min = cnvFilters.min
      self$cnvFilters.max = cnvFilters.max
      
      self$learning_rate.min = learning_rate.min
      self$learning_rate.max = learning_rate.max
      
      self$start_date = ifelse(is.null(start_date), 
                               Sys.time(), 
                               start_date)
      
      self$conv_blocks = conv_blocks
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
      
      put_log("Maximum number of Convolution Blocks: %1", self$conv_blocks)
      
      # kernel_size <- hp$Choice('kernel_size',
      #                          c(2L, 3L))
      
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
      
      learning_rate <- hp$Float('learning_rate',
                                min_value = self$learning_rate.min,
                                max_value = self$learning_rate.max,
                                sampling = "log")
      
      put_log("Tuning the model of %1 Convolution Blocks with the following hype-parameters: 
Min `conv filters`: %2,
Max `conv filters`: %3, 
Learning Rate: %4.
dropout layer 1 rate: %5,
dense layer units: %6,
dropout layer 2 rate: %7",
              self$conv_blocks,
              self$cnvFilters.min,
              self$cnvFilters.max,
              learning_rate,
              drop1_rate,
              dense_units,
              drop2_rate)

      for (i in 1:self$conv_blocks) {
        
        conv_filters <- hp$Int(paste0('conv', i, '_filters'),
                               min_value = self$cnvFilters.min,
                               max_value = self$cnvFilters.max,
                               step = 32L)
        
        kernel_size <- hp$Int(paste0('conv', i,'_kernel.size'), 
                              min_value = self$kernal_size.min,
                              max_value = self$kernal_size.max,
                              step = 1L)

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
          optimizer = keras3::optimizer_adamax(learning_rate),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      put_log("The next model for tuning has been compiled.")
      put_end_date(start)
      
      return(model)
    }
  )
)
