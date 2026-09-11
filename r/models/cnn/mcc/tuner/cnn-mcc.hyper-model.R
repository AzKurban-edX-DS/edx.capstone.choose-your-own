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
                          learning_rate = NULL) {
      
      self$num_classes = num_classes
      self$learning_rate = learning_rate
      self$max_blocks = 5
      NULL
    },
    
    build = function(self, hp) { # [2]
      input_layer <- layer_input(shape = shape(28L, 28L, 1L))
      layer <- input_layer

      # max_blocks <- ifelse(is.null(self$max_blocks), 5, self$max_blocks)
      
      put_log("Maximum number of Convolution Blocks: %1", self$max_blocks)
      
      conv_blocks <- hp$Int('conv_blocks',
                         min_value = 2,
                         max_value = self$max_blocks,
                         default = 2)
      
      conv_filters <- hp$Int('conv_filters',
                             min_value = 32,
                             max_value = 256,
                             step = 32)
      
      kr <- hp$Choice('kernel_size',
                      c(2L, 3L))
        
      lr <- self$learning_rate
      
      if(is.null(lr))
        lr <- hp$Choice('learning_rate', 
                        c(1e-1, 
                          1e-2, 
                          1e-3, 
                          1e-4))
      
      put_log("Building next model for tuning with learning rate %1...",
              lr)
      
      put_log("Adding %1 Convolution Blocks to the CNN MCC Model...", 
              conv_blocks)
      
      for (i in 1:conv_blocks) {
        
        lfilters <- conv_filters * i
        
        put_log("Adding the Conv Block %1 with %2 filters & kernel size = %3 
for the Conv_2d layer...", 
                i, lfilters, kr)

        add_block.result <- try(
          {
            layer <- layer |>
              layer_conv_2d(filters = lfilters,
                            kernel_size = kr,
                            # padding = 'same',
                            # strides = list(1L, 1L),
                            activation = "relu") |>
              layer_max_pooling_2d()
          }, silent = TRUE)

        if("try-error" %in% class(add_block.result)) {
          put_log("Failed to add Convolution Block %1", i)
          put_log(add_block.result)
          
          for(call in as.character(sys.calls())) {
            put_log(call)
          }

          put_log("The Convolution Block %1 HAS NOT BEEN ADDED to the CNN MCC Model.
Building the model with %2 Conv Blocks", i, i -1)
          
          self$max_blocks <- (i - 1)
          break
        }
        
        rm(add_block.result)
        put_log("The Convolution Block %1 has been added to the CNN MCC Model.", i)
      }

      put_log("Adding the final dense hidden layers block...")
      
      
      layer <- layer |>
        layer_dropout(hp$Float('dropout1',
                               min_value = 0.1,
                               max_value = 0.5,
                               step = 0.1,
                               default = 0.5)) |>
        layer_flatten() |>
        layer_dense(hp$Int('hidden_size',
                           min_value = 128L,
                           max_value = 512L,
                           step = 16,
                           default = 256L),
                    activation = 'relu') |>
        layer_dropout(hp$Float('dropout2',
                               min_value = 0.1,
                               max_value = 0.5,
                               step = 0.1,
                               default = 0.5))

      put_log("Creating an output layer...")
      
      output_layer <- layer |>
        layer_dense(as.integer(self$num_classes),
                    activation = 'softmax')

      put_log("Compiling the next model being tuned...")
      
      model <- keras_model(input_layer, output_layer) |>
        compile(
          optimizer = keras3::optimizer_adamax(learning_rate = lr),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      put_log("The next model for tuning has been compiled.")
      
      return(model)
    }
  )
)
