#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Chapter 3: Introduction to Keras and TensorFlow
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 3.6 First Steps with TensorFlow ---------------------------------------------

### 3.6.1 TensorFlow tensors ---------------------------------------------------
library(keras3)
library(tensorflow)

r_array <- array(1:6, c(2, 3))
r_array
#      [,1] [,2] [,3]
# [1,]    1    3    5
# [2,]    2    4    6


tf_tensor <- as_tensor(r_array)
tf_tensor
# tf.Tensor(
# [[1 3 5]
#  [2 4 6]], shape=(2, 3), dtype=int32)

dim(tf_tensor)

tf_tensor + tf_tensor
# tf.Tensor(
# [[ 2  6 10]
#  [ 4  8 12]], shape=(2, 3), dtype=int32)

methods(class = "tensorflow.tensor")
#  [1] -           !           !=          %%          %*%         %/%         &           *           /          
# [10] @           @<-         [           [<-         ^           |           +           <           <=         
# [19] ==          >           >=          abs         acos        all.equal   all         any         aperm      
# [28] Arg         as.array    asin        atan        cbind       ceiling     Conj        cos         cospi      
# [37] digamma     dim         exp         expm1       floor       Im          is.finite   is.infinite is.nan     
# [46] length      lgamma      log         log10       log1p       log2        max         mean        min        
# [55] Mod         print       prod        range       rbind       Re          rep         round       sign       
# [64] sin         sinpi       sort        sqrt        str         sum         t           tan         tanpi      
# [73] type_sum   
# see '?methods' for accessing help and source code

tf_tensor$ndim
# 2

as_tensor(1)$ndim
# 0

as_tensor(1:20)$ndim
# 1

tf_tensor$shape
# TensorShape([2, 3])

methods(class = class(shape())[1])
# [1] !=         [          ==         as.integer as.list    format     print      r_to_py    Summary   
# see '?methods' for accessing help and source code

shape(2, 3)
# shape(2, 3)

tf_tensor$dtype
# tf.int32

r_array <- array(1)
typeof(r_array)
# [1] "double"

as_tensor(r_array)$dtype
# tf.float64

## 3.8 Anatomy of a Neural Network: Understanding Core `Keras` APIs ------------

### 3.8.1 Layers: The building blocks of deep learning -------------------------

#### Listing 3.20 Implementing a dense layer as a Keras Layer class

layer_simple_dense <- new_layer_class(
  classname = "SimpleDense",
  
  
  initialize = function(units, activation = NULL) {
    super$initialize()
    self$units <- as.integer(units)
    self$activation <- activation
  },
  
  
  build = function(input_shape) {                 # ➊
    input_dim <- input_shape[length(input_shape)] # ➋
    self$W <- self$add_weight(
      shape = c(input_dim, self$units),           # ➌
      initializer = "random_normal")
    self$b <- self$add_weight(
      shape = c(self$units),
      initializer = "zeros")
  },
  
  
  call = function(inputs) {                       # ➍
    y <- tf$matmul(inputs, self$W) + self$b
    if (!is.null(self$activation))
      y <- self$activation(y)
    y
  }
)

#   ➊ Weight creation takes place in the build() method.
#   ➋ Take the last dim.

#>  ➌ add_weight() is a shortcut method for creating weights. 
#>  It is also possible to create standalone variables 
#>  and assign them as layer attributes, like this: 
  self$W < - tf$Variable(tf$random$normal(w_shape)).

#   ➍ We define the forward pass computation in the call() method.

#> Once instantiated, a layer instance can be used just like a function, 
#> taking as input a TensorFlow tensor:
  
  
  my_dense <- layer_simple_dense(units = 32, #            ➊
                                 activation = tf$nn$relu)
  input_tensor <- as_tensor(1, shape = c(2, 784)) #       ➋
  output_tensor <- my_dense(input_tensor) #               ➌
  output_tensor$shape
  
  
 # TensorShape([2, 32])
  
  
# ➊ Instantiate our layer, defined previously.
# ➋ Create some test inputs.
# ➌ Call the layer on the inputs, just like a function.

#### AUTOMATIC SHAPE INFERENCE: BUILDING LAYERS ON THE FLY ---------------------

layer <- layer_dense(units = 32, activation = "relu") # ➊

# ➊ A dense layer with 32 output units
  
#> Suppose you write the following:

model <- keras_model_sequential(list(
  layer_dense(units = 32, activation = "relu"),
  layer_dense(units = 32)
))
  
  
#> The layers didn’t receive any information about the shape of their inputs—instead, 
#> they automatically inferred their input shape as being the shape of the first inputs they see.  

# Optimizers:
ls(pattern = "^optimizer_", "package:keras3")
#  [1] "optimizer_adadelta"   "optimizer_adafactor"  "optimizer_adagrad"    "optimizer_adam"      
#  [5] "optimizer_adam_w"     "optimizer_adamax"     "optimizer_ftrl"       "optimizer_lamb"      
#  [9] "optimizer_lion"       "optimizer_loss_scale" "optimizer_muon"       "optimizer_nadam"     
# [13] "optimizer_rmsprop"    "optimizer_sgd" 

#Losses:
ls(pattern = "^loss_", "package:keras3")
#  [1] "loss_binary_crossentropy"                   "loss_binary_focal_crossentropy"            
#  [3] "loss_categorical_crossentropy"              "loss_categorical_focal_crossentropy"       
#  [5] "loss_categorical_generalized_cross_entropy" "loss_categorical_hinge"                    
#  [7] "loss_circle"                                "loss_cosine_similarity"                    
#  [9] "loss_ctc"                                   "loss_dice"                                 
# [11] "loss_hinge"                                 "loss_huber"                                
# [13] "loss_kl_divergence"                         "loss_log_cosh"                             
# [15] "loss_mean_absolute_error"                   "loss_mean_absolute_percentage_error"       
# [17] "loss_mean_squared_error"                    "loss_mean_squared_logarithmic_error"       
# [19] "loss_poisson"                               "loss_sparse_categorical_crossentropy"      
# [21] "loss_squared_hinge"                         "loss_tversky"  

# Metrics:  
ls(pattern = "^metric_", "package:keras3")
#  [1] "metric_auc"                               "metric_binary_accuracy"                  
#  [3] "metric_binary_crossentropy"               "metric_binary_focal_crossentropy"        
#  [5] "metric_binary_iou"                        "metric_categorical_accuracy"             
#  [7] "metric_categorical_crossentropy"          "metric_categorical_focal_crossentropy"   
#  [9] "metric_categorical_hinge"                 "metric_concordance_correlation"          
# [11] "metric_cosine_similarity"                 "metric_f1_score"                         
# [13] "metric_false_negatives"                   "metric_false_positives"                  
# [15] "metric_fbeta_score"                       "metric_hinge"                            
# [17] "metric_huber"                             "metric_iou"                              
# [19] "metric_kl_divergence"                     "metric_log_cosh"                         
# [21] "metric_log_cosh_error"                    "metric_mean"                             
# [23] "metric_mean_absolute_error"               "metric_mean_absolute_percentage_error"   
# [25] "metric_mean_iou"                          "metric_mean_squared_error"               
# [27] "metric_mean_squared_logarithmic_error"    "metric_mean_wrapper"                     
# [29] "metric_one_hot_iou"                       "metric_one_hot_mean_iou"                 
# [31] "metric_pearson_correlation"               "metric_poisson"                          
# [33] "metric_precision"                         "metric_precision_at_recall"              
# [35] "metric_r2_score"                          "metric_recall"                           
# [37] "metric_recall_at_precision"               "metric_root_mean_squared_error"          
# [39] "metric_sensitivity_at_specificity"        "metric_sparse_categorical_accuracy"      
# [41] "metric_sparse_categorical_crossentropy"   "metric_sparse_top_k_categorical_accuracy"
# [43] "metric_specificity_at_sensitivity"        "metric_squared_hinge"                    
# [45] "metric_sum"                               "metric_top_k_categorical_accuracy"       
# [47] "metric_true_negatives"                    "metric_true_positives"        
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
