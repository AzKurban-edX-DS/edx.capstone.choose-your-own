#%%%%%%%%%%%%%%%%%%%%%
# Load Flatten Dataset
#%%%%%%%%%%%%%%%%%%%%%

## Open log: Load Input Data ---------------------------------------------------
open_logfile(".load-input-data")
## Load Flatten Dataset --------------------------------------------------------

stopifnot(file.exists(my_emnist.file_path))
start <- put_start_date()
          
put_log("Loading the Flattened Training Dataset from the backup file...")
my_emnist.set <- readRDS(my_emnist.file_path)

put_log("The Flattened Training Dataset has been loaded from the following backup:
%1", my_emnist.file_path)

put_log("The Flattened Dataset Set (containing the  EMNIST-like dataset) has the following structure:
  %1", capture.output(str(my_emnist.set)))

if(!exists("Y.Labels")) {
  stopifnot(file.exists(classifier.label_list.file_path))
  Y.Labels <- readRDS(classifier.label_list.file_path)
}

put_log("The Classifier Handwritten Character Class List contains the following labels:
%1", Y.Labels, .sep = " ")

## Quick data analysis ---------------------------------------------------------

x <- my_emnist.set$img.mx
  
y.groups <- ds.get_classIDs.grouped(x)
y <- y.groups$classID
str(y)
length(y)
rm(y)

# y.int <- as.integer(y)
y.chars <- y.groups$groupByClass
rm(y.groups)

put_log("The number of handwritten character images by class:
%1", capture.output(print(y.chars, n = length(y.chars$classID))))
{
# A tibble: 39 × 2
# classID     n
# <fct> <int>
# 1 #     15600
# 2 $     16199
# 3 &     13000
# 4 @     38009
# 5 0     65504 # max(n)
# 6 1     43773
# 7 2     39351
# 8 3     39996
# 9 4     38112
# 10 5     32317
# 11 6     38879
# 12 7     41080
# 13 8     38795
# 14 9     38319
# 15 A     17205
# 16 B      8666
# 17 C     13560
# 18 D     15509
# 19 E     32627
# 20 F     11635
# 21 G      5443
# 22 H     12133
# 23 I     13873
# 24 J      4261 # min(n)
# 25 K      4334
# 26 L     21648
# 27 M     12089
# 28 N     21421
# 29 P     11095
# 30 Q      4707
# 31 R     20498
# 32 S     25910
# 33 T     30853
# 34 U     16385
# 35 V      7246
# 36 W      7266
# 37 X      5106
# 38 Y      6762
# 39 Z      4866
invisible(NULL)
}

max(y.chars$n)
# 65504
y.chars$classID[which.max(y.chars$n)]
#> [1] 0
#> Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

min(y.chars$n)
# 4261
y.chars$classID[which.min(y.chars$n)]
#> [1] J
#> Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z

# clean up Environment

rm(y.chars)

## Split the Flatten Dataset into a Train & Test Sets with 20% Test Ratio ------
stopifnot(exists("my_emnist.split.file_path"))

put_log("Splitting the Flattened Training Dataset into a Train and Test Sets with 20% Test Ratio...")

set.seed(N.classes)
ds_flatten.split_list <- sample_train_test_sets.mx(x, my_emnist.set$img.file_path)

put_log("The Flattened Training Dataset has been split into a Train and Test Sets:
%1", capture.output(str(ds_flatten.split_list)))

put_log("Saving a backup copy of a split dataset...")

saveRDS(ds_flatten.split_list, 
        my_emnist.split.file_path)

rm(ds_flatten.split_list)

put_log("The backup copy of a split dataset has been saved in the following file:
%1", my_emnist.split.file_path)

## Split the Dataset into Sets with 90% Test Ratio (10% for Training Set) ---------
my_emnist.0.1split.file_path <- file.path(train.data.dir, "my_emnist-split(10%train-set).rds")

put_log("Splitting the Flattened Training Dataset into a Train and Test Sets 
with 90% Test Ratio (10% for Training Set)...")

set.seed(N.classes)
ds_flatten.0.1split_list <- sample_train_test_sets.mx(x, 
                                                      my_emnist.set$img.file_path,
                                                      test.ratio = 0.9)
rm(x)
rm(my_emnist.set)

put_log("The Flattened Training Dataset has been split into a Train and Test Sets 
with 90% Test Ration (10% for Training Set):
%1", capture.output(str(ds_flatten.0.1split_list)))

put_log("Saving a backup copy of a split dataset (10% for Training Set)...")

saveRDS(ds_flatten.0.1split_list, 
        my_emnist.0.1split.file_path)

rm(ds_flatten.0.1split_list)

put_log("The backup copy of a split dataset (10% for Training Set) has been saved in the following file:
%1", my_emnist.0.1split.file_path)

## Close Log -------------------------------------------------------------------
log_close()
