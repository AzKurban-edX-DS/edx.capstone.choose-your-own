#%%%%%%%%%%%%%%%%%%
# Load CNN Dataset
#%%%%%%%%%%%%%%%%%%

#### Open log: Prepare CNN Datasets -------------
open_logfile(".prepare-cnn-datasets")
#### Load CNN Dataset ------------------------------------------------------

x_cnn <- readRDS(train.img28x28mx.array.file_path)
put_log("The CNN-based models input dataset structure:
%1", capture.output(str(x_cnn)))

if(!exists("y.labels"))   
  y.labels <- readRDS(classifier.label_list.file_path)

put_log("The Classifier Handwritten Character List contains the following labels:
%1", y.labels, .sep = " ")


dim.x_cnn <- dim(x_cnn)
dim.x_cnn
#> [1] 834032     28     28
nrow(x_cnn)
#> [1] 834032

y_cnn <- as.factor(rownames(x_cnn))
str(y_cnn)
length(y_cnn)

# Input image dimensions
img_rows <- dim.x_cnn[2]
img_rows
img_cols <- dim.x_cnn[3]
img_cols

first_G.idx <- which(y_cnn == "G")[1]
first_G.idx
#> [1] 598137

char.image(x_cnn[first_G.idx,,])
char.image(x_cnn[first_G.idx - 1,,])

#> [1] 28 28  1
#### Close Log ------------------------------------------------------------------
log_close()


