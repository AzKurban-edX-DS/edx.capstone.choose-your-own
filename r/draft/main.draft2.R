# Main Script ------------------------------------------------------------------

## Initial Paths ---------------------------------------------------------------
r.path <- "r"

support_functions.folder <- "support-functions"
support_scripts.folder <- "support-scripts"

support_scripts.path <- file.path(r.path, support_scripts.folder)
support_functions.path <- file.path(r.path, support_functions.folder)

setup_script.file_path <- file.path(support_scripts.path, "setup.R")

data.path <- "data"
raw_data.path <- file.path(data.path, "raw")
raw_data.path

raw_data.folder_name <- "Vaibs.HW-Chars"
raw_data.chars.path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.chars.path

img.train.root_path <- file.path(raw_data.chars.path, "Train")
img.train.root_path

img.validation.root_path <- file.path(raw_data.chars.path, "Validation")
img.validation.root_path

dataset.path <- file.path(data.path, "dataset")
dir.create(dataset.path)
dataset.path

train.data.path <- file.path(dataset.path, "train")
dir.create(train.data.path)
train.data.path

final_test.data.path <- file.path(dataset.path, "final_test")
dir.create(final_test.data.path)
final_test.data.path

ds.subsets.path <- file.path(train.data.path, "subsets")
dir.create(ds.subsets.path)
ds.subsets.path

## Setup -----------------------------------------------------------------------
source(setup_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## Load Logging Helper Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-functions.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Load Data Helper Functions --------------------------------------------------
data_helper.funcs.file_path <- file.path(support_functions.path, "data-helper.funcs.R")


source(data_helper.funcs.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


### Open log: Load Dataset -----------------------------------------------------
open_logfile(".load-dataset")
## Download the Kaggle Dataset -------------------------------------------------

# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters
kaggle_dataset <- "vaibhao/handwritten-characters"

if(!dir.exists(raw_data.chars.path)) {
  print_log1("Downloading dataset `%1` ...", kaggle_dataset)
  kaggle_cli.download(kaggle_dataset, raw_data.chars.path, unzip = TRUE)
  print_log1("The Kaggle dataset has been downloaded and unzip to folder: `%1`", raw_data.chars.path)
} else {
  warning(get_log1("Nothing to do: directory already exists: `%1`", raw_data.chars.path))
}

# Remove duplicate files:
dir.to_remove <- file.path(raw_data.chars.path, "dataset")
dir.to_remove

if (dir.exists(dir.to_remove)) {
  print_log1("Deleting the folder with duplicate files: `%1`...", dir.to_remove)
  unlink(dir.to_remove, recursive = TRUE, force = TRUE)
  print_log1("Directory removed: `%1`", dir.to_remove)
} else {
  warning(get_log1("Nothing to do: directory does not exist: `%1`", dir.to_remove))
}

## Crating Datasets ------------------------------------------------------------
### Open log: Load Train Data --------------------------------------------------
open_logfile(".load-train-data")
### Load Train Data ------------------------------------------------------------
ds.train.list.file_path <- file.path(train.data.path, "train-data-list.RData")
ds.train.list.file_path

if (file.exists(ds.train.list.file_path)) {
  put_log1("Loading Train Data from cache file: 
%1", ds.train.list.file_path)
  
  start <- put_start_date()
  load(ds.train.list.file_path)
  put_log("Train Data list has been loaded from cache.")
  put_end_date(start)
} else {
  put_log1("Creating Train Data list from raw data files from root directory:
%1", img.train.root_path)
  start <- put_start_date()
  img.train.dat <- hwChar_data.load(img.train.root_path)
  
  put_log1("Final Test Data list structure:
%1", capture.output(str(img.train.dat)))
  
  put_log("Train Data list has been created from raw data files.")
  put_end_date(start)
  
  img.train.files <- img.train.dat$img.files
  train.labels <- img.train.dat$label.list
  img.train.list <- img.train.dat$img.list
  my_minst.train <- img.train.dat$my_mnist
  
  rm(img.train.dat)

  put_log1("Saving Train Data to the cache file: 
%1", ds.train.list.file_path)
  start <- put_start_date()
  save(img.train.files,
       img.train.list,
       my_minst.train, 
       file = ds.train.list.file_path)
  put_log("Train Data list has been cached to the File System.")
  put_end_date(start)
}

put_log1("Train image file list structure:
%1", capture.output(str(img.train.files)))

put_log1("Train dataset labels:
%1", train.labels, .sep = " ")

put_log1("`img.train.list` data structure:
%1", capture.output(str(img.train.list)))

put_log1("`my_minst.train` dataset matrix dimensions: 
%1", dim(my_minst.train), .sep = " ")

# Visualize the first char:
char.image(my_minst.train[1,])

# rm(img.train.files)
rm(img.train.list)
rm(my_minst.train)

### Close Log ---------------------------------------------------------------
log_close()

### Open log: Load Final Test Data --------------------------------------------------
open_logfile(".load-final-test-data")
### Load Final Test Data -------------------------------------------------------

ds.final_test.list.file_path <- file.path(final_test.data.path, "final-test-data-list.RData")
ds.final_test.list.file_path

if (file.exists(ds.final_test.list.file_path)) {
  put_log1("Loading Final Test Data from cache file: 
%1", ds.final_test.list.file_path)

  start <- put_start_date()
  load(ds.final_test.list.file_path)
  put_log("Final Test Data list has been loaded from cache.")
  put_end_date(start)
} else {
  put_log1("Creating Final Test Data list from raw data files from root directory:
%1", img.validation.root_path)
  start <- put_start_date()
  img.final_test.dat <- hwChar_data.load(img.validation.root_path,
                                    shuffle.rows = TRUE,
                                    shuffle.seed = 1)
  
  put_log1("Final Test Data list structure:
%1", capture.output(str(img.final_test.dat)))

  put_log("Final Test Data list has been created from raw data files.")
  put_end_date(start)
  
  img.final_test.files <- img.final_test.dat$img.files
  img.final_test.list <- img.final_test.dat$img.list
  final_test.labels <- img.final_test.dat$label.list
  my_minst.final_test <- img.final_test.dat$my_mnist
  
  rm(img.final_test.dat)
  
  put_log1("Saving Final Test Data to the cache file: 
%1", ds.final_test.list.file_path)
  start <- put_start_date()
  save(img.final_test.files,
       img.final_test.list,
       my_minst.final_test, 
       file = ds.final_test.list.file_path)
  put_log("Final Test Data list has been cached to the File System.")
  put_end_date(start)
}

put_log1("Final Test image file list structure:
%1", capture.output(str(img.final_test.files)))

put_log1("Final Test dataset labels:
%1", final_test.labels, .sep = " ")

put_log1("`img.final_test.list` data structure:
%1", capture.output(str(img.final_test.list)))

put_log1("`my_minst.final_test` dataset matrix dimensions: 
%1", dim(my_minst.final_test), .sep = " ")

char.image(my_minst.final_test[1,])

my_minst.final_test[1:120, 1:7]

# rm(img.final_test.files)
rm(img.final_test.list)
# rm(my_minst.final_test)

### Close Log ---------------------------------------------------------------
log_close()

## Data Analysis ---------------------------------------------------------------
### Load Train Data Subset----------------------------------------------------
char_files.max <- 64 
char_files.max


ds.train.subset64.file_path <- file.path(train.data.path, "train-data-subset64.RData")
ds.train.subset64.file_path

if (file.exists(ds.train.subset64.file_path)) {
  start <- put_start_date()
  load(ds.train.subset64.file_path)
  put_end_date(start)
} else {
  # Create Train Data list
  train.dat.subset64 <- hwChar_data.load(img.train.root_path, 
                                         char_files.max = char_files.max)
  
  start <- put_start_date()
  save(train.dat.subset64, file = ds.train.subset64.file_path)
  put_end_date(start)
}

str(train.dat.subset64)

### Close Log ---------------------------------------------------------------
log_close()

### Load Test Data Subset----------------------------------------------------
char_files.max <- 64 
char_files.max


ds.train.subset64.file_path <- file.path(train.data.path, "train-data-subset64.RData")
ds.train.subset64.file_path

if (file.exists(ds.train.subset64.file_path)) {
  start <- put_start_date()
  load(ds.train.subset64.file_path)
  put_end_date(start)
} else {
  # Create Train Data list
  train.dat.subset64 <- hwChar_data.load(img.train.root_path, 
                                         char_files.max = char_files.max)
  
  start <- put_start_date()
  save(train.dat.subset64, file = ds.train.subset64.file_path)
  put_end_date(start)
}

str(train.dat.subset64)


### Close Log ---------------------------------------------------------------
log_close()

### Qustion: iS Matrix centered? --------------------------
# Reference:
# 21.4 Vectorization for matrices /
# Matrix–vector operations
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#matrixvector-operations

x <- my_minst.train
dim(x)
class(x)
str(x)

y <- as.factor(rownames(x))
str(y)

row_means.x <- rowMeans(x)
max(row_means.x)
#> 0.4063876
min(row_means.x)
#> 0

#### Answer: No, let's center the matrix ----------------------
x.centered <- x - row_means.x
max(x.centered)
#> 0.9755002
min(x.centered)
#> -0.4063876


y[1]
#> "#"
# plot(x.centered[1,])
char.image(x.centered[1,])

middle.idx <- as.integer(length(y)/2)
middle.idx
#> 417018

y[middle.idx]
#> "7"
char.image(x.centered[middle.idx,])

row_means.x.centered <- rowMeans(x.centered)

### Question: Do some chars require more ink to write than others? ------------
# Reference:
# Do some digits require more ink to write than others?
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#do-some-digits-require-more-ink-to-write-than-others

boxplot(row_means.x ~ y)



### Do some chars require more ink to write than others? ----------------------
# Reference*:
# Do some digits require more ink to write than others?
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#sec-mnist)

x.sds <- colSds(x)
str(x.sds)
hist(x.sds, breaks = 30, main = "SDs")
char.image(x.sds)


#> [*] So if we wanted to remove uninformative predictors from our matrix, 
#> we could write this one line of code:

x_min <- 60/255
x_min
#> 0.2352941

clarified_x <- x[,x.sds > x_min]
dim(clarified_x)

### Can we remove smudges? ---------------------------------------
# Reference:
# Can we remove smudges?
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#can-we-remove-smudges


#> We will first look at the distribution of all pixel values.
x.hist <- hist(as.vector(x), breaks = 30, main = "Pixel intensities")
str(x.hist)

# Binarize the data -----------------------------
# Reference:
# Binarize the data
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#binarize-the-data

# img.train.dat$my_mnist[1,]
# my_minst.train[1,]

# f <- x.hist$counts

# bin_x <- x
# bin_x[x < 0.5] <- 0 
# bin_x[bin_x > 0] <- 1

str(x[1,])
x[1,]
char.image(x[1,])
char.image(x[2,])

bin.x1 <- (x[1,] > 0.5)*1
bin.x1
char.image(bin.x1)

bin.x2e4 <- (x[2e4,] > 0.5)*1
# bin.x2e4
char.image(bin.x2e4)

bin_x <- (x > 0.5)*1
char.image(bin_x[1,])
char.image(bin_x[20000,])

### Standardize the chars --------------------------------------------------------
# Reference:
# Standardize the digits
# https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#standardize-the-digits

# x.scaled <- t(scale(t(x)))
x.scaled <- sweep(sweep(x, 2, colMeans(x)), 2, colSds(x), FUN = "/")
str(x.scaled)
char.image(x.scaled[1,])
char.image(x[1,])

# min(x)
#> 0
min(x.scaled)
#> -1.227616

max(x)
#> 1
max(x.scaled)
#> 360.634

char.image(x[1e5,])
char.image(bin_x[1e5,])
char.image(x.scaled[1e5,])

## Preprocessing ---------------------------------------------------------------
# Reference:
# 31.3 Preprocessing
# https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#preprocessing

library(matrixStats)
hist(colSds(x), breaks = 256)
hist(colSds(x.scaled), breaks = 256)
hist(colSds(bin_x), breaks = 256)

start <- put_start_date()
nzv <- nearZeroVar(x)
put_end_date(start)
nzv

# Columns to remove:
length(nzv)
#> 41 

image(matrix(1:784 %in% nzv, 28, 28))

### Split Train Dataset --------------------------------------------------------
seed <- 1234
train.dataset.list <- sample_train_test_sets.mx(my_minst.train, seed)
str(train.dataset.list)

x.train <- train.dataset.list$train_set
x.test <- train.dataset.list$test_set
dim(x.test)
#> [1] 166823    784

### Close Log ---------------------------------------------------------------
log_close()

### The First MOdel ------------------------------------------------------------
library(caret)
library(doParallel)

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

pp <- preProcess(x.train, method = c("nzv", "center"))
centered_subsetted_x_test <- predict(pp, newdata = x.test)
stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(centered_subsetted_x_test)
#> [1] 166823    743
str(centered_subsetted_x_test)

y.test <- as.factor(rownames(x.test))
y.test
overall_accuracy <- mean(y.test == as.factor(rownames(centered_subsetted_x_test)))
overall_accuracy

## Cross Validation ------------------------------------------------------------
# Reference:
#
# 































