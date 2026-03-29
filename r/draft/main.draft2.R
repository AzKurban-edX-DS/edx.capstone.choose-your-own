# Main Script ------------------------------------------------------------------

## Initial Paths ---------------------------------------------------------------
r.path <- "r"

support_functions.folder <- "support-functions"
support_scripts.folder <- "support-scripts"

support_scripts.path <- file.path(r.path, support_scripts.folder)
support_functions.path <- file.path(r.path, support_functions.folder)

setup_script.file_path <- file.path(support_scripts.path, "setup.R")

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

## Download the Kaggle Dataset -------------------------------------------------
# Reference: https://www.kaggle.com/datasets/vaibhao/handwritten-characters
s
# Kaggle CLI command:
# kaggle datasets download vaibhao/handwritten-characters
kaggle_dataset <- "vaibhao/handwritten-characters"
raw_data.path <- "data/raw"

raw_data.folder_name <- "Vaibs.HW-Chars"
raw_data.chars.path <- file.path(raw_data.path, raw_data.folder_name)
raw_data.chars.path

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
### Create Train Dataset ----------------------------------------------

img.train.root_path <- file.path(raw_data.chars.path, "Train")
img.train.root_path

img.train.dat <- create.hwChar_dataset(img.train.root_path)
str(img.train.dat)
my_minst.train <- img.train.dat$my_mnist
char.image(my_minst.train[1,])

### Create Final Test Dataset ---------------------------------------------------
img.validation.root_path <- file.path(raw_data.chars.path, "Validation")
img.validation.root_path

img.validation.dat <- create.hwChar_dataset(img.train.root_path)
str(img.validation.dat)
my_minst.final_test <- img.validation.dat$my_mnist
char.image(my_minst.final_test[1,])


## Data Analysis ---------------------------------------------------------------
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


## Cross Validation ------------------------------------------------------------
# Reference:
#
# 






### Split Train Dataset --------------------------------------------------------
seed <- 1
train.dataset.list <- sample_train_test_sets.mx(my_minst.train, seed)
str(train.dataset.list)


























