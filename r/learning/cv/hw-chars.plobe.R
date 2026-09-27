#load the packages in R

library(matrixStats)
library(dslabs)
library(tidyverse)

library(prodlim)
library(caret)
library(randomForest)

library(logr)
library(reticulate)

library(pacman)

library(tensorflow)
library(keras3)

library(imager)
library(magick)

# Raw Data Loading -------------------------------------------------------------
kaggle_dataset <- "tarunkumarkaggle/english-alphabets-28-and-64-px-handwritten-for-ocr"
raw_data_path <- "learning/cv/data/raw"
# zip_file_name <- "drone-detection-dataset.zip"

# full_zip_file_name <- file.path(raw_data_path, zip_file_name)
raw_data_folder_name <- "hw-en-chars"
raw_data_folder_path <- file.path(raw_data_path, raw_data_folder_name)


if(!dir.exists(raw_data_folder_path)) {
  # if(!file.exists(full_zip_file_name)) {
  if (system("kaggle --version", ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    stop("Kaggle CLI is not installed or not in the PATH.")
  }
  
  if (system(paste("kaggle datasets download", kaggle_dataset, "--path", raw_data_folder_path, "--unzip")) != 0) {
    stop("Failed to download the dataset with Kaggle CLI.")
  }
  
}

if(!dir.exists(raw_data_folder_path)) {
  stop("Failed to download and unzip the dataset `BirdVsDroneVsAirplane`.")
  
}

A_Z.img_path <- file.path(raw_data_folder_path, "A-Z/Augmentated 28 X 28")
A_Z.img_path

charLabels_A_Z28 <- dir(A_Z.img_path)
charLabels_A_Z28

output_n <- length(charLabels_A_Z28)
output_n

#list all the img files in img path
A_Z.imgfiles <- list.files(A_Z.img_path, 
                           full.names = TRUE, 
                           pattern = "png",
                           recursive = TRUE) 
head(A_Z.imgfiles)


## read the images
# imlist.A_Z <- map(A_Z.imgfiles, magick::image_read)
# str(imlist.A_Z)

# img_list <- map(A_Z.imgfiles[1:10], load.image)
img_list3 <- lapply(A_Z.imgfiles[1:3], imager::load.image)
img_list3
str(img_list3)

# imgnames <- list.files(imgpath, pattern = "jpg") #without full names gets only the img title.   

img_list5 <- map_il(A_Z.imgfiles[1:5], imager::load.image)
img_list5
str(img_list5)

names(img_list5) <- c("A","A","A","A","A")
str(img_list5)

str(img_list5[[1]])

img1 <- img_list5[[1]]
dim(img1)
dim(img1)[4]

nch <- channels(img1)
str(nch)

sh1 <- shape(img1)

str(sh1)
class(sh1)

sh1[[4]]

img1[,,1,1]

gray.img1 <- imager::grayscale(img1)
dim(gray.img1)

img1_mx <- gray.img1[,,1,1]
dim(img1_mx)
img1_mx

image(img1_mx)

plot(gray.img1)
plot(img1)
















#read the images in the folder with imager and store it into a variable

# imgList <- imager::load.dir(A_Z.img_path)
# str(imgList)
# head(imgList)


#read the images in the folder with magick

# imglist <- list() # define empty list
# 
# 
# 
# for(img in seq_along(imgfiles)) {
#   
#   imglist[[img]] <- magick::image_read(imgfiles[[img]])
#   
# }      









