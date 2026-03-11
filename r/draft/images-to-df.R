if(!require(imager))
  install.packages("imager")

if(!require(OpenImageR))
  install.packages("OpenImageR")

library(imager)

# Path to your image folder
# img_dir <- "path/to/your/images"
# files <- list.files(img_dir, pattern = "\\.(jpg|jpeg|png)$", full.names = TRUE)

# Install packages if you haven't already
# install.packages(c("OpenImageR", "dplyr"))

library(OpenImageR)
library(dplyr)

# 1. Setup path and image specs
img_dir <- "path/to/your/images/"
img_files <- list.files(img_dir, full.names = TRUE)
width <- 64  # Keep it small to save memory
height <- 64

# 2. Function to process each image
process_image <- function(path) {
  img <- readImage(path)
  
  # Resize so all images have the same number of pixels
  img_resized <- resizeImage(img, width, height, method = "bilinear")
  
  # Convert to Grayscale if it's RGB (reduces 3D array to 2D matrix)
  if (length(dim(img_resized)) == 3) {
    img_gray <- 0.21 * img_resized[,,1] + 0.72 * img_resized[,,2] + 0.07 * img_resized[,,3]
  } else {
    img_gray <- img_resized
  }
  
  # Flatten to a 1D vector
  return(as.vector(img_gray))
}

# 3. Apply to all files and convert to Data Frame
img_list <- lapply(img_files, process_image)
image_df <- as.data.frame(do.call(rbind, img_list))

# 4. Add labels (assuming folder names or file patterns represent classes)
colnames(image_df) <- paste0("pixel", 1:ncol(image_df))
image_df$label <- as.factor(gsub(".*_([a-z]+)\\.jpg", "\\1", img_files)) # Example regex