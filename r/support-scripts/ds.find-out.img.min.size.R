#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figuring Out the Smallest Image Size
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Analyzing the Training Data --------------------------------------------------

open_logfile(".find-out.smallest-img_dat-size")

start <- put_start_date()
put_log("Loading list of Trimmed Image objects...")
img_dat.list <- readRDS(train.img_dat.list.file)
put_log("The list of Trimmed Image objects has been loading from the following file:
%1", train.img_dat.list.file)
put_end_date(start)

# length(img_dat.list)
# str(img_dat.list$img.list$W$img.list)

x <- lapply(img_dat.list$img.list, function(ch.class) {
  img_dat.ls <- ch.class$img.list
  x <- lapply(img_dat.ls, function(img_dat) {
    dim(img_dat[1,,])
  })
}) |> unlist() |>
  matrix(ncol = 2, 
         byrow = T, 
         dimnames = list(NULL, 
                         c('width', 'height')))

put_log("Shape of the Image Size Array:
%1", capture.output(shape(x)))

put_log("Structure of the Image Size Array:
%1", capture.output(str(x)))

z <- sapply(seq(nrow(x)), function(i) {
  x[i,1]*x[i,2]
})

names(z) <- NULL
put_log("Structure of the Flatten Image Size Array:
%1", capture.output(str(z)))

min.z <- min(z)

put_log("Number of pixels in the smallest image:
%1", min.z)


min.x <- x[z == min.z,]

put_log("Structure of the Smallest Image Size Array:
%1", capture.output(str(min.x)))

head(min.x)

rm(img_dat.list, x, z)

put_log("Size of the smallest image in the training dataset,
%1", capture.output(min.x[which.min(min.x[,1]),]))

# [1] 28 28
rm(min.x)
gc()

log_close()
# =========================================================================
# Log End Time: 2026-10-07 20:01:15.702716
# Log Elapsed Time: 0 00:00:29
# =========================================================================
