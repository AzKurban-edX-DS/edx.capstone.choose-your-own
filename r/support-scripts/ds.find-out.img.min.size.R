img_mx.ls <- readRDS(train.trimmed_img.file)
length(img_mx.ls)

str(img_mx.ls$img.list$W$img.list)

x <- lapply(img_mx.ls$img.list, function(ch.class) {
  img.ls <- ch.class$img.list
  x <- lapply(img.ls, function(img) {
    dim(img)
  })
}) |> unlist() |>
  matrix(ncol = 2, byrow = T)

dim(x)
str(x)

z <- sapply(seq(nrow(x)), function(i) {
  x[i,1]*x[i,2]
})

str(z)

min.z <- min(z)
min.z

min.x <- x[z == min.z,]
str(min.x)

head(min.x)

min.x[which.min(min.x[,1]),]
# [1] 28 28

rm(img_mx.ls)
rm(z)
rm(x)
rm(min.x)
