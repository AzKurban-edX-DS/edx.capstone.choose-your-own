support_scripts.dir <- "r/support-scripts"


## Run Project Scripts ---------------------------------------------------------
index.script.path <- "r/index.R"

stopifnot(file.exists(index.script.path))

source(index.script.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Render PDF Reports ----------------------------------------------------------
rmd_render.script.path <- file.path(support_scripts.dir, "__rmd.render.R")
stopifnot(file.exists(rmd_render.script.path))

source(rmd_render.script.path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
