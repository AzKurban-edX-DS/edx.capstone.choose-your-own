support_scripts.dir <- "r/support-scripts"

index.script_path <- "r/index.R"
rmd_render.script_path <- file.path(support_scripts.dir, "__rmd.render.R")

stopifnot(file.exists(index.script_path,
                      rmd_render.script_path))

## Run Project Scripts ---------------------------------------------------------

source(index.script_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Render PDF Reports ----------------------------------------------------------

source(rmd_render.script_path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
