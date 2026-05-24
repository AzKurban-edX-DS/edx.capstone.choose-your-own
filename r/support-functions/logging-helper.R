# Logging Helper functions

open_logfile <- function(file_name){
  log_file_name <- as.character(Sys.time()) |> 
    str_replace_all(':', '_') |> 
    str_replace(' ', 'T') |>
    str_c(file_name)
  
  log_open(file_name = log_file_name)
}
print_start_date <- function(){
  print(date())
  Sys.time()
}
put_start_date <- function(){
  put(date())
  Sys.time()
}
print_end_date <- function(start){
  print(date())
  print(Sys.time() - start)
}
put_end_date <- function(start){
  put(date())
  put(Sys.time() - start)
}

arg.to_str <- function(arg, .sep = "\n"){
  arg <- as.character(arg)
  
  if(length(arg) > 1)
    arg <- paste(arg, collapse = .sep)
  arg
}

str.build <- function(str.template, ...,
                      .sep = "\n") {
  arg_ls <- list(...)
  str <- str.template
  
  for (i in seq_len(length(arg_ls))) {
    str <- str |>
      str_replace_all(paste0("%", as.character(i)), 
                      arg.to_str(arg_ls[[i]], .sep))
  }
  str_glue(str)
}

print_log <- function(msg_template, ...,
                      .sep = "\n"){
  print(str.build(msg_template, ...,
                .sep))
}
put_log <- function(msg_template, ...,
                    .sep = "\n"){
  put(str.build(msg_template, ..., 
              .sep = .sep))
}
