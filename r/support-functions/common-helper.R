# %%%%%%%%%%%%%%%%%%%%%%%
# Common Helper Functions
# %%%%%%%%%%%%%%%%%%%%%%%

## Results Tibble --------------------------------------------------------------

tblResults.Create <- function(method,
                              acc_value,
                              loss = NULL,
                              comment = ""){
  tibble(Method = method,
         Accuracy = acc_value,
         Loss = ifelse(is.null(loss),
                       1 - acc_value,
                       loss),
         Comment = comment)
}

tblResults.AddRow <- function(tbl_results = NULL,
                         method,
                         acc_value,
                         loss = NULL,
                         comment = "",
                         before = NULL){
  if(is.null(tbl_results)) {
    tblResults.Create(method,
                      acc_value,
                      loss,
                      comment)
  } else {
    tbl_results |>
      add_row(Method = method,
              Accuracy = acc_value,
              Loss = ifelse(is.null(loss),
                            1 - acc_value,
                            loss),
              Comment = comment,
              .before = before)
  }
}

tblResults_F1Macro.AddRow <- function(tbl_results,
                              method,
                              acc_value,
                              loss,
                              f1_macro,
                              comment = "",
                              before = NULL){
  tbl_results |>
    add_row(Method = method,
            Accuracy = acc_value,
            Loss = loss,
            F1Macro = f1_macro,
            Comment = comment,
            .before = before)
}

tblResults.AddColumn.F1Macro <- function(tbl_results) {
  tbl_results |> add_column(F1Macro = rep_len(NA, length.out = nrow(tbl_results)),
                            .before = "Comment")
}

tblResults.AddDiffColumn <- function(tbl_results){
  tblResults.Diff <- tbl_results |>
    tblResults_F1Macro.AddRow(NULL, 
                              tbl_results$Accuracy[1],
                              NULL,  # Loss
                              NA,    # F1Macro
                              NULL,  # Comment
                              before = 1)
  
  tblResults.Diff <- tblResults.Diff[-nrow(tblResults.Diff),]
  
  tbl_results <- tbl_results |>
    add_column(AccDiff = tbl_results$Accuracy - tblResults.Diff$Accuracy,
               .before = "Comment")
  
  tbl_results$AccDiff[1] <- NA
  tbl_results
}

tblResults.kable <- function(tbl_results,
                       col1width = 15,
                       col2width = 5,
                       col3width = 5,
                       col4width = 25){
  tbl_results |>
    kable(align='lcl', booktabs = T, padding = 5) |>
    row_spec(0, bold = T) |>
    column_spec(column = 1, width = tblResults.col_width(col1width)) |>
    column_spec(column = 2, width = tblResults.col_width(col2width)) |>
    column_spec(column = 3, width = tblResults.col_width(col3width)) |>
    column_spec(column = 4, width = tblResults.col_width(col4width))
}

tblResults_F1Macro.kable <- function(tbl_results,
                                     col1width = 15,
                                     col2width = 5,
                                     col3width = 5,
                                     col4width = 5,
                                     col5width = 25){
  tbl_results |>
    tblResults.kable(col1width,
                     col2width,
                     col3width,
                     col4width) |>
    column_spec(column = 5,
                width = tblResults.col_width(col5width))
}

tblResults.Total_kable <- function(tbl_results,
                             col1width = 10,
                             col2width = 5,
                             col3width = 5,
                             col4width = 5,
                             col5width = 5,
                             col6width = 20){
  tbl_results |>
    tblResults_F1Macro.kable(col1width,
               col2width,
               col3width,
               col4width,
               col5width) |>
    column_spec(column = 6,
                width = tblResults.col_width(col6width))
}
tblResults.col_width <- function(x){
  "%1em" |> msg.glue(x)
}

##Utility Functions ------------------------------------------------------------

msg.set_arg <- function(msg_template, arg, arg.name = "%1") {
  msg_template |> 
    str_replace_all(arg.name, as.character(arg))
}
msg.glue <- function(msg_template, arg, arg.name = "%1"){
  msg_template |>
    msg.set_arg(arg, arg.name) |>
    str_glue()
}
make_ordinal_no <- function(n){
  if(n == 1){
    "1st"
  } else if(n == 2) {
    "2nd"
  } else if(n == 3) {
    "3rd"
  } else {
    str_glue("{n}th")
  }
}

