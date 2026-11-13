# RMD Render

# Render RMD Report
start <- print_start_date()
rmarkdown::render("reports/HW-Chars.Recognition.Site/index.Rmd",
                  output_file = "capstone-CYO.report.pdf",
                  knit_root_dir = here(),
                  run_pandoc = TRUE)
print_end_date(start)
