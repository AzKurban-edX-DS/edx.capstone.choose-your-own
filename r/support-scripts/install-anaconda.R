# Reference:
# https://www.anaconda.com/docs/getting-started/anaconda/install/windows-cli-install#installation-steps

install_anaconda_cmd <- "Invoke-WebRequest -Uri 'https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Windows-x86_64.exe' -OutFile './Anaconda3-2025.12-2-Windows-x86_64.exe'"

# Run it cleanly by disabling the profile loading for faster execution (-NoProfile)
output <- system2("powershell", 
                  args = c("-NoProfile", "-Command", paste0('"', install_anaconda_cmd, '"')), 
                  stdout = TRUE,
                  stderr = TRUE)
print(output)
