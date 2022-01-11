import ftplib

FTP_HOST = "10.3.141.1"
# enter the login details to the pi
FTP_USER = "pi"
FTP_PASS = "lux12345"

print("Connecting to FTP...")

# connect to the FTP server
ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)

print("Connected")

# # force UTF-8 encoding
ftp.encoding = "utf-8"
# the name of file you want to download from the FTP server

files = ftp.nlst()

print(files)
print(f"There are {len(files)} files in the directory")

# quit and close the connection
ftp.quit()
