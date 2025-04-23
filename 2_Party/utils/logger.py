from datetime import datetime

def log_message(message, log_file, print_msg=True):
    with open(log_file, 'a') as f:
        f.write(message + "\n")
    if print_msg:
        print(message)
