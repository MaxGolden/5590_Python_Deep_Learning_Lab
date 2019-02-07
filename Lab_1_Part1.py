
current = 0
new_log = 0

while True:
    new_log = input("enter Deposit or Withdraw followed by the amount: ")
    log_split =  new_log.split(" ")
    if log_split[0].lower()== "deposit":
        try:
            current += int(log_split[1])
        except:
            print ("invalid input")

    elif log_split[0].lower()== "withdraw":
        try:
            current -= int(log_split[1])
        except:
            print ("invalid input")

    else:
        print("invalid input main")

    print ("current balance: ",current)


