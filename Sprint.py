def sprint(text: object,verbose:int = 1,filepath:str= "system")-> None:
    """
    this function is used to either print directly to the console or write to a file
    """
    if verbose == 1:
            print(text)
    else:
        with open(f"{filepath}.txt","a+") as f:
            print(text,file=f)

    return None
    