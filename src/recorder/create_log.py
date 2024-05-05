import datetime
import os

def CreateLog(name : str, dir : str, **vars) -> None:
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist")
    
    with open(f"{dir}/log_{name}.txt", "w") as f:
        f.write(f"Log created at {datetime.datetime.now()}\n\n\n")
        for key, value in vars.items():
            f.write(f"{key} : {value}\n")




