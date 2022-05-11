
with open("sim_params.txt", "r") as f:
    lines = f.readlines()

for x in lines[3:]:
    splitted = x.split("=")
    RHS = splitted[1]
    print(RHS)
    print(type(RHS))