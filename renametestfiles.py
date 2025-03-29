import os
path = os.getcwd()
for (root,dirs,files) in os.walk(path + '/testcases'):
    filetemp = files[:]
    filetemp.sort()
    i = 1
    for file in filetemp:
        if file == ".DS_Store": continue
        os.rename(f"{path}/testcases/{file}",f"{path}/testcases/{i:03d}.png")
        i += 1