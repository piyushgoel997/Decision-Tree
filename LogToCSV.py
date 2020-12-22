import os

directory = 'C:/MyFolder/Thesis Work/Decision Trees/DataSets/Bin_classes/logs/'
out = ""
for filename in os.listdir(directory):
    f = open(directory + filename).read().split("\n")
    # open("DataSets/MadeDataSets/logs/car-log.txt")
    line1 = ""
    line2 = ""
    for i in [0, 13, 26]:
        for l in f[-40 + i:-37 + i]:
            line1 += l.split("=")[-1] + ";"
        for l in f[-35 + i:-32 + i]:
            line2 += l.split("=")[-1] + ";"
    line1 += " " + f[-3] + f[-4].split("(")[1][:-1] + " " + filename
    line2 += " " + f[-2]
    out += line1 + "\n"
    out += line2 + "\n"

out = out.replace(",", "~")
out = out.replace(";", ",")
out = out.replace("~", ";")
f = open("Final_results_bin.csv", 'w')
f.write(out)
