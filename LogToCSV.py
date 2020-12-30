import os

directory = r'C:/MyFolder/Thesis Work/Decision Trees/DataSets/Bin_features_bin_classes-cut/logs/'
out = ""
out += "Missing-data percentage;;;;;;;;;;;;Information about the data set;\n"
out += "25;;;;50;;;;75;\n"
out += "No Sampling;Random Sampling;IF Sampling;LEU Sampling;" \
       "No Sampling;Random Sampling;IF Sampling;LEU Sampling;" \
       "No Sampling;Random Sampling;IF Sampling;LEU Sampling;" \
       "Name;Clf Acc;Tr Acc;Num attrs; Num Instances\n"

for filename in os.listdir(directory):
    f = open(directory + filename).read().split("\n")
    # open("DataSets/MadeDataSets/logs/car-log.txt")
    line1 = ""
    line2 = ""
    for i in [0, 15, 30]:
        for l in f[-48 + i:-44 + i]:
            line1 += l.split("=")[-1] + ";"
        for l in f[-42 + i:-38 + i]:
            line2 += l.split("=")[-1] + ";"
    line1 += " " + filename.split(".")[0] + ";"  # filename
    line1 += f[-7].split("=")[1] + ";"  # acc with complete data
    line1 += f[-6].split("=")[1] + ";"  # acc of the trivial classifier
    line1 += f[-2].split("=")[1] + ";"  # num attrs
    line1 += f[-3].split("=")[1] + f[-4].split("(")[1][:-1] + ";"  # num instances and class counts
    out += line1 + "\n"
    out += line2 + ";;;;\n"

out = out.replace(",", "~")
out = out.replace(";", ",")
out = out.replace("~", ";")
f = open("Final_results_bfbc.csv", 'w')
f.write(out)
