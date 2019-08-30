import sys
import os
import csv
folder_path = 'stable'
origin_label=[]
survey_label=[]
for fname in sorted(os.listdir(folder_path)):
    # print(fname)
    filename = 'stable/%s'%(fname)
    # print(filename)
    with open(filename,'r',encoding="utf-8") as f:
        csvreader = csv.reader(f,delimiter=',',quotechar='"')
        labels=[]
        for row in csvreader:
            # print(csvreader.line_num)
            if csvreader.line_num != 1:
                labels.append(row[4])
        survey_label.append(labels)
    
filename = 'stable/stable_test_no1 - stable_test_no1.csv'
with open(filename,'r',encoding="utf-8") as f:
    csvreader = csv.reader(f,delimiter=',',quotechar='"')
    for row in csvreader:
        if csvreader.line_num!=1:
            origin_label.append(row[3])

# row = len(origin_label)
accuracy_chance = 0
col = len(survey_label)
reliable_data = []
i=0
for label in origin_label:
    labels = [label]
    for survey in survey_label:
        labels.append(survey[i])
    i=i+1
    count_0 =[0,0]
    count_1 =[1,0]
    count_2 =[2,0]
    for l in labels:
        if l == '2':
            count_2[1] = count_2[1]+1
        if l == '1':
            count_1[1] = count_1[1]+1
        if l == '0':
            count_0[1] = count_0[1]+1
    a = [count_0,count_1,count_2]
    chance = [0,0]
    for count in a:
        if count[1] > chance[1]:
            chance[0] = count[0]
            chance[1] = count[1]
    if int(labels[0]) == chance[0]:
        accuracy_chance = accuracy_chance+1
    chances = chance[1]/(col+1)
    reliable_data.append(chances)
# print(reliable_data)
sum_chance =0
for _chance in reliable_data:
    sum_chance = sum_chance+_chance
sum_chance = sum_chance/len(reliable_data)
print(sum_chance)
print(accuracy_chance)
# print(origin_label)
# print(survey_label)