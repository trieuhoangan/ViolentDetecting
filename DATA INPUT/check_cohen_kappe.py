import sys
import os
import csv
from sklearn.metrics import cohen_kappa_score
folder_path = 'random'

cohen_acc =[]
different_acc = []
for fname in sorted(os.listdir(folder_path)):
    filename = 'random/%s'%(fname)
    origin_label=[]
    survey_label=[]
    labels=[0,1,2]
    counter = 0
    with open(filename,'r',encoding="utf-8") as f:
        csvreader = csv.reader(f,delimiter=',',quotechar='"')
        labels=[]
        for row in csvreader:
            if csvreader.line_num != 1:
                
                if row[4]!='NULL':
                    # print(row[4])
                    survey_label.append(int(row[4]))
                    origin_label.append(int(row[3]))
                    # if row[4]==row[3]: 
                    #     counter = counter+1
        # print(survey_label)
        cohen = cohen_kappa_score(survey_label,origin_label)
        cohen_acc.append(cohen)
        i = 0
        count = 0
        for label in origin_label:
            if label==survey_label[i]:
                count= count+1
            i= i+1
        print(count)
print(cohen_acc)
