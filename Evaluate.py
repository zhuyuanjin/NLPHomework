import numpy as np
def ComputeR10_1(scores,labels,count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        #if labels[i] is not None:
        total = total+1
        #    sublist = scores[i*10:i*10+10]
        #    if np.argmax(sublist) == labels[i]:
        if scores[i] == labels[i]:
            correct = correct + 1
    print(float(correct)/ total )
    return float(correct)/ total 

def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    print(float(correct)/ total )
