def check_accuracy_of_label(numbertestcase,label,pred):
    count = 0
    _all = 0
    number = label*numbertestcase
    tests = pred[number:(number+numbertestcase)]
    for row in pred:
        if row == label:
            _all = _all +1
    if label==0:   
        for test in tests:
            if test ==0:
                count = count +1
    if label == 1:
        for test in tests:
            if test == 1:
                count = count +1
    if label == 2:
        for test in tests:
            if test == 2:
                count = count +1
    return count/_all