
import numpy as np
import split
X = np.array([[1,2],
             [3,4],
             [5,6],
             [7,8],
             [9,10],
             [11,12],
             [13,14]])
y = np.array([0,1,2,3,4,5,6])

X,y=split.delete_question_mark(X,y)
print(X,y)
print(split.split_wake_and_asleep(X,y))
print(split.split_deep_sleep(X,y))
print(split.split_1_2(X,y))
print(split.split_3_4(X,y))