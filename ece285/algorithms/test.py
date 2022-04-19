
# from ece285.utils.data_processing import get_cifar10_data
from optparse import Values
from turtle import distance
import numpy as np 
a = np.array([[1,  2, 6],  [3,  4, 7]])  
print (a)
print (np.shape(a))


# # Use a subset of CIFAR10 for KNN assignments
# dataset = get_cifar10_data(subset_train=5000, subset_val=250, subset_test=500)

# print(dataset.keys())
# print("Training Set Data  Shape: ", dataset["x_train"].shape)
# print("Training Set Label Shape: ", dataset["y_train"].shape)

# a = np.array([[1,  2, 6],  [3,  4, 7], [0,0,0]])  
# b = np.array([1,2,3], [7,7,7])
# print(b - a)
# dist = np.linalg.norm(a,b)
# print(dist.shape())

a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度

print(np.array([[1,2,3,4]]))
print(np.array([list(np.linalg.norm(bb+a,axis= 1))]))
print(np.insert(np.array([list(np.linalg.norm(bb+a,axis= 1))]), 0, values = np.array([[1,2,3,4]]),axis = 0))
print(type(np.linalg.norm(bb,axis= 1)))

print(a)
print(bb)
print(a + bb)


# distance

# idex = heapq.nsmallest(k_test, range(len(distance)), distance[i].take)

classes = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

for y, cls in enumerate(classes):
    print([y,cls])


labels=np.array([[1],[2],[0],[1]]) #一共三类
print(labels)
print(labels.reshape(-1))
res=np.eye(3)[labels.reshape(-1)]
print("labels转成one-hot形式的结果:\n",res,"\n")
print("labels转化成one-hot后的大小:",res.shape)
print(res.reshape(list(labels.shape)+[3]).shape )
print(1/(1+np.exp(-res)))
res[res==0] = -1
print(res)
# weights = np.random.randn(num_classes, D + 1) * 0.0001




print(np.sum([0.5, 0.7, 0.2, 1.5],axis= 0))

print(weights.shape)

