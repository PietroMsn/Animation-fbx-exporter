import numpy as np

res = []
count = 0
with open('01/01_03.bvh','rb') as fileBvh:
	for line in fileBvh:
		if len(line) > 700:
			res.append(line)
			count += 1
BvhJoint = [0,2,7,11,3,8,12,4,9,13,5,10,15,17,24,16,18,25,19,26,20,27,21,28]

result = []

for j in range(0,len(res)):
	x = [float(i) for i in res[j].split()]
	result.append(x)

ult = np.zeros((len(result),len(result[0])/3,3))

for i in range(0,len(result)):
	for j in range(0,len(result[i])):
		ult[i][j/3][j%3] = result[i][j]

print ult[0][0]

SMPLrot = np.zeros((len(result),len(BvhJoint),3))

for j in range(0,len(result)):
	for i in range(0,len(BvhJoint)):
		SMPLrot[j][i] = ult[j][BvhJoint[i]]

print len(SMPLrot)

np.save('BVHtoSMPL.npy', SMPLrot)
