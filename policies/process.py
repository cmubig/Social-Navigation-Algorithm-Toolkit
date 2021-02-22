import numpy as np



a=[]
#test.txt contain crowd_zara2.txt
#timestamp 1250 to 1310 data, only preserving the 10 agents that are present throughout this 8 timesteps

with open("test.txt") as text:
    for row in text:
        #print(row.split())
        #print(float(row.split()[-1]))

        timestamp, agent_index, x, y = row.split()

        a.append( [ int(float(timestamp)), int(float(agent_index)), float(x), float(y) ] )

array=np.array(a)
#print(array)
#print(array[np.lexsort((array[:,0], array[:,1] )) ] )

#first sort by timestamp, then secondary sort by ID
sorted_list = array[np.lexsort((array[:,0], array[:,1] )) ]

#since there are 10 id, we can split by 10  (splitting it according to id)
splitted_id_list = np.array(np.split(sorted_list,10))
print(splitted_id_list)
print(splitted_id_list.shape)

#since everything is already in order, remove timestamp and id field
trimmed_list = splitted_id_list[:,:,2:]
print(trimmed_list)

print(repr(trimmed_list))
