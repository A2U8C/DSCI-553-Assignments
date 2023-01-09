import os
import sys
from pyspark import SparkContext
import random
import itertools
import csv
import time

start_time=time.time()
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

input_path_val = r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata\yelp_train.csv"
output_path_val = r"Ass-3_output_task1.csv"

sc = SparkContext('local[*]', 'Assignment_3_Tasks')


similarity_threshold = 0.5
bands=50
rows=4

RDDFile = sc.textFile(input_path_val)
print(len(RDDFile.collect()))
#column_head=RDDFile.take(1)[0]
column_head=RDDFile.collect()[0]
print(column_head)
RDDFile=RDDFile.filter(lambda x: x!=column_head)


user_business_RDD=RDDFile.map(lambda x:x.split(','))    #user_id,business_id,rating
print(user_business_RDD.collect()[:15])


#business_RDD=user_business_RDD.map(lambda x: x[1]).distinct().zipWithIndex().map(lambda x:(x[1],x[0])).collectAsMap()  #index: business_id
business_RDD=user_business_RDD.map(lambda x: x[1]).distinct().zipWithIndex()  #[(business_id, index)]
business_RDD2_dict=business_RDD.map(lambda x:(x[1],x[0])).collectAsMap()        #index: business_id
business_RDD_dict=business_RDD.collectAsMap()    #business_id: index
#print(business_RDD2_dict)

user_RDD = user_business_RDD.map(lambda x: x[0]).distinct().zipWithIndex()  #[(user_id, index)]
user_RDD_dict = user_RDD.collectAsMap()     #user_id: index
#print(user_RDD_dict)

num_user=len(user_RDD_dict)
#print(num_user)


def generate_hash_sequence(x):
    ans = []
    Total_hashes = bands * (rows)
    for i in range(Total_hashes):
        a = random.randint(1, sys.maxsize - 1)
        b = random.randint(0, sys.maxsize - 1)
        #temp = (((a * x) + b) % (7427466391)) % num_user
        temp = (((a * x) + b) % (33489857205)) % num_user

        ans.append(temp)
    return tuple([x,ans])



user_sequence_hash=user_RDD.map(lambda f:generate_hash_sequence(f[1]))      #user_index,[all hash values]
#print(user_sequence_hash.collect()[0])
business_user_RDD_index=user_business_RDD.map(lambda x:(business_RDD_dict[x[1]],user_RDD_dict[x[0]])).groupByKey().mapValues(list)    #business_index,[user_index]
business_user_RDD_index_2=business_user_RDD_index.collectAsMap()        #business_index: [user_index]
#print(business_user_RDD_index_2)
#print(business_user_RDD_index.collect()[0])

user_business_RDD_index=user_business_RDD.map(lambda x:(user_RDD_dict[x[0]],business_RDD_dict[x[1]])).groupByKey().mapValues(list)      #user_index,[business_index]
#print(user_business_RDD_index.collect()[0])


Jaccard_RDD = user_business_RDD_index.leftOuterJoin(user_sequence_hash)     #(user_index,[hash_values],[business_ids])
Jaccard_RDD=Jaccard_RDD.map(lambda x:x[1])                                  #([hash_values],business_ids)
#print(Jaccard_RDD.collect()[:2])
Jaccard_RDD = Jaccard_RDD.flatMap(lambda x: [(i, x[1]) for i in x[0]])      #(hashed_value,[business_ids])
#print(Jaccard_RDD.collect()[0])

Jaccard_RDD=Jaccard_RDD.reduceByKey(lambda x,y:[min(i,j) for i,j in zip(x,y)])        #hashed_value, group of business_ids then return minimum between those business ids
#print(Jaccard_RDD.collect()[0])



filtered_Jaccard_RDD = Jaccard_RDD.flatMap(lambda x: [(tuple(k), x[0]) for k in [(j-1, hash(tuple(x[1][i-1:((i-1) + (rows))]))) for j,i in enumerate(range(1, len(x[1])+1, (rows)))]])

print(filtered_Jaccard_RDD.collect()[:10])
filtered_Jaccard_RDD=filtered_Jaccard_RDD.groupByKey().map(lambda x: list(x[1]))        #group by (j-1,hash of business_id) and values are hashed_user_id
#print(filtered_Jaccard_RDD.collect()[0])
filtered_Jaccard_RDD=filtered_Jaccard_RDD.filter(lambda x: len(x) > 1)          #hash of business_id must have more than or equal to 2 hashed_user_id
#print("-------------------------",filtered_Jaccard_RDD.collect()[:10])
filtered_Jaccard_RDD=filtered_Jaccard_RDD.flatMap(lambda x: [i for i in itertools.combinations(x, 2)])      #pair of 2 user_ids
#print(filtered_Jaccard_RDD.collect()[:10])



def similar_business_using_jaccard_Func(filtered_RDD,business_user_RDD_index_2,business_RDD2_dict):
    F_ans = list()
    temp_set_considered = set()
    for i in filtered_RDD:
        if i not in temp_set_considered:
            temp_set_considered.add(i)
            intersection_user_id = float(len(set(business_user_RDD_index_2[i[1]]).intersection(set(business_user_RDD_index_2[i[0]]))))
            union_user_id = float(float(len(set(business_user_RDD_index_2[i[1]]))) + float(len(set(business_user_RDD_index_2[i[0]]))) - intersection_user_id)
            sim = float(intersection_user_id/union_user_id)
            if sim >= similarity_threshold:
                temp=sorted([business_RDD2_dict[i[0]],business_RDD2_dict[i[1]]])
                F_ans.append([temp[0], temp[1], sim])
    F_ans.sort(key = lambda x: (x[0],x[1]))
    return F_ans

res = similar_business_using_jaccard_Func(set(filtered_Jaccard_RDD.collect()),business_user_RDD_index_2,business_RDD2_dict)



with open(output_path_val, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['business_id_1','business_id_2','similarity'])
    csvwriter.writerows(res)

print("Duration: ",time.time()-start_time)
