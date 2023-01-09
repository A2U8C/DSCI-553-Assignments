from pyspark import SparkContext
import os
import json
import sys
import time


# Local
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#Path
input_path_val = r"C:\Users\ankus\PycharmProjects\CSCI553_Assignment-1\yelp_dataset\test_review.json"
output_path_val=r"output_task2.json"
#n_partitions=sys.argv[3]

F_ans={}
n_Cust_partitions=3

F_ans={"default":{}, "customized":{}}

sc = SparkContext("local[*]", "Task_Exploration")


#Task1: F
start1 = time.time()
RDDFile = sc.textFile(input_path_val).map(lambda x: json.loads(x))
top10_business_default=RDDFile.map(lambda x:(x['business_id'],1))
n_partitions_default=top10_business_default.getNumPartitions()
n_items_default=top10_business_default.glom().map(len).collect()
F_ans['default']["n_partition"]=int(n_partitions_default)
F_ans['default']["n_items"]=n_items_default
#print(n_items_default)
top10_business_default=top10_business_default.groupByKey().mapValues(len).takeOrdered(10, key=lambda a: [-a[1],a[0]])
top10_business_default=list(top10_business_default)
#print(top10_business_default)
end1 = time.time()
F_ans['default']["exe_time"]=end1-start1



#Customized partitions
start2 = time.time()
RDDFile2 = sc.textFile(input_path_val).map(lambda x: json.loads(x))
top10_business_customized=RDDFile2.map(lambda x:(x['business_id'],1)).partitionBy(n_Cust_partitions)
n_partitions_customized=top10_business_customized.getNumPartitions()
n_items_customized=top10_business_customized.glom().map(len).collect()
F_ans['customized']["n_partition"]=int(n_partitions_customized)
F_ans['customized']["n_items"]=n_items_customized
top10_business_customized=top10_business_customized.groupByKey().mapValues(len).takeOrdered(10, key=lambda a: [-a[1],a[0]])
top10_business_customized=list(top10_business_customized)
end2 = time.time()
F_ans['customized']["exe_time"]=end2-start2


#Store information into the output file using output_path_val
json_data = json.dumps(F_ans, indent = 4)
with open(output_path_val, 'w') as file:
    file.write(json_data)