from pyspark import SparkContext
import os
from pyspark.sql import SparkSession
import itertools
import sys
from pyspark.sql import *
from graphframes import *


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


input_path_val = r"C:\Users\ankus\PycharmProjects\dsci544_Assignment-4\ub_sample_data.csv"
output_path_val = r"Assignment-4_output_Realtask1.txt"
threshold_val = 7

sc = SparkContext('local[*]', 'Assignment_4_Tasks')
sparkSession_variable = SparkSession(sc)

print(sc.version)

RDDFile = sc.textFile(input_path_val)
print(RDDFile.collect()[:10])
column_head=RDDFile.collect()[0]
print(column_head)
RDDFile=RDDFile.filter(lambda x: x!=column_head).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1]))
print(RDDFile.collect()[:10])

user_business_RDD=RDDFile.groupByKey().mapValues(list)          #(user,[businesses])
user_business_Dict=user_business_RDD.collectAsMap()             #user: [businesses]
print(user_business_RDD.collect()[:10])
all_users=user_business_RDD.map(lambda x:x[0]).collect()                  #[users]      #Nodes
print(all_users[:10])

user_pairs=[i for i in itertools.combinations(all_users,2)]           #
print(user_pairs[:5])
#Filtering
edges_user_list=[]
node_user_set=set()
for i in user_pairs:
    u_1=set(user_business_Dict[i[0]])
    u_2=set(user_business_Dict[i[1]])
    if len(u_1.intersection(u_2))>=threshold_val:
        edges_user_list.append(i)
        edges_user_list.append(tuple([i[1],i[0]]))
        node_user_set.add(i[0])
        node_user_set.add(i[1])

node_user_list=list(node_user_set)
edge_col_name=['src','dst']
node_col_name=['id']
edges_users=sc.parallelize(edges_user_list).toDF(edge_col_name)
nodes_users=sc.parallelize(node_user_list).map(lambda x:tuple([x])).toDF(node_col_name)      #Need to create tuple otherwise input is string error

g = GraphFrame(nodes_users, edges_users)

g.vertices.show()
g.edges.show()
g.degrees.show()

result = g.labelPropagation(maxIter=5)
#result.select("id", "label").show()
result_RDD=result.rdd.map(lambda x:(x[1],x[0])).groupByKey().mapValues(set).map(lambda x: (x[0],list(x[1]))).map(lambda x:sorted(x[1])).sortBy(lambda x:[len(x),x]).collect()
print(result_RDD)
with open(output_path_val,'w') as filewriter:
    for i in result_RDD:
        filewriter.write(str(i).strip('[]')+'\n')