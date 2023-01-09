from pyspark import SparkContext
import os
import json
import sys
import time


# Local
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#Path
test_ReviewPath = r"C:\Users\ankus\PycharmProjects\CSCI553_Assignment-1\yelp_dataset\test_review.json"
businessPath = r"C:\Users\ankus\PycharmProjects\CSCI553_Assignment-1\yelp_dataset\business.json"
output_path_qa=r"output_task3_qa.txt"
output_path_qb=r"output_task3_qb.json"
#n_partitions=sys.argv[3]


sc = SparkContext('local[*]', 'Task_Exploration')

Overall_Loading_Start = time.time()
test_ReviewRDD = sc.textFile(test_ReviewPath).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['stars']))
businessRDD = sc.textFile(businessPath).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['city']))

CombinedRDD = test_ReviewRDD.join(businessRDD)  # (business_id,(star,city))

sample_1 = CombinedRDD.map(lambda x: (x[1][1], x[1][0]))  # City -> Review
sample_2 = CombinedRDD.map(lambda x: (x[1][1], 1))  # City -> 1

sample_1 = sample_1.reduceByKey(lambda x, y: x + y)  # City Review Sum
# for i in sample_1.collect():
#    print(i)

sample_2 = sample_2.groupByKey().mapValues(len)  # Total ratings for specific city
# for i in sample_2.collect():
#    print(i)

sample_final = sample_2.join(sample_1)  # (City,Rating_Sum,Number_of_ratings)

# for i in sample_final.collect():
#    print(i)


Final_Solution = sample_final.map(lambda x: (x[0], x[1][1] / x[1][0]))
# for i in Final_Solution.collect():
#    print(i[0],i[1])

Overall_Loading_End = time.time()
Overall_Loading_time_Elapsed = Overall_Loading_End - Overall_Loading_Start

F_ans = {}

# Method1: Python time
Start_Python = time.time()
Final_Solution_list_Python = list(Final_Solution.collect())
Final_Solution_list_Python.sort(key=lambda x: (-x[1], x[0]))
for i in Final_Solution_list_Python[:10]:
    print(i[0], i[1])
# print(Final_Solution_list[:10])
End_Python = time.time()
F_Time_Python = End_Python - Start_Python
F_ans['m1'] = Overall_Loading_time_Elapsed + F_Time_Python

# Method2: Spark Sort
Start_Spark = time.time()
Final_Solution_count_Pyspark = Final_Solution.count()
print(Final_Solution_count_Pyspark)
Final_Solution_list_Pyspark = Final_Solution.takeOrdered(Final_Solution_count_Pyspark, key=lambda a: [-a[1], a[0]])
for i in Final_Solution_list_Pyspark[:10]:
    print(i[0], i[1])
# print(Final_Solution_list[:10])
End_Spark = time.time()
F_Time_Spark = End_Spark - Start_Spark
F_ans['m2'] = Overall_Loading_time_Elapsed + F_Time_Spark

# Store City and average ratings
with open(output_path_qa, 'w') as f:
    f.write("city,stars" + '\n')
    for wr in Final_Solution_list_Pyspark:
        f.write('%s,%s\n' % (wr[0], wr[1]))

# Reason
F_ans['reason'] = "With the help of previous comparison, we can discern that Pyspark is faster than python. This is because Pyspark enables us to execute our program parallely which improves the overall performance of the pyspark program."


# Store execution time information of Python and Spark
json_file = json.dumps(F_ans, indent=4)
with open(output_path_qb, 'w') as file:
    file.write(json_file)

# print("----------------", F_ans['m1'],F_ans['m2'])