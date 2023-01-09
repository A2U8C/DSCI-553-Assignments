from pyspark import SparkContext
import os
import json
import sys
import time
import math
import collections
import itertools
import copy
import operator
import ast

start_time = time.time()


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


#Path
input_path_val = r"C:\Users\ankus\PycharmProjects\DSCI553_Assignment2\publicdata\small2.csv"
output_path_val=r"output_file_2.json"

support=10



sc = SparkContext('local[*]', 'SimulatedData')
sc.setLogLevel("ERROR")
final_final = list()
final_dict = dict()
final_final2 = list()
final_dict2 = dict()


def get_frequent_itemset_singleton(partition, par_threshold):
    new_s = set()
    final = set()
    for i in partition:
        temp = set(i)
        new_s = new_s.union(temp)
    for i in new_s:
        cnt = 0 * 0
        for j in partition:
            if i in j:
                cnt = (cnt + 1 + 0) * 1
        if cnt >= par_threshold:
            final.add(tuple({i}))
    # print(final)
    return final


def generate_candidates(current, k):
    res = set()
    temp = list(current)
    cnt = 0
    for i in range(0, len(temp)):
        con = tuple()
        for j in range(i + 1, len(temp)):
            # print(temp[i],temp[j])
            con = con + temp[i] + temp[j]
            # print(con)
            cnt = cnt + 1
            # print(tuple(set(con)),len(tuple(set(con))),cnt)
            if len(tuple(set(con))) == k:
                a = tuple(sorted(tuple(set(con))))
                # print(a,type(a),type(res))
                if a in res:
                    con = tuple()
                else:
                    res.add(a)
                    con = tuple()
            else:
                con = tuple()
    # print("Res",res)
    # print(set(res))
    return res


def candidate_filter(partition, new_current, par_threshold):
    res = set()
    # print(new_current)
    # print(partition)
    # print()
    for i in new_current:
        cnt = 0
        for j in partition:
            # print(set(i),set(j),cnt)
            # if i.issubset(frozenset(j)):
            if set(i).issubset(set(j)):
                cnt = cnt + 1
        if cnt >= par_threshold:
            res.add(i)
    return res


def apriori(partition, par_threshold):
    res = dict()
    flagsingle = 1
    if flagsingle == 1:
        single = get_frequent_itemset_singleton(partition, par_threshold)
        flagsingle = 0
    length = 1
    print(single, "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", len(single))
    while True:
        new_current = generate_candidates(single, length + 1)
        res[length] = single
        single = candidate_filter(partition, new_current, par_threshold)
        length = length + 1
        print(single,"\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", len(single))
        if len(single) == 0:
            break
    return res


def generate_output(temp_candidates):
    final_res = set()
    for i in temp_candidates:
        for j in temp_candidates[i]:
            final_res.add(j)
    return final_res


def partition_threshold(x, support, dataset):
    partition = copy.deepcopy(list(x))
    par_threshold = math.ceil(support * len(list(partition)) / dataset)

    print("...........................................", par_threshold)
    return par_threshold, partition


def findcandidates(x, support, dataset):
    par_threshold, partition = partition_threshold(x, support, dataset)
    temp_candidates = apriori(partition, par_threshold)
    output = generate_output(temp_candidates)
    return output


textRDD = sc.textFile(input_path_val).cache()
header = textRDD.first()
textRDD = textRDD.filter(lambda x: x != header)
bucketRDD = textRDD.map(lambda x: x.split(",")).groupByKey().mapValues(set).map(lambda x: list(x[1]))




dataset = bucketRDD.count()
candidates = bucketRDD.mapPartitions(lambda x: findcandidates(x, int(support), dataset)).distinct().collect()
print("******************************************************************",candidates,"\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^",len(candidates))
final_final.append(candidates)
for i in final_final:
    for j in i:
        ll = sorted(set(j))
        final_dict.setdefault(len(list(j)), []).append(ll)


def filterfreqitems(x, support, candidates):
    res = dict()
    partition = copy.deepcopy(list(x))
    for i in candidates:
        cnt = 0
        for j in partition:
            if set(list(i)).issubset(set(j)):
                cnt = cnt + 1
            res[str(list(i))] = cnt

    return [(i, j) for i, j in res.items()]

print("/////////////////////final_final//////////////////",final_final)
print("/////////////////////final_dict//////////////////",final_dict)
final_dict = dict(sorted(final_dict.items()))
# print(final_dict)
itemset = bucketRDD.mapPartitions(lambda x: filterfreqitems(x, int(support), candidates))
itemset2 = itemset.reduceByKey(operator.add).filter(lambda x: x[1] >= int(support)).map(lambda x: x[0]).collect()


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",itemset2,"\n",len(itemset2))



final_final2.append(itemset2)
for i in final_final2:
    for j in i:
        j = ast.literal_eval(j)
        ll = sorted(set(j))
        final_dict2.setdefault(len(list(j)), []).append(ll)

final_dict2 = dict(sorted(final_dict2.items()))

f = open(output_path_val, 'w+')
f.write("Candidates:")
f.write("\n")
for key, value in final_dict.items():
    value.sort()
    flag = 0
    flag1 = 0
    for i in value:
        if len(i) == 1:
            if flag1 == 0:
                f.write("('" + str(i[0]) + "')")
                flag1 = 1
            else:
                f.write(",")
                f.write("('" + str(i[0]) + "')")
        else:
            if flag == 0:
                f.write(str(tuple(i)))
                flag = 1
            else:
                f.write(",")
                f.write(str(tuple(i)))
    f.write("\n")
    f.write("\n")

f.write("Frequent Itemsets:")
f.write("\n")
for key, value in final_dict2.items():
    value.sort()
    flag = 0
    flag1 = 0
    for i in value:
        if len(i) == 1:
            if flag1 == 0:
                f.write("('" + str(i[0]) + "')")
                flag1 = 1
            else:
                f.write(",")
                f.write("('" + str(i[0]) + "')")
        else:
            if flag == 0:
                f.write(str(tuple(i)))
                flag = 1
            else:
                f.write(",")
                f.write(str(tuple(i)))
    f.write("\n")
    f.write("\n")

end_time = time.time()
print("Duration:", end_time - start_time)