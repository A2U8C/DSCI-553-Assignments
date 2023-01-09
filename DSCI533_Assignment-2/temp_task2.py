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
import csv

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Path
input_file_path = r"C:\Users\ankus\PycharmProjects\DSCI553_Assignment2\publicdata\ta_feng_all_months_merged.csv"
output_file_path = r"output_file_3.txt"
output_file_path1 = 'preprocess.csv'
support_threshold=50
filter=20
start_time = time.time()
sc = SparkContext('local[*]','PreprocessingData')
sc.setLogLevel("ERROR")
fields = ['DATE-CUSTOMER_ID','PRODUCT_ID']


def candidate_filter(partition, new_current, par_threshold):
    res = set()
    i = 1
    if (len(new_current) * i) == (0 * i):
        return res
    cnt = dict()

    # cnt = collections.defaultdict(int)

    for i in partition:
        print(i)
        for j in new_current:
            if j.issubset(i):
                if j in cnt:
                    cnt[j] = cnt[j] + 1
                else:
                    cnt[j] = 1
    # print(cnt)
    for i in cnt:
        # for j in cnt[i]:
        if cnt[i] >= par_threshold:
            res.add(i)
    return res


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
            final.add(frozenset({i}))
    return final


def generate_candidates(current, k):
    res = set()
    # print(current)
    for i in current:
        for j in current:
            # print(i,j)
            con = i.union(j)
            # print(con)
            if len(con) == k:
                res.add(con)
    return res





# def generate_candidates(current,k):
# 	res = set()
# 	temp = list(current)
# 	# print(temp)
# 	cnt = 0
# 	for i in range(0,len(temp)):
# 		con = tuple()
# 		for j in range(i+1,len(temp)):
# 			# print(temp[i],temp[j])
# 			con = con + temp[i] + temp[j]
# 			# print(con)
# 			cnt = cnt + 1
# 			# print(tuple(set(con)),len(tuple(set(con))),cnt)
# 			if len(tuple(set(con))) == k:
# 				a = tuple(sorted(tuple(set(con))))
# 				# print(a,type(a),type(res))
# 				if a in res:
# 					con = tuple()
# 				else:
# 					res.add(a)
# 					con = tuple()
# 			else:
# 				con = tuple()
# 		# print("Res",res)
# 	# print(set(res))
# 	return res


def apriori(partition, par_threshold):
    res = dict()
    flagsingle = 1
    if flagsingle == 1:
        single = get_frequent_itemset_singleton(partition, par_threshold)
        flagsingle = 0
    print("***************************************", len(single))

    length = 1
    while True:
        new_current = generate_candidates(single, length + 1)
        res[length] = single
        #single = candidate_filter(partition, new_current, par_threshold)
        res2 = set()
        i = 1

        cnt = dict()

        # cnt = collections.defaultdict(int)

        for i in partition:
            for j in new_current:
                if j.issubset(i):
                    if j in cnt:
                        cnt[j] = cnt[j] + 1
                    else:
                        cnt[j] = 1
        # print(cnt)
        for i in cnt:
            # for j in cnt[i]:
            if cnt[i] >= par_threshold:
                res2.add(i)
        single=res2
        length = length + 1
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
    return par_threshold, partition


def findcandidates(x, support, dataset):
    par_threshold, partition = partition_threshold(x, support, dataset)
    temp_candidates = apriori(partition, par_threshold)
    output = generate_output(temp_candidates)
    return output


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


textRDD = sc.textFile(input_file_path).map(lambda x: x.split(","))
textRDD = textRDD.map(lambda x: [x[0].strip('"') + "-" + x[1].strip('"'), x[5].strip('"')])
print(textRDD.collect())
#header = textRDD.first()
header = textRDD.take(1)[0]
print(header)
#textRDD = textRDD.filter(lambda x: x != header)

final_final = list()
final_dict = dict()
final_final2 = list()
final_dict2 = dict()
'''
bucketRDD = textRDD.map(lambda x: [x.split(',')[0].strip('"'), x.split(',')[1].strip('"').lstrip('0'),x.split(',')[5].strip('"').lstrip('0')]).map(lambda x: [x[0] + '-' + x[1], x[2]]).collect()
with open(output_file_path1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(bucketRDD)

textRDD = sc.textFile(output_file_path1).cache()

header = textRDD.first()'''
textRDD = textRDD.filter(lambda x: x != header)
bucketRDD = textRDD.map(lambda x:(x[0],x[1])).groupByKey().filter(lambda x: len(x[1]) > int(filter)).map(lambda x:list(set(x[1])))

# print(bucketRDD)

num_basket = bucketRDD.count()
# # print(dataset)
start2 = time.time()
candidatesRDD = bucketRDD.mapPartitions(
    lambda x: findcandidates(x, int(support_threshold), num_basket)).distinct().collect()

print("candidates",candidatesRDD,"\n",len(candidatesRDD))


# candidatesRDD = bucketRDD.mapPartitions(SON_1).distinct().collect()
# candidates = candidatesRDD.collect()
# print(candidates)
final_final.append(candidatesRDD)
# print(final_final)
for i in final_final:
    for j in i:
        # print(j)
        ll = sorted(set(j))
        # print(ll)
        final_dict.setdefault(len(list(j)), []).append(ll)
final_dict = dict(sorted(final_dict.items()))
# print(final_dict)


itemset = bucketRDD.mapPartitions(lambda x: filterfreqitems(x, int(support_threshold), candidatesRDD)).reduceByKey(
    operator.add).filter(lambda x: x[1] >= int(support_threshold)).map(lambda x: x[0]).collect()
# itemset2 = itemset.reduceByKey(operator.add).filter(lambda x: x[1] >= int(support_threshold)).map(lambda x: x[0]).collect()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",len(itemset))

final_final2.append(itemset)
# print(final_final2)
for i in final_final2:
    for j in i:
        j = ast.literal_eval(j)
        ll = sorted(set(j))
        final_dict2.setdefault(len(list(j)), []).append(ll)

final_dict2 = dict(sorted(final_dict2.items()))

f = open(output_file_path, 'w+')
f.write("Candidates:")
f.write("\n")
for key, value in final_dict.items():
    value.sort()
    flag = 0
    flag1 = 0
    for i in value:
        # print(len(i))
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
        # print(len(i))
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

