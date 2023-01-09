from pyspark import SparkContext
import os
import sys
import math
import time

start_time = time.time()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#Path
input_path_val = r"C:\Users\ankus\PycharmProjects\DSCI553_Assignment2\publicdata\small2.csv"
output_path_val=r"output_file.txt"
support=4
case_no=1
sc = SparkContext('local[*]', 'Task_Exploration')


def create_Single_element(p_threshold, cluster):
    dict_elem={}
    final_elem=[]
    for i in cluster:
        temp_el=set(i)
        for j in temp_el:
            if j not in dict_elem:
                dict_elem[j]=0
            dict_elem[j]+=1
    for k in dict_elem.keys():
        if dict_elem[k]>=p_threshold:
            final_elem.append([k])
    return final_elem

def Candidate_list_Generator(element_list,p_threshold,pair_size):
    ans=[]
    if pair_size==2:
        for i in range(len(element_list)):
            temp=[]
            for j in range(i+1,len(element_list)):
                temp=list(set(element_list[i]+element_list[j]))
                temp=sorted(temp)
                ans.append(temp)
    else:
        #for i in element_list:
        #    temp=[]
        #    for j in element_list:
        #        if i is not j:
        #            if i[:-1] == j[:-1]:
        #                temp = list(set(i + j))
        #                if temp not in ans:
        #                    ans.append(temp)
        res=set()
        for i in range(len(element_list)):
            con = []
            for j in range(i + 1, len(element_list)):
                con = element_list[i] + element_list[j]
                if len(list(set(con))) == pair_size:
                    a = tuple(sorted(list(set(con))))
                    if a not in ans:
                        res.add(a)
                con = []
        ans=list(res)
    return ans

def apriori_Algo(p_threshold,cluster_list):
    single_el_list=create_Single_element(p_threshold,cluster_list)
    #print(single_el_list,"\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", len(single_el_list))
    candidate_dict={}
    pair_size=1
    while len(single_el_list)!=0:
        candidate_dict[pair_size] = single_el_list
        pair_size+=1
        single_el_list=Candidate_list_Generator(single_el_list,p_threshold,pair_size)
        final_single_el_list=[]
        for i in single_el_list:
            count=0
            for j in cluster_list:
                if set(i).issubset(set(j)):
                    count+=1
            if count>=p_threshold:
                final_single_el_list.append(i)

        single_el_list=final_single_el_list
        #print(single_el_list,"\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", len(single_el_list))
        #print("...........................................",pair_size,single_el_list )
    #print(candidate_dict,pair_size)
    return candidate_dict

def create_candidates_set(x,n_size_data,threshold):
    individual_cluster_list = list(x)
    p_threshold = math.ceil(threshold*(len(individual_cluster_list) / n_size_data))
    set_after_apriori=apriori_Algo(p_threshold,individual_cluster_list)
    #print(".....................set_after_apriori......................",set_after_apriori)
    Fans=set()
    for el_key in set_after_apriori.keys():
        for each_val in set_after_apriori[el_key]:
            #print("--------------------------------each_val------------------------------------------------",tuple(each_val))
            Fans.add(tuple(each_val))
    #print(len(Fans),"/////////////////////////////////////////////////////////////////////////")
    return Fans

RDDFile = sc.textFile(input_path_val)
print(len(RDDFile.collect()))
print(RDDFile.collect())



column_head=RDDFile.take(1)[0]
print(column_head)


RDDFile=RDDFile.filter(lambda x: x!=column_head)

if int(case_no) == 1:
    RDDFile = RDDFile.map(lambda x:x.split(",")).map(lambda x:(x[0],x[1])).groupByKey().mapValues(set).map(lambda x:list(set(x[1])))
    #print(RDDFile.collect())
elif int(case_no) == 2:
    RDDFile = RDDFile.map(lambda x:x.split(",")).map(lambda x: (x[1],x[0])).groupByKey().mapValues(set).map(lambda  x:list(set(x[1])))


n_size_data = RDDFile.count()
print(n_size_data)
#candidates=RDDFile.partitionBy(2).map(lambda x: create_candidates_set(x, n_size_data)).distinct()
candidates=RDDFile.mapPartitions(lambda x: create_candidates_set(x, n_size_data, support)).distinct().collect()
print("candidates\n","^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",len(set(candidates)))



def Phase_two_mapcandidate(x,candidates):
        individual_cluster_list = list(x)
        Fans = dict()
        for el_candidate in candidates:
            count=0
            for each_el_x in individual_cluster_list:
                if set(el_candidate).issubset(set(each_el_x)):
                    count += 1
            Fans[el_candidate]=count
        Fans_List=[(candidate_el,Fans[candidate_el]) for candidate_el in Fans.keys()]
        return Fans_List

new_candidates_map_two=RDDFile.mapPartitions(lambda x:Phase_two_mapcandidate(x,candidates))
#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",new_candidates_map_two.collect())
new_candidate_reduce_two= new_candidates_map_two.reduceByKey(lambda a,b : a+b).filter(lambda x: x[0] if x[1] >= int(support) else None).collect()
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@","\n",len(new_candidate_reduce_two))

final_candidate_reduce_first=dict()
for i in candidates:
    if len(i) not in final_candidate_reduce_first:
        final_candidate_reduce_first[len(i)]=[]
    if sorted(i) not in final_candidate_reduce_first[len(i)]:
        final_candidate_reduce_first[len(i)].append(sorted(i))
        final_candidate_reduce_first[len(i)]=sorted(final_candidate_reduce_first[len(i)])

final_candidate_reduce_dict=dict()
for i in new_candidate_reduce_two:
    #print(i)
    if len(i[0]) not in final_candidate_reduce_dict:
        final_candidate_reduce_dict[len(i[0])]=[]
    if sorted(i[0]) not in final_candidate_reduce_dict[len(i[0])]:
        final_candidate_reduce_dict[len(i[0])].append(sorted(i[0]))
        final_candidate_reduce_dict[len(i[0])]=sorted(final_candidate_reduce_dict[len(i[0])])

#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",final_candidate_reduce_first)
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",final_candidate_reduce_dict)

#======================================Write========================================================

def printing_function(value):
    printFlag, printFlag_1 = 0, 0
    for i in value:
        if len(i) == 1: #To avoid "," in i
            if printFlag_1 == 0:
                f.write("('" + str(i[0]) + "')")
                printFlag_1 = 1
            else:
                f.write(",")
                f.write("('" + str(i[0]) + "')")
        else:
            if printFlag == 0:
                f.write(str(tuple(i)))
                printFlag = 1
            else:
                f.write(",")
                f.write(str(tuple(i)))
    f.write("\n\n")

f = open(output_path_val, 'w+')
f.write("Candidates:\n")
for key in sorted(final_candidate_reduce_first.keys()):
    value=final_candidate_reduce_first[key]
    #value.sort()
    printing_function(value)

f.write("Frequent Itemsets:\n")
for key in sorted(final_candidate_reduce_dict.keys()):
    value = final_candidate_reduce_dict[key]
    #value.sort()
    printing_function(value)

end_time = time.time()
print("Duration:", end_time - start_time)

