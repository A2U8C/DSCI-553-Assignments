from blackbox import BlackBox
import binascii
import random
import sys
import csv
import time

start_time=time.time()
bx=BlackBox()

input_file="users.txt"
stream_size=int(100)
num_of_asks=int(30)
output_file="task1.csv"


global_filter_bit_array_n=69997
global_filter_bit_array=[0 for i in range(global_filter_bit_array_n)]


a = random.sample(range(1, sys.maxsize - 1), 8)
b = random.sample(range(0, sys.maxsize - 1), 8)
p=[70981, 70991, 70997, 70999, 71011, 71023, 71039, 71059]


hash_functions_n=len(a)
userID_set=set()


def myhashs(s):
    result=[]
    for f in range(hash_functions_n):
        x=int(binascii.hexlify(s.encode('utf8')),16)
        # print(x)
        temp_val=((a[f]*x+b[f])%p[f])%global_filter_bit_array_n
        result.append(temp_val)
    return result

def bloom_filter_func(stream_users):
    global global_filter_bit_array
    global userID_set
    tp_count = 0
    fp_count = 0
    for i in stream_users:
        x=myhashs(i)
        for j in x:
            if global_filter_bit_array[j]==1:
                flag=1
            else:
                flag=0
                break
        if flag==1:
            if i not in userID_set:
                fp_count+=1
            else:
                tp_count+=1
        else:
            for k in x:
                global_filter_bit_array[k] = 1
        userID_set.add(i)
    if (tp_count+fp_count)>0:
        fp_value = float(fp_count / (tp_count + fp_count))
    else:
        fp_value = float(0)
    return fp_value


result_count_fpr=[]
counter=0
for i in range(num_of_asks):
    stream_users=bx.ask(input_file,int(stream_size))
    if len(set(stream_users)) != stream_size:
        while len(set(stream_users)) != stream_size:
            stream_users = bx.ask(input_file, int(stream_size))
    fp_value_i=bloom_filter_func(stream_users)
    result_count_fpr.append([counter,fp_value_i])
    counter+=1


with open(output_file,'w',newline="") as csv_file:
	filewriter = csv.writer(csv_file)
	filewriter.writerow(["Time","FPR"])
	filewriter.writerows(result_count_fpr)

print("Duration: ",time.time()-start_time)