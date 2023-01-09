from blackbox import BlackBox
import binascii
import random
import sys
import csv
import time

start_time=time.time()

bx=BlackBox()


input_file="users.txt"
stream_size=int(300)
num_of_asks=int(30)
output_file="task2.csv"

p = 1342757201
global_filter_bit_array_n=sys.maxsize-1
hash_functions_n=100
windows_n=5
ground_sum=0
size_var=int(hash_functions_n/windows_n)
estimate_val_sum=0

def myhashs(s):
    result=[]
    for f in range(hash_functions_n):
        a = random.randint(1, sys.maxsize - 1)
        b = random.randint(0, sys.maxsize - 1)
        x=int(binascii.hexlify(s.encode('utf8')),16)
        temp_val = (((a*x)+b) % p) % global_filter_bit_array_n
        result.append(temp_val)
    return result


def flajolet_martin_algo(stream_users):
    global estimate_val_sum
    global ground_sum
    max_zeroes_trailing=[0 for kl in range(hash_functions_n)]
    ground_truth_n=len(stream_users)
    for i in stream_users:
        x=myhashs(i)
        bin_x=[bin(k) for k in x]
        trailing_zeroes=[len(bin_x[j])-len(bin_x[j].rstrip("0")) for j in range(len(bin_x))]
        max_zeroes_trailing=[max(x,y) for x,y in zip(max_zeroes_trailing,trailing_zeroes)]

    two_factor=sorted([pow(2,r) for r in max_zeroes_trailing])
    sub_two_list = [two_factor[size_var * i:size_var * (i + 1)] for i in range(windows_n)]
    average_two_factor = [(sum(k) / len(k)) for k in sub_two_list]
    estimated_value= round(average_two_factor[int(len(average_two_factor) / 2)])
    estimate_val_sum+=estimated_value
    ground_sum+=ground_truth_n
    return ground_truth_n,estimated_value



csv_output_tuple=[]
counter=0
for i in range(num_of_asks):
    stream_users=bx.ask(input_file,int(stream_size))
    ground_truth_n_i,estimated_value_i=flajolet_martin_algo(set(stream_users))
    csv_output_tuple.append(tuple([counter,ground_truth_n_i,estimated_value_i]))
    counter+=1

with open(output_file,'w',newline="") as csv_file:
	filewriter = csv.writer(csv_file)
	filewriter.writerow(["Time","Ground Truth","Estimation"])
	filewriter.writerows(csv_output_tuple)

print(float(estimate_val_sum/ground_sum))
print("Duration: ",time.time()-start_time)




