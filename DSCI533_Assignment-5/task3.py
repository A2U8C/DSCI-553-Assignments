import random
from blackbox import BlackBox
import binascii
import sys
import csv
import time

start_time=time.time()
bx=BlackBox()

input_file="users.txt"
stream_size=int(100)
num_of_asks=int(30)
output_file="task3.csv"


counter=100
flag=0
history_list=[]

def fixed_sample_funct(stream_users):
    global flag
    global counter
    global history_list
    stream_users = stream_users[0:100]
    if flag==1:
        for i in stream_users:
            counter = counter + 1
            temp_prob = float(100.0 / counter)
            if random.random() < temp_prob:
                pos = random.randint(0, len(history_list) - 1)
                history_list.pop(pos)
                history_list.insert(pos, i)
    elif flag==0:
        flag=1
        history_list = stream_users
        # history_list=stream_users[0:100]
        # if len(stream_users)>100:
        #     for jk in stream_users[100:]:
        #         counter+=1
        #         temp_prob = float(100.0 / counter)
        #         if random.random() < temp_prob:
        #             pos = random.randint(0, len(history_list) - 1)
        #             history_list.pop(pos)
        #             history_list.insert(pos, jk)

    return counter,history_list

if __name__ == "__main__":
    csv_output_tuple=[]
    random.seed(553)
    for i in range(num_of_asks):
        stream_users=bx.ask(input_file,int(stream_size))
        if len(set(stream_users)) != stream_size:
            while len(set(stream_users)) != stream_size:
                stream_users = bx.ask(input_file, int(stream_size))
        count_val,history_i=fixed_sample_funct(stream_users)
        csv_output_tuple.append([count_val,history_i[0],history_i[20],history_i[40],history_i[60],history_i[80]])

    with open(output_file, 'w',newline="") as csv_file:
        filewriter = csv.writer(csv_file)
        filewriter.writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
        filewriter.writerows(csv_output_tuple)