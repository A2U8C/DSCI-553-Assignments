import csv
import os
import sys
import time
from pyspark import SparkContext
import math

start_time=time.time()
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

input_train_path = r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata\yelp_train.csv"
input_test_path=r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata\yelp_val.csv"
output_path_val = r"Ass-3_output_task2.csv"


similarity_overall={}
sc = SparkContext('local[*]', 'Assignment_3_Tasks')



training_RDD=sc.textFile(input_train_path)
header=training_RDD.collect()[0]
training_RDD=training_RDD.filter(lambda x:x!=header)
training_RDD=training_RDD.map(lambda x: x.split(','))


user_business_pair_RDD_ = training_RDD.map(lambda x:(x[0],x[1])).groupByKey().mapValues(set)    #(user,{business_ids})
user_business_pair_RDD_=user_business_pair_RDD_.collectAsMap()      #user:{business_ids}
business_user_pair_RDD_ = training_RDD.map(lambda x:(x[1],x[0])).groupByKey().mapValues(set)    #(business,{user})
business_user_pair_RDD_ =business_user_pair_RDD_.collectAsMap()     #business:{user_id}

user_sum_len_RDD = training_RDD.map(lambda x:(x[0],float(x[2]))).groupByKey().mapValues(lambda x: (sum(x),len(x))).collectAsMap()           #user_id:(sum(rating),count_raters)
business_user_rating_group_RDD = training_RDD.map(lambda x:((x[1],x[0]),float(x[2]))).collectAsMap()                #(business_id,user_id):rating
business_sum_len_RDD = training_RDD.map(lambda x:(x[1],float(x[2]))).groupByKey().mapValues(lambda x: (sum(x),len(x))).collectAsMap()       #business_id:(sum(rating),count_raters)


all_rating_average=training_RDD.map(lambda x:(1,float(x[2]))).groupByKey().mapValues(lambda x:(sum(x),len(x))).collect()
default_rating_set=float(all_rating_average[0][1][0]/all_rating_average[0][1][1])
print(default_rating_set)


testing_RDD=sc.textFile(input_test_path)

header2=testing_RDD.collect()[0]
testing_RDD=testing_RDD.filter(lambda x:x!=header2)
testing_RDD=testing_RDD.map(lambda x: x.split(','))
testing_business_user_RDD=testing_RDD.map(lambda x:(x[0],x[1]))         #(user_id,businessId)




def prediction_func(x):
    x_user_id=x[0]
    x_business_id=x[1]
    similarity_local_dictionary={}
    if x_user_id not in user_business_pair_RDD_ and x_business_id in business_user_pair_RDD_:
        pred_rating_bus=float(business_sum_len_RDD[x_business_id][0]/business_sum_len_RDD[x_business_id][1])        #new user but we can use past business data for pred_ratings
        return tuple([x_user_id,x_business_id,pred_rating_bus])
    elif x_user_id not in user_business_pair_RDD_ and x_business_id not in business_user_pair_RDD_:
        return tuple([x_user_id, x_business_id, default_rating_set])  # For complete coldstart, no user data and business data
    elif x_user_id in user_business_pair_RDD_ and x_business_id not in business_user_pair_RDD_:
        pred_rating_user=float(user_sum_len_RDD[x_user_id][0]/user_sum_len_RDD[x_user_id][1])         #new business but we can use past user data for pred_ratings
        return tuple([x_user_id, x_business_id, pred_rating_user])

    all_business_by_x_user=user_business_pair_RDD_[x_user_id]         #all business which x_user_id rated
    all_users_for_business=business_user_pair_RDD_[x_business_id]       #all users who rated x_business_id
    if x_business_id in all_business_by_x_user:
        return tuple([x_user_id,x_business_id,business_user_rating_group_RDD[(x_user_id,x_business_id)]])

    average_business_ratings=float(business_sum_len_RDD[x_business_id][0]/business_sum_len_RDD[x_business_id][1])   #Avg rating for x_business_id
    for i in all_business_by_x_user:
        temp_index=tuple(sorted([i,x_business_id]))
        if temp_index in similarity_overall.keys():
            similarity_local_dictionary[i]=similarity_overall[temp_index]
            continue
        else:
            prod_business1_business2=0
            sum_squares_business1=0
            sum_squares_business2=0
            average_rating_business_i=float(business_sum_len_RDD[i][0]/business_sum_len_RDD[i][1])  #Average rating for business_i for which we are calulating W(i,x_business)


            all_user_business_i=business_user_pair_RDD_[i]
            intersection_users=set(all_user_business_i & all_users_for_business)
            #intersection_users=set(all_user_business_i).intersection(set(all_users_for_business))
            if len(intersection_users)==0:          #Eventhough no same users are rating businesses, we will still take the impact into consideration as small value
                division_similarity=float(average_business_ratings/average_rating_business_i)
                if division_similarity>1:
                    division_similarity=float(1/division_similarity)
                similarity_local_dictionary[i]=division_similarity
                similarity_overall[temp_index]=division_similarity
                continue
            for j in intersection_users:
                rating_user_i = business_user_rating_group_RDD[(i, j)]-average_rating_business_i
                rating_x_user_id = business_user_rating_group_RDD[(x_business_id, j)] - average_business_ratings
                prod_business1_business2 += (rating_user_i * rating_x_user_id)
                sum_squares_business1 += (rating_user_i * rating_user_i)
                sum_squares_business2 += (rating_x_user_id * rating_x_user_id)
            if prod_business1_business2==0:
                pearson_value=0
            else:
                pearson_value=float(prod_business1_business2/(float(float(math.sqrt(sum_squares_business1))*float(math.sqrt(sum_squares_business2)*1))))

            similarity_local_dictionary[i]=pearson_value
            similarity_overall[temp_index]=pearson_value

    for i in similarity_local_dictionary.keys():
        #if similarity_local_dictionary[i]>0:            #Charging small ratings
        if similarity_local_dictionary[i] < 0:
            print(similarity_local_dictionary[i])
        similarity_local_dictionary[i]=float(similarity_local_dictionary[i] * pow(abs(similarity_local_dictionary[i]),1.5))
        similarity_overall[tuple(sorted([i,x_business_id]))]=similarity_local_dictionary[i]

    similarity_local_dictionary_items=sorted(similarity_local_dictionary.items(),key=lambda x:x[1],reverse=True)
    total_num=len(similarity_local_dictionary)

    average_sum=0
    average_weight_elements=0

    for pair in similarity_local_dictionary_items:
        if similarity_local_dictionary[pair[0]]>=0:
            total_num-=1
            average_sum+=business_user_rating_group_RDD[pair[0],x_user_id]*similarity_local_dictionary[pair[0]]
            average_weight_elements+=abs(similarity_local_dictionary[pair[0]])
        if total_num==0:
            break

    if average_sum==0:
        pred_rate=0
    else:
        pred_rate=float(average_sum/average_weight_elements)

    if pred_rate==0:
        def_business_rat=float(business_sum_len_RDD[x_business_id][0]/business_sum_len_RDD[x_business_id][1])
        def_user_rat=float(user_sum_len_RDD[x_user_id][0]/user_sum_len_RDD[x_user_id][1])
        average_business_user_def=float((def_user_rat+def_business_rat)/2)
        return tuple([x_user_id,x_business_id,average_business_user_def])

    pred_rate=min(pred_rate,5)
    return tuple([x_user_id, x_business_id, pred_rate])



predicted_ratings_test=testing_RDD.map(lambda x: prediction_func(x))
#print(predicted_ratings_test.collect())


with open(output_path_val,'w',newline="") as fileoutput:
    filewriter=csv.writer(fileoutput)
    filewriter.writerow(['user_id','business_id','prediction'])
    filewriter.writerows(predicted_ratings_test.collect())


print("Duration: ",time.time()-start_time)

output_file=open(output_path_val,"r")
reference_file=open(input_test_path,"r")
n=0
rmse=0
while(True):
    output_file_line=output_file.readline()
    reference_file_line=reference_file.readline()
    if "user_id" in reference_file_line:
        continue
    if reference_file_line=="":
        break
    if not output_file_line and not reference_file_line:
        break
    n+=1
    rmse+=math.pow(float(output_file_line.split(",")[2][:-1])-float(reference_file_line.split(",")[2][:-1]),2)


rmse=math.sqrt(rmse/n)

print(rmse)