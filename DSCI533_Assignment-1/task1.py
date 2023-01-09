from pyspark import SparkContext
import os
import json
import sys


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

F_ans={}

#Path
input_path_val = r"C:\Users\ankus\PycharmProjects\CSCI553_Assignment-1\yelp_dataset\test_review.json"
output_path_val=r"output_file.json"

sc = SparkContext('local[*]', 'Task_Exploration')

RDDFile = sc.textFile(input_path_val) #list of strings
RDDFile = RDDFile.map(lambda x: json.loads(x)) 	#Creates list of dictionary

#Task-1: A (The total number of reviews)
n_review=RDDFile.map(lambda x : x['review_id']).count()
F_ans['n_review']=n_review
#print(n_review)


#Task-1: B (The number of reviews in 2018)
#n_reviews_in_2018=RDDFile.map(lambda x:x['date'][0:4]).filter(lambda x: x=='2018').count()     #Doesn't take MM-DD-YYYY date format
n_reviews_in_2018=RDDFile.map(lambda a:a['date'].split("-")).filter(lambda x: '2018' in x).count()
#n_review_2018=RDDFile.filter(lambda a:'2018' in a['date'].split("-")).count()
F_ans['n_review_2018']=n_reviews_in_2018
#print(n_reviews_in_2018)


#Task-1: C (The  number of distinct users who wrote reviews)
n_user=RDDFile.map(lambda y: y['user_id']).distinct()
n_user=n_user.count()
F_ans['n_user']=n_user
#print(n_user)


#Task-1: D (The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote)
users_by_no_of_reviews = RDDFile.map(lambda x: [x['user_id'],1] ).reduceByKey(lambda a,b : a+b).sortBy(keyfunc=lambda a:[ -a[1], a[0] ]).take(10)
list_users_by_no_of_reviews=list(map(list,users_by_no_of_reviews))
F_ans['top10_user']=list_users_by_no_of_reviews
#print(users_by_no_of_reviews)


#Task-1: E (The number of distinct businesses that have been reviewed)
n_business=RDDFile.map(lambda x: x['business_id']).distinct().count()
F_ans['n_business']=n_business
#print(n_business)


#Task-1: F (The top 10 businesses that had the largest numbers of reviews and the number of reviews they had)
top10_business=RDDFile.map(lambda x:(x['business_id'],1)).reduceByKey(lambda a,b:a+b).sortBy(keyfunc=lambda a: [ -a[1], a[0]]).take(10)
top10_business_list=list(map(list,top10_business))
F_ans['top10_business']=top10_business_list
#print(top10_business_list)


#Store information into the output file using output_path_val
json_data = json.dumps(F_ans, indent = 4)
with open(output_path_val, 'w') as file:
    file.write(json_data)
