import pandas as pd
import math
import time
import xgboost
from sklearn.preprocessing import LabelEncoder
import csv
from pyspark import SparkContext
import sys
import os


start = time.time()


all_data_files=r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata"
training_data = pd.read_csv(all_data_files+"/yelp_train.csv")
input_test_path=r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata\yelp_val.csv"
output_path_val=r"Ass-3_output_task3.csv"

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'Assignment_3_Tasks')

testdata_data = pd.read_csv(input_test_path)
user_data_file=pd.read_json(all_data_files+"/user.json",lines=True,chunksize=80000)
business_data_file=pd.read_json(all_data_files+"/business.json",lines=True,chunksize=80000)



flag_val=0
for i in user_data_file:
    if flag_val==0:
        user_id_data=i
        flag_val=1
        continue
    user_id_data=user_id_data.append(i)


flag_val_2=0
for i in business_data_file:
    if flag_val_2==0:
        business_id_data=i
        flag_val_2=1
        continue
    business_id_data=business_id_data.append(i)



user_data=user_id_data[['user_id','average_stars','review_count','useful','fans']]
business_data=business_id_data[['business_id','stars','review_count']].rename(columns={'stars':'business_stars','review_count':'business_review_count'})


training_data_comb=pd.merge(training_data,user_data,on='user_id',how='inner')
training_data_comb=pd.merge(training_data_comb,business_data,on='business_id',how='inner')


d1={}
data_var={}
data_number_user_id={}
data_max_star={}
data_min_star={}



for i, j in training_data_comb.iterrows():
    if j["user_id"] not in data_var.keys():
        data_var[j["user_id"]]=pow((float(j["stars"])-float(j["average_stars"])),2)
        data_number_user_id[j["user_id"]]=1
    else:
        data_var[j["user_id"]] += pow((float(j["stars"]) - float(j["average_stars"])), 2)
        data_number_user_id[j["user_id"]] += 1

    if j["user_id"] not in data_max_star.keys():
        data_max_star[j["user_id"]]=j["stars"]
    else:
        if data_max_star[j["user_id"]]<j["stars"]:
            data_max_star[j["user_id"]] = j["stars"]
    if j["user_id"] not in data_min_star.keys():
        data_min_star[j["user_id"]]=j["stars"]
    else:
        if data_min_star[j["user_id"]]>j["stars"]:
            data_min_star[j["user_id"]] = j["stars"]


###################################################
# for i, content in training_data_comb.iterrows():
#     if (content['user_id'] not in d1):
#         data_number_user_id[content['user_id']] = 1
#         data_var[content['user_id']] = (content['stars'] - content['average_stars']) * (content['stars'] - content['average_stars']) * 1 + 0
#         data_max_star[content['user_id']] = content['stars']
#         data_min_star[content['user_id']] = content['stars']
#
#     else:
#
#         data_number_user_id[content['user_id']] = data_number_user_id[content['user_id']] + 1
#         data_var[content['user_id']] +=((content['stars'] - content['average_stars']) * (content['stars'] - content['average_stars']))
#         if (content['stars'] > data_max_star[content['user_id']]):
#             data_max_star[content['user_id']] = content['stars']
#         if (content['stars'] < data_min_star[content['user_id']]):
#             data_min_star[content['user_id']] = content['stars']


###################################################






new_temp_df=pd.DataFrame(columns=["user_id","user_variance","Max_rating","Min_rating"])
for i in data_var.keys():
    f_variance = float(data_var[i]/data_number_user_id[i])
    new_temp_df=new_temp_df.append({'user_id': i,'user_variance':f_variance,'Max_rating':data_max_star[i],'Min_rating':data_min_star[i]},ignore_index=True)

#print(new_temp_df)
training_data_comb=pd.merge(training_data_comb,new_temp_df,on='user_id',how='left')#how='left'


temp_train_data=training_data_comb
for i in temp_train_data.columns:
    if temp_train_data[i].dtype=='object':
        temp_pre=LabelEncoder()
        temp_pre.fit(list(temp_train_data[i].values))
        temp_train_data[i]=temp_pre.transform(list(temp_train_data[i].values))

#print(temp_train_data)

#ratings_val=temp_train_data.stars.values
ratings_val=temp_train_data["stars"].values
print(len(temp_train_data.stars.values))
print(len(list(temp_train_data["stars"].values)))
training_dataset=temp_train_data.drop(['stars'],axis=1)
training_dataset=training_dataset.drop(['user_id'],axis=1)
training_dataset=training_dataset.drop(['business_id'],axis=1)
training_dataset=training_dataset.values



model_prepare=xgboost.XGBRegressor(
        max_depth=5,
        min_child_weight=1,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=800
    )
model_prepare.fit(training_dataset,ratings_val)




testestdata_comb=pd.merge(testdata_data,user_data,on='user_id',how='left')#how='left'
testestdata_comb=pd.merge(testestdata_comb,business_data,on='business_id',how='left')#how='left'
testestdata_comb=pd.merge(testestdata_comb,new_temp_df,on='user_id',how='left')#how='left'


user_test_data_id=testestdata_comb['user_id'].values
business_test_data_id=testestdata_comb['business_id'].values

test_data=testestdata_comb

for i in test_data.columns:
    if test_data[i].dtype=='object':
        test_pre=LabelEncoder()
        test_pre.fit(list(test_data[i].values))
        test_data[i]=test_pre.transform(list(test_data[i].values))

test_data.fillna((-99),inplace=True)

testing_dataset=test_data.drop(['user_id'],axis=1)
testing_dataset=testing_dataset.drop(['business_id'],axis=1)
testing_dataset=testing_dataset.drop(['stars'],axis=1)
testing_dataset=testing_dataset.values


prediction_sol=model_prepare.predict(testing_dataset)

Predicted_solution_File=pd.DataFrame()
Predicted_solution_File['user_id']=user_test_data_id
Predicted_solution_File['business_id']=business_test_data_id
Predicted_solution_File['prediction']=prediction_sol

# print(user_test_data_id[0])
# Predicted_solution_File.to_csv(output_path_val,sep=',',encoding='utf-8',index=False)

temp_RDD_list=[]
for i in range(len(user_test_data_id)):
    temp_RDD_list.append((user_test_data_id[i],business_test_data_id[i],prediction_sol[i]))

temp_RDD=sc.parallelize(temp_RDD_list)
with open(output_path_val,'w',newline="",encoding='utf-8') as fileoutput:
    filewriter=csv.writer(fileoutput)
    filewriter.writerow(['user_id','business_id','prediction'])
    filewriter.writerows(temp_RDD.collect())

print("Duration:",(time.time() - start))

reference_file=open(input_test_path,"r")
output_file=open(output_path_val,"r")
n=0
rmse=0
while(True):
    output_file_line=output_file.readline()
    reference_file_line=reference_file.readline()
    if "user_id" in reference_file_line:
        continue
    if reference_file_line=="":
        break
    n+=1
    rmse+=math.pow(float(output_file_line.split(",")[2][:-1])-float(reference_file_line.split(",")[2][:-1]),2)
    if not output_file_line and not reference_file_line:
        break
rmse=math.sqrt(rmse/n)
print(rmse)


reference_file=open(input_test_path,"r")
output_file=open(output_path_val,"r")
n=0
rmse=0
test_dict={}
output_dict={}
while(True):
    l1=output_file.readline()
    l2=reference_file.readline()
    if "user_id" in l2:
        continue
    if l2=="":
        break
    if not l1 and not l2:
        break
    #print(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),float(l1.split(",")[2][:-1]), float(l2.split(",")[2][:-1]))
    test_dict[(l2.split(",")[0],l2.split(",")[1])] = float(l2.split(",")[2][:-1])
    output_dict[(l1.split(",")[0], l1.split(",")[1])] = float(l1.split(",")[2][:-1])
for k in output_dict:
    n+=1
    rmse += math.pow(output_dict[k] - test_dict[k], 2)
rmse=math.sqrt(rmse/n)
print(rmse)
print("**********************************",data_var['TibBhm-fbksozIDFD8wjPQ'])
