import pandas as pd
import math
import time
import xgboost
from sklearn.preprocessing import LabelEncoder
import csv
from pyspark import SparkContext
import sys
import os

start_time = time.time()
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

all_data_files=r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata"
input_test_path=r"C:\Users\ankus\PycharmProjects\DSCI_553_Assignment-3\publicdata\yelp_val.csv"
output_path_val_item=r"Ass-3_output_task4_item.csv"
output_path_val_model=r"Ass-3_output_task4_model.csv"

output_file_val=r"Ass-3_output_task4.csv"




sc = SparkContext('local[*]', 'Assignment_3_Tasks')

similarity_overall={}

def item_based_recommendation_function():
    input_train_path=all_data_files+"\yelp_train.csv"
    training_RDD = sc.textFile(input_train_path)
    header = training_RDD.collect()[0]
    training_RDD = training_RDD.filter(lambda x: x != header)
    training_RDD = training_RDD.map(lambda x: x.split(','))

    user_business_pair_RDD_ = training_RDD.map(lambda x: (x[0], x[1])).groupByKey().mapValues(
        set)  # (user,{business_ids})
    user_business_pair_RDD_ = user_business_pair_RDD_.collectAsMap()  # user:{business_ids}
    business_user_pair_RDD_ = training_RDD.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)  # (business,{user})
    business_user_pair_RDD_ = business_user_pair_RDD_.collectAsMap()  # business:{user_id}

    user_sum_len_RDD = training_RDD.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(
        lambda x: (sum(x), len(x))).collectAsMap()  # user_id:(sum(rating),count_raters)
    business_user_rating_group_RDD = training_RDD.map(
        lambda x: ((x[1], x[0]), float(x[2]))).collectAsMap()  # (business_id,user_id):rating
    business_sum_len_RDD = training_RDD.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(
        lambda x: (sum(x), len(x))).collectAsMap()  # business_id:(sum(rating),count_raters)

    all_rating_average = training_RDD.map(lambda x: (1, float(x[2]))).groupByKey().mapValues(
        lambda x: (sum(x), len(x))).collect()
    default_rating_set = float(all_rating_average[0][1][0] / all_rating_average[0][1][1])
    # default_rating_set=2.5

    testing_RDD = sc.textFile(input_test_path)

    header2 = testing_RDD.collect()[0]
    testing_RDD = testing_RDD.filter(lambda x: x != header2)
    testing_RDD = testing_RDD.map(lambda x: x.split(','))
    testing_business_user_RDD = testing_RDD.map(lambda x: (x[0], x[1]))  # (user_id,businessId)

    def prediction_func(x):
        x_user_id = x[0]
        x_business_id = x[1]
        similarity_local_dictionary = {}
        if x_user_id not in user_business_pair_RDD_ and x_business_id in business_user_pair_RDD_:
            pred_rating_bus = float(business_sum_len_RDD[x_business_id][0] / business_sum_len_RDD[x_business_id][
                1])  # new user but we can use past business data for pred_ratings
            return tuple([x_user_id, x_business_id, pred_rating_bus])
        elif x_user_id not in user_business_pair_RDD_ and x_business_id not in business_user_pair_RDD_:
            return tuple([x_user_id, x_business_id,
                          default_rating_set])  # For complete coldstart, no user data and business data
        elif x_user_id in user_business_pair_RDD_ and x_business_id not in business_user_pair_RDD_:
            pred_rating_user = float(user_sum_len_RDD[x_user_id][0] / user_sum_len_RDD[x_user_id][
                1])  # new business but we can use past user data for pred_ratings
            return tuple([x_user_id, x_business_id, pred_rating_user])

        all_business_by_x_user = user_business_pair_RDD_[x_user_id]  # all business which x_user_id rated
        all_users_for_business = business_user_pair_RDD_[x_business_id]  # all users who rated x_business_id
        if x_business_id in all_business_by_x_user:
            return tuple([x_user_id, x_business_id, business_user_rating_group_RDD[(x_user_id, x_business_id)]])

        average_business_ratings = float(business_sum_len_RDD[x_business_id][0] / business_sum_len_RDD[x_business_id][
            1])  # Avg rating for x_business_id
        for i in all_business_by_x_user:
            temp_index = tuple(sorted([i, x_business_id]))
            if temp_index in similarity_overall.keys():
                similarity_local_dictionary[i] = similarity_overall[temp_index]
                continue
            else:
                prod_business1_business2 = 0
                sum_squares_business1 = 0
                sum_squares_business2 = 0
                average_rating_business_i = float(business_sum_len_RDD[i][0] / business_sum_len_RDD[i][
                    1])  # Average rating for business_i for which we are calulating W(i,x_business)

                all_user_business_i = business_user_pair_RDD_[i]
                intersection_users = set(all_user_business_i & all_users_for_business)
                # intersection_users=set(all_user_business_i).intersection(set(all_users_for_business))
                if len(intersection_users) == 0:  # Eventhough no same users are rating businesses, we will still take the impact into consideration as small value
                    division_similarity = float(average_business_ratings / average_rating_business_i)
                    if division_similarity > 1:
                        division_similarity = float(1 / division_similarity)
                    similarity_local_dictionary[i] = division_similarity
                    similarity_overall[temp_index] = division_similarity
                    continue
                for j in intersection_users:
                    rating_user_i = business_user_rating_group_RDD[(i, j)] - average_rating_business_i
                    rating_x_user_id = business_user_rating_group_RDD[(x_business_id, j)] - average_business_ratings
                    prod_business1_business2 += (rating_user_i * rating_x_user_id)
                    sum_squares_business1 += (rating_user_i * rating_user_i)
                    sum_squares_business2 += (rating_x_user_id * rating_x_user_id)
                if prod_business1_business2 == 0:
                    pearson_value = 0
                else:
                    pearson_value = float(prod_business1_business2 / (
                        float(float(math.sqrt(sum_squares_business1)) * float(math.sqrt(sum_squares_business2) * 1))))

                similarity_local_dictionary[i] = pearson_value
                similarity_overall[temp_index] = pearson_value

        for i in similarity_local_dictionary.keys():
            if similarity_local_dictionary[i] > 0:  # Charging small ratings
                similarity_local_dictionary[i] = float(
                    similarity_local_dictionary[i] * pow(abs(similarity_local_dictionary[i]), 1.5))
                similarity_overall[tuple(sorted([i, x_business_id]))] = similarity_local_dictionary[i]

        similarity_local_dictionary_items = sorted(similarity_local_dictionary.items(), key=lambda x: x[1],
                                                   reverse=True)
        total_num = len(similarity_local_dictionary)

        average_sum = 0
        average_weight_elements = 0

        for pair in similarity_local_dictionary_items:
            if similarity_local_dictionary[pair[0]] >= 0:
                total_num -= 1
                average_sum += business_user_rating_group_RDD[pair[0], x_user_id] * similarity_local_dictionary[pair[0]]
                average_weight_elements += abs(similarity_local_dictionary[pair[0]])
            if total_num == 0:
                break

        if average_sum == 0:
            pred_rate = 0
        else:
            pred_rate = float(average_sum / average_weight_elements)

        if pred_rate == 0:
            def_business_rat = float(business_sum_len_RDD[x_business_id][0] / business_sum_len_RDD[x_business_id][1])
            def_user_rat = float(user_sum_len_RDD[x_user_id][0] / user_sum_len_RDD[x_user_id][1])
            average_business_user_def = float((def_user_rat + def_business_rat) / 2)
            return tuple([x_user_id, x_business_id, average_business_user_def])

        pred_rate = min(pred_rate, 5)
        return tuple([x_user_id, x_business_id, pred_rate])

    predicted_ratings_test = testing_RDD.map(lambda x: prediction_func(x))
    # print(predicted_ratings_test.collect())

    with open(output_path_val_item, 'w', newline="") as fileoutput:
        filewriter = csv.writer(fileoutput)
        filewriter.writerow(['user_id', 'business_id', 'prediction'])
        filewriter.writerows(predicted_ratings_test.collect())

    output_file = open(output_path_val_item, "r")
    reference_file = open(input_test_path, "r")
    n = 0
    rmse = 0
    while (True):
        output_file_line = output_file.readline()
        reference_file_line = reference_file.readline()
        if "user_id" in reference_file_line:
            continue
        if reference_file_line == "":
            break
        n += 1
        rmse += math.pow(float(output_file_line.split(",")[2][:-1]) - float(reference_file_line.split(",")[2][:-1]), 2)
        if not output_file_line and not reference_file_line:
            break

    rmse = math.sqrt(rmse / n)

    print("ItemBased:",rmse)


def model_based_recommendation_function():
    training_data = pd.read_csv(all_data_files + "/yelp_train.csv")
    testdata_data = pd.read_csv(input_test_path)
    user_data_file = pd.read_json(all_data_files + "/user.json", lines=True, chunksize=80000)
    business_data_file = pd.read_json(all_data_files + "/business.json", lines=True, chunksize=80000)


    flag_val = 0
    for i in user_data_file:
        if flag_val == 0:
            user_id_data = i
            flag_val = 1
            continue
        user_id_data = user_id_data.append(i)

    flag_val_2 = 0
    for i in business_data_file:
        if flag_val_2 == 0:
            business_id_data = i
            flag_val_2 = 1
            continue
        business_id_data = business_id_data.append(i)

    user_data = user_id_data[['user_id', 'average_stars', 'review_count', 'useful', 'fans']]
    business_data = business_id_data[['business_id', 'stars', 'review_count']].rename(
        columns={'stars': 'business_stars', 'review_count': 'business_review_count'})

    training_data_comb = pd.merge(training_data, user_data, on='user_id', how='inner')
    training_data_comb = pd.merge(training_data_comb, business_data, on='business_id', how='inner')

    data_var = {}
    data_number_user_id = {}
    data_max_star = {}
    data_min_star = {}

###########################################

    for i, j in training_data_comb.iterrows():
        if j["user_id"] not in data_var.keys():
            data_var[j["user_id"]] = pow((float(j["stars"]) - float(j["average_stars"])), 2)
            data_number_user_id[j["user_id"]] = 1
        else:
            data_var[j["user_id"]] += pow((float(j["stars"]) - float(j["average_stars"])), 2)
            data_number_user_id[j["user_id"]] += 1

        if j["user_id"] not in data_max_star.keys():
            data_max_star[j["user_id"]] = j["stars"]
        else:
            if data_max_star[j["user_id"]] < j["stars"]:
                data_max_star[j["user_id"]] = j["stars"]
        if j["user_id"] not in data_min_star.keys():
            data_min_star[j["user_id"]] = j["stars"]
        else:
            if data_min_star[j["user_id"]] > j["stars"]:
                data_min_star[j["user_id"]] = j["stars"]

    ###################################################
    # d1={}
    # for i, content in training_data_comb.iterrows():
    #     if (content['user_id'] not in d1):
    #         data_number_user_id[content['user_id']] = 1
    #         data_var[content['user_id']] = (content['stars'] - content['average_stars']) * (
    #                     content['stars'] - content['average_stars']) * 1 + 0
    #         data_max_star[content['user_id']] = content['stars']
    #         data_min_star[content['user_id']] = content['stars']
    #
    #     else:
    #
    #         data_number_user_id[content['user_id']] = data_number_user_id[content['user_id']] + 1
    #         data_var[content['user_id']] += (
    #                     (content['stars'] - content['average_stars']) * (content['stars'] - content['average_stars']))
    #         if (content['stars'] > data_max_star[content['user_id']]):
    #             data_max_star[content['user_id']] = content['stars']
    #         if (content['stars'] < data_min_star[content['user_id']]):
    #             data_min_star[content['user_id']] = content['stars']

    ###################################################
    new_temp_df = pd.DataFrame(columns=["user_id", "user_variance", "Max_rating", "Min_rating"])
    for i in data_var.keys():
        f_variance = float(data_var[i] / data_number_user_id[i])
        new_temp_df = new_temp_df.append(
            {'user_id': i, 'user_variance': f_variance, 'Max_rating': data_max_star[i], 'Min_rating': data_min_star[i]},
            ignore_index=True)

    # print(new_temp_df)
    training_data_comb = pd.merge(training_data_comb, new_temp_df, on='user_id', how='left')  # how='left'

    temp_train_data = training_data_comb
    for i in temp_train_data.columns:
        if temp_train_data[i].dtype == 'object':
            temp_pre = LabelEncoder()
            temp_pre.fit(list(temp_train_data[i].values))
            temp_train_data[i] = temp_pre.transform(list(temp_train_data[i].values))

    # print(temp_train_data)

    # ratings_val=temp_train_data.stars.values
    ratings_val = temp_train_data["stars"].values
    print(len(temp_train_data.stars.values))
    print(len(list(temp_train_data["stars"].values)))
    training_dataset = temp_train_data.drop(['stars'], axis=1)
    training_dataset = training_dataset.drop(['user_id'], axis=1)
    training_dataset = training_dataset.drop(['business_id'], axis=1)
    training_dataset = training_dataset.values

    model_prepare = xgboost.XGBRegressor()
    model_prepare.fit(training_dataset, ratings_val)

    testestdata_comb = pd.merge(testdata_data, user_data, on='user_id', how='left')  # how='left'
    testestdata_comb = pd.merge(testestdata_comb, business_data, on='business_id', how='left')  # how='left'
    testestdata_comb = pd.merge(testestdata_comb, new_temp_df, on='user_id', how='left')  # how='left'

    user_test_data_id = testestdata_comb['user_id'].values
    business_test_data_id = testestdata_comb['business_id'].values

    test_data = testestdata_comb

    for i in test_data.columns:
        if test_data[i].dtype == 'object':
            test_pre = LabelEncoder()
            test_pre.fit(list(test_data[i].values))
            test_data[i] = test_pre.transform(list(test_data[i].values))

    test_data.fillna((-99), inplace=True)

    testing_dataset = test_data.drop(['user_id'], axis=1)
    testing_dataset = testing_dataset.drop(['business_id'], axis=1)
    testing_dataset = testing_dataset.drop(['stars'], axis=1)
    testing_dataset = testing_dataset.values

    prediction_sol = model_prepare.predict(testing_dataset)

    Predicted_solution_File = pd.DataFrame()
    Predicted_solution_File['user_id'] = user_test_data_id
    Predicted_solution_File['business_id'] = business_test_data_id
    Predicted_solution_File['prediction'] = prediction_sol

    # print(user_test_data_id[0])
    # Predicted_solution_File.to_csv(output_path_val,sep=',',encoding='utf-8',index=False)

    temp_RDD_list = []
    for i in range(len(user_test_data_id)):
        temp_RDD_list.append((user_test_data_id[i], business_test_data_id[i], prediction_sol[i]))

    temp_RDD = sc.parallelize(temp_RDD_list)
    with open(output_path_val_model, 'w', newline="", encoding='utf-8') as fileoutput:
        filewriter = csv.writer(fileoutput)
        filewriter.writerow(['user_id', 'business_id', 'prediction'])
        filewriter.writerows(temp_RDD.collect())

    reference_file = open(input_test_path, "r")
    output_file = open(output_path_val_model, "r")
    n = 0
    rmse = 0
    while (True):
        output_file_line = output_file.readline()
        reference_file_line = reference_file.readline()
        if "user_id" in reference_file_line:
            continue
        if reference_file_line == "":
            break
        n += 1
        rmse += math.pow(float(output_file_line.split(",")[2][:-1]) - float(reference_file_line.split(",")[2][:-1]), 2)
        if not output_file_line and not reference_file_line:
            break
    rmse = math.sqrt(rmse / n)
    print("ModelBased: ",rmse)




#
# item_based_recommendation_function()
# model_based_recommendation_function()
#
# output_file_final = open(output_file_val,"w")
# test_line_checker = open(input_test_path,"r")
# lines_n=0
# for i in test_line_checker:
#     if i!="":
#         break
#     lines_n+=1
#
# train_data_RDD = sc.textFile(all_data_files+"/yelp_train.csv")
# header = train_data_RDD.collect()[0]
# train_data_RDD = train_data_RDD.filter(lambda x:x!=header)
# user_business_train_RDD = train_data_RDD.map(lambda x: x.split(',')).map(lambda x:(x[0],x[1])).groupByKey().mapValues(set).collectAsMap()
# #user:{businesses}
#
# test_data_RDD = sc.textFile(input_test_path)
# header = test_data_RDD.collect()[0]
# test_data_RDD = test_data_RDD.filter(lambda x:x!=header)
# user_business_test_RDD = test_data_RDD.map(lambda x: x.split(',')).map(lambda x:(x[0],x[1])).collect()
#
# alpha_f={}
# for i in user_business_test_RDD:
#     alpha_f[i]=len(user_business_train_RDD[i[0]])       #(user, business): Number of businesses by that User
#
# item_output_file = open(output_path_val_item)
# model_output_file = open(output_path_val_model)
# max_val=max(alpha_f.items(),key=lambda x:x[1])[1]
# output_file_final.write("user_id,business_id,prediction\n")
# line_no=1
# while True:
#     line_no+=1
#     item_line=item_output_file.readline()
#     model_line=model_output_file.readline()
#     if "user_id" in model_line:
#         continue
#     if model_line=="":  # or item_line==""
#         break
#     if not item_line and not model_line:
#         break
#     #print(model_line.split(","))
#     user_id_data=model_line.split(",")[0]
#     business_id_data=model_line.split(",")[1]
#     if model_line.split(',')[-1]!='\n':
#         rating_model=float(model_line.split(",")[2])
#     else:
#         rating_model=float(model_line.split(",")[2][:-1])
#     if item_line.split(',')[-1]!='\n':
#         rating_item=float(item_line.split(",")[2])
#     else:
#         rating_item=float(item_line.split(",")[2][:-1])
#     alpha = float((alpha_f[user_id_data, business_id_data])/ (3*max_val))
#     combined_rating=((alpha*rating_item)+((1-alpha)*rating_model))
#     tempstr=user_id_data+","+business_id_data+","+str(combined_rating)
#     output_file_final.write(tempstr)
#
#     if lines_n!=line_no:
#         output_file_final.write("\n")


n=0
rmse=0

output_file_rmse = open(output_file_val,"r")
reference_file_final_rmse = open(input_test_path,"r")
while(True):
    l1=output_file_rmse.readline()
    l2=reference_file_final_rmse.readline()
    if "user_id" in l2:
        continue
    if l2=="" :
        break
    n+=1
    if not l1 and not l2:
        break
    rmse+=math.pow(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),2)


rmse=math.sqrt(rmse/n)

print(rmse)

print("Duration: ",time.time()-start_time)

