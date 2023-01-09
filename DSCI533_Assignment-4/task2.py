from pyspark import SparkContext
import os
import itertools
import sys
import random
import time

start_time=time.time()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


input_path_val = r"C:\Users\ankus\PycharmProjects\dsci544_Assignment-4\ub_sample_data.csv"
betweenness_output_path_val = r"Assignment-4_output_Realtask2_betweenness_output.txt"
communities_output_path_val= r"Assignment-4_output_Realtask2_communities_output.txt"

threshold_val = int(7)

sc = SparkContext('local[*]', 'Assignment_4_Tasks')

# print(sc.version)

RDDFile = sc.textFile(input_path_val)
column_head=RDDFile.collect()[0]
RDDFile=RDDFile.filter(lambda x: x!=column_head).map(lambda x:x.split(',')).map(lambda x:(x[0],x[1]))


user_business_RDD=RDDFile.groupByKey().mapValues(list)          #(user,[businesses])
user_business_Dict=user_business_RDD.collectAsMap()             #user: [businesses]
all_users=user_business_RDD.map(lambda x:x[0]).collect()                  #[users]      #Nodes


user_pairs=[i for i in itertools.combinations(all_users,2)]           #

print(len(user_pairs))

#Filtering
edges_user_list=[]
node_user_set=set()
for i in user_pairs:
    u_1=set(user_business_Dict[i[0]])
    u_2=set(user_business_Dict[i[1]])
    if len(u_1.intersection(u_2))>=threshold_val:
        edges_user_list.append(i)
        edges_user_list.append(tuple([i[1],i[0]]))
        node_user_set.add(i[0])
        node_user_set.add(i[1])

node_user_list=list(node_user_set)

node_neighbours=dict()

for i in edges_user_list:
    if i[0] not in node_neighbours:
        node_neighbours[i[0]]=set()
    if i[1] not in node_neighbours:
        node_neighbours[i[1]] = set()
    node_neighbours[i[0]].add(i[1])
    node_neighbours[i[1]].add(i[0])


def Girvan_Newman_Func(x):
    visited_nodes=set()
    visited_nodes.add(x)
    all_parents=dict()      #Parents of an element
    all_children=dict()         #Children for an element
    number_of_parents=dict()        #parents count
    number_of_parents[x]=1
    elements_each_level=dict()       #at each level
    elements_each_level[0]=x
    available_nodes=node_neighbours[x]
    all_children[x]=available_nodes
    for k in available_nodes:
        if k not in all_parents:
            all_parents[k] = set()
        all_parents[k].add(x)
    table_height=0
    while available_nodes:
        table_height += 1
        elements_each_level[table_height]=available_nodes
        visited_nodes=visited_nodes.union(available_nodes)
        next_neighbours=set()
        for i in available_nodes:
            next_neighbours_part=node_neighbours[i]
            next_child=next_neighbours_part-visited_nodes
            all_children[i]=next_child
            # if i not in all_parents:
            #     all_parents[i] = set()
            for k in next_child:
                if k not in all_parents:
                    all_parents[k]=set()
                all_parents[k].add(i)
            previous_parent=all_parents[i]
            number_of_parents[i]=sum(number_of_parents[k] for k in previous_parent)             #To calculate number of parents from its parent's  number of parent data
            next_neighbours=next_neighbours.union(next_neighbours_part)
        available_nodes=next_neighbours-visited_nodes

    previous_value = dict()
    previous_value[x] = 0
    final_ans_betweenness = dict()
    for i in node_user_list:
        if i != x:
            previous_value[i] = 1

    while table_height > 0:
        for i in elements_each_level[table_height]:
            parents_for_i = all_parents[i]
            for k in parents_for_i:
                weight_betweenness = float(number_of_parents[k] / number_of_parents[i])
                temp_sort_index = tuple(sorted([i, k]))
                final_ans_betweenness[temp_sort_index] = weight_betweenness * previous_value[i]
                previous_value[k] += final_ans_betweenness[temp_sort_index]
        table_height -= 1
    betweenness_calculator=list(final_ans_betweenness.items())
    return betweenness_calculator

betweenness = sc.parallelize(node_user_list).map(lambda x: Girvan_Newman_Func(x)).flatMap(lambda x: [i for i in x])
betweenness=betweenness.groupByKey()
betweenness=betweenness.mapValues(sum).mapValues(lambda x: (x/2)).sortBy(lambda x: [-x[1], x[0]]).collect()

# print(betweenness[:3])

with open(betweenness_output_path_val, 'w',newline="") as file_writer:
    for i in betweenness:
        val=round(i[1],5)
        file_writer.write(str(tuple([i[0],val])).rstrip(")")[1:])
        if i!=(betweenness[len(betweenness)-1]):
            file_writer.write('\n')


degree_node=dict()
for k,v in node_neighbours.items():
    degree_node[k]=len(v)

matrix_edge_presence=dict()
for i in range(len(node_user_list)):
    for j in range(len(node_user_list)):
        if (node_user_list[i],node_user_list[j]) not in edges_user_list:
            matrix_edge_presence[(node_user_list[i],node_user_list[j])]=0
        else:
            matrix_edge_presence[(node_user_list[i], node_user_list[j])] = 1

total_edges=len(edges_user_list)/2
remaining_edges=total_edges
modularity=-1


def visited_nodes_func(x, node_neighbours):
    visited_nodes = {x}
    neighbour_x = node_neighbours[x]
    while True:
        visited_nodes = visited_nodes | neighbour_x
        overall_neighbours = set()
        for i in neighbour_x:
            next_neighbour = node_neighbours[i]
            overall_neighbours = overall_neighbours.union(next_neighbour)
        All_nodes = visited_nodes.union(overall_neighbours)
        if (len(visited_nodes)) == (len(All_nodes)):
            break
        neighbour_x = overall_neighbours - visited_nodes
    return visited_nodes


for klm in range(len(node_user_list)):
    highest_betweenness=betweenness[0][1]
    for i in range(len(betweenness)):
        if betweenness[i][1]==highest_betweenness:
            if betweenness[i][0][1] in  node_neighbours[betweenness[i][0][0]]:
                node_neighbours[betweenness[i][0][0]].remove(betweenness[i][0][1])
            if betweenness[i][0][0] in node_neighbours[betweenness[i][0][1]]:
                node_neighbours[betweenness[i][0][1]].remove(betweenness[i][0][0])
            remaining_edges-=1
    common_nodes = []
    nodes = node_user_set
    visited = visited_nodes_func(node_user_list[klm], node_neighbours)
    unvisited = nodes - visited
    common_nodes.append(visited)
    while True:
        visited_nodes = visited_nodes_func(random.choice(list(unvisited)), node_neighbours)
        common_nodes.append(visited_nodes)
        visited = visited.union(visited_nodes)
        unvisited = nodes - visited
        if (len(unvisited)) == (0):
            break
    
    local_mod = 0
    for i in common_nodes:
        jk_mod = 0
        for j in i:
            for k in i:
                jk_mod = (jk_mod + matrix_edge_presence[(j, k)] - ((degree_node[j] * degree_node[k]) / (2 * total_edges)) )
        local_mod +=jk_mod
    local_mod = local_mod / (2 * total_edges)

    if local_mod > modularity:
        modularity = local_mod
        final_comm_nodes=common_nodes
    if remaining_edges==0:
        break
    betweenness = sc.parallelize(node_user_list).map(lambda x: Girvan_Newman_Func(x)).flatMap(lambda x: [i for i in x])
    betweenness=betweenness.groupByKey()
    betweenness=betweenness.mapValues(sum).mapValues(lambda x: (x/2)).sortBy(lambda x: [-x[1], x[0]]).collect()

final_communitites=sc.parallelize(final_comm_nodes).map(lambda x:sorted(x)).sortBy(lambda x:[len(x),x]).collect()

with open(communities_output_path_val, 'w+',newline="") as file_writer_2:
    for i in final_communitites:
        file_writer_2.write(str(i).strip('[]'))
        if i!=(final_communitites[len(final_communitites)-1]):
            file_writer_2.write('\n')

print("Duration: ",time.time()-start_time)
