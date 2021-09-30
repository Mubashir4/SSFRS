import numpy as np
import utility as utl

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from User_Module import User
from Cluster import Clustring



#_______________________________________________
#MACORS
Path = 'C:/Mubashir/Work/VLDB/Movie Lens/archive/Users/'
seq_length=16
batch_size = 50
#_______________________________________________

clusters_list = []

def executre_all_iteratively(user_100,user_d,user_w):
    user_dic = user_d
    user_weights = user_w
    for user in user_100:
        print("User: ",user[1])
        if user[1] not in user_dic:
            user_dic[user[1]] = User(user[0],str(user[1]),seq_length,batch_size,client_server_directory = 'client_server_directory/')
            user_weights[user[1]] = []
            user_dic[user[1]].get_recommendations()
            if user_dic[user[1]].current_iteration > -1:
                user_weights[user[1]] =  user_dic[user[1]].get_weights()
    return user_dic,user_weights




for i in range(0,1):
    if(i==0):
        user_files = utl.get_file_addresses(Path)
        user_100 = user_files[0:50]
        #user_100.append(['C:/Mubashir/Work/VLDB/Movie Lens/archive/Users/','8405'])
        user_d,user_w = {},{}
        user_dic,user_weights = executre_all_iteratively(user_100,user_d,user_w)
        c = Clustring(user_weights)
        labels = c.labels
        clusters_list.append(c.x)
        num_clusters = max(labels)+1
        avg,avg_count = c.weight_aggregator(labels,user_weights,num_clusters)
        user_d = c.update_weights(labels,user_dic,avg)
        user_w = {}
    else:
        user_w = {}
        user_dic,user_weights = executre_all_iteratively(user_100,user_d,user_w)
        users_list_for_index = list(user_weights)
        c = Clustring(user_weights)
        labels = c.labels
        clusters_list.append(c.x)
        num_clusters = max(labels)+1
        
        #For independent computation Please uncomment this
        '''
        for k,v in user_weights.items():
            l = c.compute_Cluster(v)
            ind = users_list_for_index.index(k)
            labels[ind] = l[0]
        '''
        avg,avg_count = c.weight_aggregator(labels,user_weights,num_clusters)
        user_d = c.update_weights(labels,user_dic,avg)




###############################################################################################################3


for k,v in user_dic.items():
    v.write_Embeddings()
