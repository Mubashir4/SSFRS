#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:28:33 2021

@author: s4523139
"""

import networkx as nx
import datetime
import csv
import pandas as pd




#df = pd.DataFrame(columns=['user_id','track_id','time_stamp'])

dic = {}
dic['user_id'] = []
dic['track_id'] = []
dic['time_stamp'] = []

with open('user_track.csv', 'r',encoding="utf8") as f:
    csv_reader = csv.reader(f, delimiter=',',quoting=csv.QUOTE_NONE)
    for i,row in enumerate(csv_reader):
        if(i==0):
            print('started')
            continue
        else:
            d_t = row[2].split('T')
            d_t = datetime.datetime.strptime(d_t[0]+' '+(d_t[1].replace('Z','')),'%Y-%m-%d %H:%M:%S')
            dic['user_id'].append(row[0])
            dic['track_id'].append(row[1])
            dic['time_stamp'].append(d_t)
        
        if(i%10000 == 0):
            print(i)
 
#df=pd.DataFrame.from_dict(dic,orient='index').transpose()           
#df.sort_values(by='time_stamp', inplace=True)

df = pd.DataFrame({'user_id':dic['user_id']})
df['track_id'] = dic['track_id']
print('track_id')
df['time_stamp'] = dic['time_stamp']


df.sort_values(by='time_stamp', inplace=True)

df.iloc[0]


user_graphs = {}
number_of_rows = len(df)
for i in range(0,number_of_rows):
    row = df.iloc[i]
    if(row[0] not in user_graphs):
        user_graphs[row[0]] = [[row[1],0,row[2],0]]
    else:
        user_graphs[row[0]][-1][1] = row[1]
        user_graphs[row[0]][-1][3] = row[2]
        user_graphs[row[0]].append([row[1],0,row[2],0])
        
  
file = 'Last_FM_Graphs/'
for k,v in user_graphs.items():
    with open(file+str(k)+'.csv','w',newline='') as graph:
        csv_writer = csv.writer(graph,delimiter=',')
        for line in v:
            csv_writer.writerow(line)
        graph.close()
        print(str(k)+' written')
    
    
    
    
    
user_000547 = user_graphs['user_000547']

Graph = nx.Graph()
for e1,e2,_,_ in user_000547:
    Graph.add_edge(e1,e2)
    
degrees = list(Graph.degree())
sum_of_edges = sum(degrees.values())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    