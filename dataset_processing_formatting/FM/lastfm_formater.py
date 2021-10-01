# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:28:59 2021

@author: s4523139
"""


import pandas as pd

import csv

'''

header = ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
with open('needed_data.csv', 'a',encoding="utf8",newline='') as g:
                    csv_writer = csv.writer(g)
                    csv_writer.writerow(header )
                    g.close()
'''

_id = 0
track_id = {}
abnormal_data = {}

header = ['user_id','track_id','time_stamp']
with open('user_track.csv', 'a',encoding="utf8",newline='') as g:
                    csv_writer = csv.writer(g)
                    csv_writer.writerow(header)
                    g.close()

with open('userid-timestamp-artid-artname-traid-traname.csv', 'r',encoding="utf8") as f:
    csv_reader = csv.reader(f, delimiter='\t',quoting=csv.QUOTE_NONE)
    for i,row in enumerate(csv_reader):
            if(len(row)>6):
                abnormal_data.append(row)
            else:
                with open('user_track.csv', 'a',encoding="utf8",newline='') as g:
                    csv_writer = csv.writer(g)
                    if(row[5] not in track_id):
                        _id+=1
                        track_id[row[5]]=_id
                    csv_writer.writerow([row[0],track_id[row[5]],row[1]])
                    g.close()
            if(i%10000==0):
                print(i)

        