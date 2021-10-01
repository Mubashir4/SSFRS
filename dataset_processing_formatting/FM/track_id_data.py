#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:25:19 2021

@author: s4523139
"""

import csv

_id = 0
track_id = {}
track_id_inverse = {}
abnormal_data = {}

tracks = []
'''

with open('user_track.csv', 'r',encoding="utf8") as f:
    csv_reader = csv.reader(f, delimiter='\t',quoting=csv.QUOTE_NONE)
    for i,row in enumerate(csv_reader):
        tracks.append(row)
        
'''        
with open('userid-timestamp-artid-artname-traid-traname.csv', 'r',encoding="utf8") as f:
    csv_reader = csv.reader(f, delimiter='\t',quoting=csv.QUOTE_NONE)
    for i,row in enumerate(csv_reader):
            if(len(row)>6):
                abnormal_data.append(row)
            else:
                if(row[5] not in track_id_inverse):
                        _id+=1
                        track_id_inverse[row[5]] = _id
                        track_id[_id]=[row[3],row[5]]
                        with open('track_data.csv', 'a',encoding="utf8",newline='') as g:
                            csv_writer = csv.writer(g)
                            csv_writer.writerow([_id,row[3],row[5]])