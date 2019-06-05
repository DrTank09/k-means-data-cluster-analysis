# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:30:07 2019

@author: Tank
"""

# finalproject2.py
# Phase 1 of final project
# Date: 04/14/2019
# Author: Ryan Leibering

import pandas as pd
import numpy as numpy

def intialize(data):
    #ensure that u2 and u4 aren't the same
    while True:
        u2index = numpy.random.randint(0,len(data[0:]))
        u4index = numpy.random.randint(0,len(data[0:]))
        if u2index != u4index:
            break
    #initalize mean
    u2data = (data[data.columns[1:10]][u2index:u2index+1])
    u4data = (data[data.columns[1:10]][u4index:u4index+1])
    return(u2data,u4data)

def assignment(data,u2,u4):
    #initialize clusters
    u2cluster = []
    u4cluster = []
    #cluster all datapoints
    for r in range(len(data[0:])):
        #set k-mean variable and compare between u2 and u4
        u2dis = [0]
        u4dis = [0]
        for c in data[data.columns[1:10]]:
            u2dis[0] += (u2[c]-data[c][r])**2
            u4dis[0] += (u4[c]-data[c][r])**2
        u2dis[0] = numpy.sqrt(u2dis[0])
        u4dis[0] = numpy.sqrt(u4dis[0])
        #assign to cluster based on k-mean variable
        if u2dis[0].iloc[0] <= u4dis[0].iloc[0]:
            u2cluster.append(r)
        else:
            u4cluster.append(r)
    return u2cluster,u4cluster

def recalculation(data,u2cluster,u4cluster):
    #calculate new u2 mean for each column
    u2i = []
    for c in data[data.columns[1:10]]:
        value = [0]
        for r in u2cluster:
            value[0] += data[c][r]
        u2i.append(value[0]/len(u2cluster))
    #calculate new u4 mean for each column
    u4i = []
    for c in data[data.columns[1:10]]:
        value = [0]
        for r in u4cluster:
            value[0] += data[c][r]
        u4i.append(value[0]/len(u2cluster))
    #turn u2 and u4 into dataframes and transpose to match format
    u2i = pd.DataFrame(u2i,index = data.columns[1:10])
    u2i = u2i.T
    u4i = pd.DataFrame(u4i,index = data.columns[1:10])
    u4i = u4i.T
    return u2i, u4i
        
            

def main():
    #open file
    data = pd.read_csv("Breast-Cancer-Wisconsin.csv",sep=",",header="infer",na_values="?")
    #find average for missing A7 data
    a7 = data["A7"]
    count = [0]
    for i in a7:
        if i >= 0:
            count[0]+=1
    a7mean = a7.sum()/count
    #replace A7 data
    data["A7"] = data["A7"].fillna(a7mean[0])
    #make data a dataframe
    data = pd.DataFrame(data)
    #initialize mean
    u2,u4 = intialize(data)
    #assign cluster values
    u2cluster,u4cluster = assignment(data,u2,u4)
    #recalculate new mean
    u2i, u4i = recalculation(data,u2cluster,u4cluster)
    #set new means
    u2 = u2i
    u4 = u4i
    #begin 1500 count loop
    count=[0]
    while count[0] <= 1500:
        #assign new cluster based off recalculation
        u2cluster, u4cluster = assignment(data,u2,u4)
        #recalculate mean
        u2i, u4i = recalculation(data,u2cluster,u4cluster)
        checku2=[0]
        checku4=[0]
        #check if update occurs
        for i in u2i.columns[:]:
            if u2i[i].iloc[0] == u2[i].iloc[0]:
                checku2[0] +=1
            if u4i[i].iloc[0] == u4[i].iloc[0]:
                checku4[0] +=1
        if checku2[0] == len(u2.columns[:]) and checku4[0] == len(u4.columns[:]):
            break
        #assign new values
        u2 = u2i
        u4 = u4i
        count[0]+=1
    #create results dataframe
    results = pd.DataFrame(columns=(data.columns[0],data.columns[10],"Predicted Class"))
    for i in u2cluster:
        results = results.append(pd.DataFrame([[data[data.columns[0]].iloc[i],data[data.columns[10]].iloc[i],2]],index=[i],columns=(data.columns[0],data.columns[10],"Predicted Class")))
    for i in u4cluster:
        results = results.append(pd.DataFrame([[data[data.columns[0]].iloc[i],data[data.columns[10]].iloc[i],4]],index=[i],columns=(data.columns[0],data.columns[10],"Predicted Class")))
    #sort dataframe
    results = results.sort_index(ascending = True)
    #create mean list for asthetic purposes
    u2results = []
    u4results = []
    for i in u2[u2.columns[:]]:
        u2results.append(u2[i][0])
    for i in u4[u4.columns[:]]:
        u4results.append(u4[i][0])
    print ("Final Mean")
    print()
    print("U2:",u2results)
    print()
    print("U4:",u4results)
    print()
    print("Cluster Assignment")
    print(results[0:21])

    
    
main()