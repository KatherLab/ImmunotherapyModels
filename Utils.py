# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:05:36 2021

@author: Narmin Ghaffari Laleh
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

##############################################################################
# BASIC FUNCTIONS FOR SPIDER PROJECT
##############################################################################

def Is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
############################################################################## 
        
def Read_Excel(rawDataPath, ArmName = 'TRT01A', split = True):
    data = pd.read_excel(rawDataPath)

    # Find Out the Actual Arms
    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    
    data['receivedTreatment'] = receivedTreatment
    return data, receivedTreatment_unique

##############################################################################
    
def Create_Result_dict(arms, trendNames, categories):
    resultDict = {}
    for arm in arms:
        resultDict[arm] = {}
        for tn in trendNames:
            resultDict[arm][tn] = {}
            for categ in categories:
                resultDict[arm][tn][categ] = []
    return resultDict
    
##############################################################################

def Correct_Time_Vector(time, convertToWeek = True):
    
    if convertToWeek:
        time = [math.ceil(i/7) for i in time]
        time = [0.1 if i<=0 else i for i in time]
    else:
        time = [0.1 if i<=0 else i for i in time]
    return time

###############################################################################
    
def Remove_String_From_Numeric_Vector(vector, valueToReplace):
    vector = [valueToReplace if not Is_number(str(i)) else i for i in vector]
    vector = [valueToReplace if  str(i) == 'nan' else i for i in vector]
    return vector

###############################################################################

def Detect_Trend_Of_Data(vector): 
    
    diff = []
    for d in range(len(vector)-1):
        diff.append(vector[d + 1] - vector[d])  
    s_pos = 0
    for x in diff:
        if x>0:
            s_pos = s_pos + x
            
    s_neg = 0
    for x in diff:
        if x<0:
            s_neg = s_neg + x
            
    if all(i >= 0 for i in diff):
        trend = 'Up'
    elif all(i <= 0 for i in diff):
        trend = 'Down'
    elif diff[0] > 0 and not abs(s_neg) >= (s_pos /2):
        trend = 'Up'
    elif diff[0] < 0 and not s_pos >= (abs(s_neg) /2):
        trend = 'Down'
    else:
        trend = 'Fluctuate'
    return trend

# =============================================================================
            
def Write_On_Result_dict(resultDict, arm, trend, categories, values):

    noCateg = len(categories)
    for i in range(noCateg):
        resultDict[arm][trend][categories[i]].append(values[i])    
    return resultDict

###############################################################################





    
