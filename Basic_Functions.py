"""
Created on Sun Jan 10 16:05:36 2021

@author: Narmin Ghaffari Laleh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Utils as utils
from matplotlib.lines import Line2D
import math

###############################################################################

def GetNumberOfPatientPerArm(rawDataPath, ArmName = 'TRT01A', split = True):
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    print(receivedTreatment_unique)
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment

    for arm in receivedTreatment_unique:    
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        print('Number of Patients in arm ' + arm + ' is: ' + str(len(patientID)))
        
###############################################################################

def PrintAverageNumberOfTumorsPerArm(rawDataPath, ArmName = 'TRT01A', split = True):
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    for arm in receivedTreatment_unique:    
        tumorNoList = []
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan'] 
            temp = [i for i in temp if '-' in i]
            temp = [str(i).split('-')[-1] for i in temp]
            temp = list(set(temp))
            tumorNoList.append(len(temp)) 
            #print(temp)
        print('Average Number of Tumors Per Patient for arm ' + arm + ' is: ' + str(np.round(np.nanmean(tumorNoList))))

###############################################################################

def PrintAverageNumberOfMeasurementsPerArm(rawDataPath, ArmName = 'TRT01A', split = True):
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    for arm in receivedTreatment_unique:   
        measurementList = []
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan'] 
            temp = [i for i in temp if '-' in i]
            #temp = [str(i).split('-')[-1] for i in temp]
            #temp = list(set(temp))
            tempList = []
            for t in temp:
                tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == t]
                tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)   
                a = tumorFiltered_Data['TRTESTCD'] == 'LDIAM'
                b = tumorFiltered_Data['TRTESTCD'] == 'SAXIS'
                c = a | b
                tumorFiltered_Data = tumorFiltered_Data.loc[c]

                tempList.append(len(tumorFiltered_Data))
            measurementList.append(np.nanmean(tempList))                
        print('Average Number of Measurements Per Patient for arm ' + arm + ' is: ' + str(np.round(np.nanmean(measurementList))))

###############################################################################

def PlotPatientPerStudy(rawDataPath, studyName, item = 0, normalizeDimension = True):

    data = pd.read_excel(rawDataPath)    
    patientID = list(data['USUBJID'].unique())
    key = patientID[item]
    filteredData = data.loc[data['USUBJID'] == key]
    temp = filteredData['TRLINKID'].unique()
    temp = [i for i in temp if not str(i) == 'nan']
    temp = [i for i in temp if not '-NT' in str(i)]
    
    plt.figure()
    c = ['#9e0142', '#3288bd', '#f46d43', '#313695']
    #c = ['#9e0142', '#3288bd', '#f46d43', '#66c2a5', '#5e4fa2']
    #c = ['#9e0142', '#3288bd', '#f46d43', '#66c2a5']
    #c = ['#9e0142', '#3288bd']
    #c = ['#9e0142', '#3288bd', '#f46d43', '#66c2a5','#006837' , '#abdda4', '#fdae61']

    index = 0
    for tumor in temp:
        tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == tumor]
        tumorFiltered_Data.dropna(subset=['TRDY'], inplace = True)            
        tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']            

        dimension = list(tumorFiltered_Data['TRORRES'])
        time = list(tumorFiltered_Data['TRDY'])
        
        #time = utils.Correct_Time_Vector(time, convertToWeek = False)

        # If the value of Dimension is nan or any other string value, we replace it with zero    
        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)

        if not len(dimension)== 0:
            dimension = [x for _,x in sorted(zip(time,dimension))]
            if normalizeDimension:
                dimension= dimension/np.max(dimension)
            time.sort()
                        
            plt.plot(time, dimension, '.-', markeredgewidth = 6, linewidth = 4, color = c[index])
            plt.xlabel('Time (days)', fontsize = 30)
            plt.ylabel('LD (mm)', fontsize = 30)
            plt.title(studyName, fontsize = 30)
            plt.xticks(fontsize = 30)
            plt.yticks(fontsize = 30)
            index += 1
            
###############################################################################   
            
# Create Legend
fig = plt.figure()
ax = fig.add_subplot(111)      
custom_lines = [Line2D([0], [0], color='#9e0142', lw=7),
               Line2D([0], [0], color='#3288bd', lw=7),
               Line2D([0], [0], color='#f46d43', lw=7),
               Line2D([0], [0], color='#66c2a5', lw=7),
               Line2D([0], [0], color='#5e4fa2', lw=7),
               Line2D([0], [0], color='#006837', lw=7),
               Line2D([0], [0], color='#abdda4', lw=7),
               Line2D([0], [0], color='#fdae61', lw=7),
               Line2D([0], [0], color='#313695', lw=7)]

ax.legend(custom_lines, ['INV_T001', 'INV_T002', 'INV_T003',
                         'INV_T004', 'INV_T005', 'RAD1-T001', 'RAD1-T002', 'RAD1-T003', 'INV-NEW201'], fontsize = 30)

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.draw()

###############################################################################

def PlotSpiderPlot(rawDataPath, studyName, numberOfPatients = 50):

    data = pd.read_excel(rawDataPath)    
    patientID = list(data['USUBJID'].unique())
    count = 0
    item = 0
    fig, ax = plt.subplots()
    
    while count <= numberOfPatients: 
        
        key = patientID[item]
        filteredData = data.loc[data['USUBJID'] == key]
        temp = filteredData['TRLINKID'].unique()
        temp = [i for i in temp if not str(i) == 'nan']
        temp = [i for i in temp if not '-NT' in str(i)]
        
        if  'INV-T001' in temp :
            tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
            tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
            tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
            
            if len(tumorFiltered_Data) >= 2:
                dimension = list(tumorFiltered_Data['TRORRES'])
                time = list(tumorFiltered_Data['TRDY'])
                
                time = utils.Correct_Time_Vector(time, convertToWeek = True)

                # If the value of Dimension is nan or any other string value, we replace it with zero    
                dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
                
                dimension = [x for _,x in sorted(zip(time,dimension))]
                time.sort()
                trend = utils.Detect_Tend_Of_Data(dimension)   
                new_dim = [dimension[0]] * len(dimension)

                change = [a_i - b_i for a_i, b_i in zip(dimension, new_dim)]

                if trend ==  'Up':
                    ax.plot(time, change, marker='o', markeredgewidth = 5, linewidth = 4, color='#d73027')
                elif trend == 'Down':
                    ax.plot(time, change, marker='o', markeredgewidth = 5, linewidth = 4, color='#1a9850')
                else:
                    ax.plot(time, change, marker='o', markeredgewidth = 5, linewidth = 4, color='#313695')
                count += 1
        item += 1
    
    
    plt.axhline(0, linestyle = '--', color='black')
    plt.xlabel('Time (days)', fontsize = 30)
    plt.ylabel('Change in  LD From Baseline(mm)', fontsize = 30)
    plt.title(studyName, fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    
    custom_lines = [Line2D([0], [0], color='#d73027', lw=4),
                    Line2D([0], [0], color='#1a9850', lw=4),
                    Line2D([0], [0], color='#313695', lw=4)]
    
    ax.legend(custom_lines, ['Up', 'Down', 'Fluctuate'], fontsize = 30)
    
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
###############################################################################

def ProbablityPredictionOfLastPointFromFirstPoints(rawDataPath, ArmName = 'TRT01A', pointNo = 1, split = True):
    probs = []
    
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    for arm in receivedTreatment_unique: 
        predResponse = []
        realResponse = []
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan']
            temp = [i for i in temp if not '-NT' in str(i)]
            
            if  'INV-T001' in temp :
                tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
                tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
                
                # Limit the Data Points for 6 and bigger!
                if len(tumorFiltered_Data) >= 6:
                    dimension = list(tumorFiltered_Data['TRORRES'])
                    time = list(tumorFiltered_Data['TRDY'])
                    
                    time = utils.Correct_Time_Vector(time, convertToWeek = True)
    
                    # If the value of Dimension is nan or any other string value, we replace it with zero    
                    dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
                    
                    dimension = [x for _,x in sorted(zip(time,dimension))]
                    time.sort()
                    
                    tempDiff = dimension[pointNo] - dimension[0]
                    if dimension[pointNo] == 0:
                        predResponse.append('CR')
                    elif tempDiff <=  - 0.3 * dimension[0]:
                        predResponse.append('PR')
                    elif tempDiff >= 0.2 * np.min(dimension[0:pointNo]):
                        predResponse.append('PD')
                    else:
                        predResponse.append('SD')
                    
                    finalPointDiff = dimension[-1] - dimension[0]
                   
                    if dimension[-1] == 0:
                        realResponse.append('CR')
                    elif finalPointDiff <=  - 0.3 * dimension[0]:
                        realResponse.append('PR')
                    elif finalPointDiff >= 0.2 * np.min(dimension):
                        realResponse.append('PD')
                    else:
                        realResponse.append('SD')
                
        counter = 0
        for i in range(len(realResponse)):
            if predResponse[i] == realResponse[i]:
                counter += 1
        #print('Probbaility to Predict the treatment response from point ' + str(pointNo) + ' is: ' +  str(np.round(counter / len(realResponse)*100)))
        probs.append(np.round(counter / len(realResponse)*100))
    return probs

###############################################################################

def ProbablityPredictionOfLastPointFromFirstPoints_FromDataSet (rawDataPath, ArmName = 'TRT01A', pointNo = 1, split = True):
    
    probs = []
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    for arm in receivedTreatment_unique: 
        predResponse = []
        realResponse = []
        print(arm)        
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan']
            temp = [i for i in temp if not '-NT' in str(i)]
            
            if  'INV-T001' in temp :
                tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
                tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
                
                # Limit the Data Points for 6 and bigger!
                if len(tumorFiltered_Data) >= 6:
                    tempProg = list(tumorFiltered_Data['RSORRES_R']) 
                    time = list(tumorFiltered_Data['TRDY'])
                    
                    time = utils.Correct_Time_Vector(time, convertToWeek = True)
                    tempProg = [x for _,x in sorted(zip(time,tempProg))]
                    predResponse.append(tempProg[pointNo + 1])
                    realResponse.append(tempProg[-1])
        
        counter = 0
        for i in range(len(realResponse)):
            if predResponse[i] == realResponse[i]:
                counter += 1
        print('Probbaility to Predict the treatment response from point ' + str(pointNo) + ' is: ' +  str(np.round(counter / len(realResponse)*100)))
        probs.append(np.round(counter / len(realResponse)*100))
    
    return probs

###############################################################################

def ClaculateNumberOfNewMetastaseInEachArm(rawDataPath, ArmName = 'TRT01A', split = True):
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    for arm in receivedTreatment_unique: 
        counter = 0
        print(arm)        
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
        
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan']
            temp = [i for i in temp if not '-NT' in str(i)]
            
            if  'INV-T001' in temp :
                tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
                tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
                
                # Limit the Data Points for 6 and bigger!
                if len(tumorFiltered_Data) >= 6:
                    print(key)
                    filteredData = data.loc[data['USUBJID'] == key]
                    temp = list(filteredData['TRLINKID'].unique())
                    temp = [i for i in temp if not str(i) == 'nan']
                    for t in temp:
                        if 'NEW' in t:
                            counter += 1
        print('Number of New metastase in arm ' +  arm +  ' is: ' +  str(counter))

###############################################################################
        
def CreateTrendDictionary(rawDataPath, ArmName = 'TRT01A', split = True, normalizeDimension = True):
    data = pd.read_excel(rawDataPath)    
    receivedTreatment = data[ArmName]   
    if split:
        receivedTreatment = [str(i).split(' ')[0] for i in receivedTreatment]
    receivedTreatment_unique = list(set(receivedTreatment))
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'nan']
    receivedTreatment_unique = [i for i in receivedTreatment_unique if not i == 'Not']
    data['receivedTreatment'] = receivedTreatment
    result_dict = utils.Create_Result_dict(receivedTreatment_unique, ['Up', 'Down', 'Fluctuate'], categories = ['patientID', 'time', 'dimension'])
    
    
    for arm in receivedTreatment_unique: 
        data_temp = data.loc[data['receivedTreatment'] == arm]    
        patientID = list(data_temp['USUBJID'].unique())
    
        for key in patientID:
            filteredData = data.loc[data['USUBJID'] == key]
            temp = filteredData['TRLINKID'].unique()
            temp = [i for i in temp if not str(i) == 'nan']
            temp = [i for i in temp if not '-NT' in str(i)]
            
            if  'INV-T001' in temp :
                tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
                tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
                
                 # Limit the Data Points for 6 and bigger!
                if len(tumorFiltered_Data) >= 6:
                    dimension = list(tumorFiltered_Data['TRORRES'])
                    time = list(tumorFiltered_Data['TRDY'])
                    
                    time = utils.Correct_Time_Vector(time, convertToWeek = True)
    
                    # If the value of Dimension is nan or any other string value, we replace it with zero    
                    dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
                    
                    dimension = [x for _,x in sorted(zip(time,dimension))]
                    if normalizeDimension:
                        dimension = dimension/np.max(dimension)
                    time.sort()
                                
                    trend = utils.Detect_Tend_Of_Data(dimension)
                    result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID', 'time', 'dimension'],
                                                          values = [key, time, dimension])

    return result_dict

###############################################################################
    
























        