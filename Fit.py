
# -*- coding: utf-8 -*-
"""
Created on Jan - 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

import math
import numpy as np
from scipy.optimize import curve_fit
import warnings
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import Utils as utils
import FitFunctions as ff
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Normalizer
import matplotlib as mpl
from matplotlib.lines import Line2D

###############################################################################

studies = ['Study1', 'Study2', 'Study3', 'Study4', 'Study5']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate']

###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []

for studyName in studies:
    rawDataPath = os.path.join(r"D:\\", studyName + '_m.xlsx')
    sind = studies.index(studyName)
    sp = splits[sind]
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']
    filtered_Data = filtered_Data.loc[filtered_Data['TRTESTCD'] == 'LDIAM']
    temp = list(filtered_Data['TRORRES'])
    temp = utils.Remove_String_From_Numeric_Vector(temp, valueToReplace = 0)
    maxList.append(max(temp))
    minList.append(min(temp))    
               
###############################################################################

# Fit Funtions to the Data Points
    
maxi = np.max(maxList)

studies = ['Study1', 'Study2', 'Study3', 'Study4', 'Study5']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
noPars = [3, 3, 3, 4, 3, 4]
count = 0

for studyName in studies:
    sind = studies.index(studyName)    
    sp = splits[sind]    
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True
    #count_Dict[studyName] = {}
    rawDataPath = os.path.join(r"D:\\", studyName + '_m.xlsx')
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    for functionToFit in functions:
        
        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = utils.Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate'], categories = ['patientID','time', 'dimension', 'prediction', 'mse',
                                               'rSquare','aic', 'params', 'cancer', 'mae', 'rmse'])
        print(functionToFit)
        print(studyName)

        for arm in arms:
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
                    keysList = []
                    if len(tumorFiltered_Data) >= 6:   
                                                
                        dimension = list(tumorFiltered_Data['TRORRES'])
                        time = list(tumorFiltered_Data['TRDY'])
                        
                        time = utils.Correct_Time_Vector(time, convertToWeek = True)
        
                        # If the value of Dimension is nan or any other string value, we replace it with zero    
                        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
                        
                        dimension = [x for _,x in sorted(zip(time,dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy/maxi
                            
                        trend = utils.Detect_Trend_Of_Data(dimension_copy)
                        
                        dimension = [i * i * i * 0.5 for i in dimension]
                        if normalizeDimension:
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.5, 0])

                        time.sort()                        
                        cn =   list(tumorFiltered_Data['TULOC']) [0]                          
                        param_bounds=([0] *noParameters ,[np.inf] * noParameters)
                        if find == 4:
                            param_bounds=([[0, -np.inf, 0],[np.inf, np.inf, np.inf]])
                        elif find == 5:
                            param_bounds=([[0, -np.inf, 2/3, 0],[np.inf, np.inf, 1, np.inf]])
                        elif find == 3:
                            param_bounds=([[0, 0, 2/3, 0],[np.inf, np.inf, 1, np.inf]])
                            
                        firstDim = dimension[0:-3]
                        firstTime = time[0:-3]
                        try:
                            fitfunc = ff.Select_Fucntion(functionToFit)
                            geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = firstTime, d = firstDim)
                            fittedParameters, pcov = curve_fit(fitfunc, firstTime, firstDim, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf') 
                            modelPredictions = fitfunc(time, *fittedParameters)                             
                        except:
                            try:
                                geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = firstTime, d = firstDim)
                                if len(param_bounds[0]) == 4:
                                   geneticParameters[0] = 0.001
                                   geneticParameters[1] = 0.001
                                   geneticParameters[2] = 0.7
                                else:
                                   geneticParameters[0] = 0.001
                                   geneticParameters[1] = 0.7                                   
                                fittedParameters, pcov = curve_fit(fitfunc, firstTime, firstDim, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf') 
                                modelPredictions = fitfunc(time, *fittedParameters) 
                            except:
                                print(key)
                                result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'mse', 'rSquare','aic', 'params', 'cancer', 'mae', 'rmse'], 
                                                                          values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn, np.nan, np.nan])
                                continue
                        
                        if len(set(dimension)) == 1:
                            modelPredictions = dimension
                        else:
                            modelPredictions = fitfunc(time, *fittedParameters) 
                        
                        modelPredictions = [0 if str(i) == 'nan' else i  for i in modelPredictions]
                        absError = modelPredictions - dimension
                        SE = np.square(absError)
                        temp_sum = np.sum(SE)

                        result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'mse', 'rSquare','aic', 'params', 'cancer', 'mae', 'rmse'], 
                                                                          values = [key, time, dimension, modelPredictions, mean_squared_error(dimension, modelPredictions),
                                                                                    r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), fittedParameters, cn,
                                                                                    mean_absolute_error(dimension, modelPredictions), math.sqrt(mean_squared_error(dimension, modelPredictions))])
                        
                        
        a_file = open(os.path.join(r"D:\Spider Project\Fit\NEW_MAY\3PointFit_nonNorm", functionToFit, studyName + '.pkl'), "wb")
        pickle.dump(result_dict, a_file)
        a_file.close()
           
###############################################################################
# Plot HeatMaps
###############################################################################
        
result = pd.DataFrame()
for f in functions:
    temp = []
    indices = []
    for s in studies:   
        result_dict = pickle.load( open( r"D:\\" + f + '\\' + s + ".pkl", "rb" ) )
        arms = list(result_dict.keys())        
        arms.sort()
        for arm in arms:
            for trend in trends:
                indices.append(arm + '_' + trend)
                temp.append(np.around(np.mean(result_dict[arm][trend]['aic']), 10))
    result[f] = temp
    
result.index = indices
result.dropna(inplace = True)
minValuesObj = result.min(axis=1)


tab_n = result.div(result.max(axis=1), axis=0)
cmap = sns.cm.rocket_r
mpl.rcParams['font.size'] = 10
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.figure()
plt.tight_layout()
#t = tab_n.T
ax = sns.heatmap(tab_n, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True ,
                  square = False, annot = True)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xticklabels(labels = functions, rotation = 30,fontsize = 10)
plt.title('AIC value for fit to the full timeline', fontsize = 20)

###############################################################################

# Calculate MAE for final point prediction
        
result = pd.DataFrame()

for f in functions:
    temp = []
    indices = []
    for s in studies:   
        result_dict = pickle.load( open( r"D:\\" + f + '\\' + s + ".pkl", "rb" ) )
        arms = list(result_dict.keys())        
        arms.sort()
        for arm in arms:
            for trend in trends:
                indices.append(arm + '_' + trend)
                content = result_dict[arm][trend]['dimension']
                g = []
                for i in range(len(content)):
                    if not str(result_dict[arm][trend]['prediction'][i]) == 'nan':
                        g.append(np.nanmean(abs(content[i][-3:] - result_dict[arm][trend]['prediction'][i][-3:])))
                temp.append(np.nanmean(g))                
    result[f] = temp
    
result.index = indices
result.dropna(inplace = True)
tab_n = result.div(result.max(axis=1), axis=0)
cmap = sns.cm.rocket
mpl.rcParams['font.size'] = 10
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.figure()
plt.tight_layout()
t = tab_n.T
ax = sns.heatmap(result, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True ,
                  square = True, annot = True)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xticklabels(labels = functions, rotation = 30,fontsize = 10 )
plt.title('MAE values for each arm', fontsize = 20 )
              
###############################################################################

