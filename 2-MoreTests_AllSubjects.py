import MyUtils
import pandas as pd
import numpy as np
from openpyxl import load_workbook
SBs = ['bandAlpha'] #, 'bandBeta', 'bandDelta', 'bandGamma', 'bandTheta']
    
IDs = [0,1]
#IDs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

PATH = "C:/Elham/ONGOING ISSUES/Martin McKeown/EEG_PatientIdentification/P1_SHAM_TASK_AK/from GPU/Models/"
wb = load_workbook("Results4AK.xlsx")
        
for pt in IDs:
    
    for sb in SBs:
        subband = sb
        print ("pt= " + str(pt+1))
        FileName_BestModel = PATH + 'BestModel_full_excl' + str(pt) + '.h5'
        rt1 = MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)
        pd_array = pd.DataFrame(rt1)
        pd_array.to_excel('Results4AK.xlsx', sheet_name=subband+'Filtered', startcol= pt + 1, startrow = 2)
        

        rt2 = MyUtils.NewTest(FileName_BestModel, subband, pt)
        pd_array = pd.DataFrame(rt2)
        pd_array.to_excel('Results4AK.xlsx', sheet_name=subband, startcol= pt + 1, startrow = 2)


wb.save('Results4AK.xlsx')
                
  
    