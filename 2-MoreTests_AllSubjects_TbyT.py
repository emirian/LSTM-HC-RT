import MyUtils

import xlsxwriter 
SBs = ['bandAlpha', 'bandBeta', 'bandDelta', 'bandGamma', 'bandTheta']
    
IDs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

#PATH = "C:/Elham/ONGOING ISSUES/Martin McKeown/EEG_PatientIdentification/P1_SHAM_TASK_AK/from GPU/Models/"
PATH = ""
#------------------- Train on Full-----------------------
workbook = xlsxwriter.Workbook('TrainOnFull_Results4AK.xlsx') 
for sb in SBs:
        subband = sb    
        worksheet = workbook.add_worksheet(subband+'Filtered') 

        for pt in IDs:
    
            print ("pt= " + str(pt+1))
            FileName_BestModel = PATH + 'BestModel_full_excl' + str(pt) + '.h5'
            rt1 = list(MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt))
      
            row = 1
            column = pt+1
            for item in rt1: 
                worksheet.write(row, column, item) 
                row += 1
           
    
            rt2 = MyUtils.NewTest(FileName_BestModel, subband, pt)
            
            row = 1
            column = pt+1
            for item in rt1: 
                worksheet.write(row, column, item) 
                row += 1  
workbook.close() 

#------------------- Train on sub-bands-----------------------
workbook = xlsxwriter.Workbook('TrainOnSB_Results4AK.xlsx') 
for sb in SBs:
        subband = sb    
        worksheet = workbook.add_worksheet(subband+'Filtered') 

        for pt in IDs:
    
            print ("pt= " + str(pt+1))
            FileName_BestModel = PATH + 'BestModel_'+ subband +'_excl' + str(pt) + '.h5'
            rt1 = list(MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt))
      
            row = 1
            column = pt+1
            for item in rt1: 
                worksheet.write(row, column, item) 
                row += 1
           
    
            rt2 = MyUtils.NewTest(FileName_BestModel, subband, pt)
            
            row = 1
            column = pt+1
            for item in rt1: 
                worksheet.write(row, column, item) 
                row += 1  
workbook.close() 