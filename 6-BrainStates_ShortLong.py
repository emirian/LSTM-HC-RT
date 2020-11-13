
import MyUtils
import matplotlib.pyplot as plt
import numpy as np


# Removed 8,9 i.e. 7, 8
IDs = [0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,19,21]


PATH = "C:/Elham/EEG_PatientIdentification/P1/P1_SHAM_TASK_AK_Saliency/from GPU_OLD/Models/"
subband = 'full'  

ef1_S = np.zeros(22) 
ef2_S = np.zeros(22) 
ef3_S = np.zeros(22) 
    

for pt in IDs:
        print ("pt= " + str(pt+1))
        FileName_BestModel = PATH + 'BestModel_full_excl' + str(pt) + '.h5'
        part = 1
        ef1_S[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'S')
        
        part = 2
        ef2_S[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'S')
        
        part = 3
        ef3_S[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'S')
        
        
new_ef1_S = [n for n in ef1_S if n > 0]
new_ef2_S = [n for n in ef2_S if n > 0]
new_ef3_S = [n for n in ef3_S if n > 0]


data = [new_ef1_S, new_ef2_S, new_ef3_S]
fig, ax = plt.subplots()
ax.set_title('(Averaged over Shortest Trials of all Subjects)')
ax.boxplot(data)
ax.set_xticklabels(['Standby', 'Preparatory', 'Execution'])
plt.ylabel("MSE")
PlotFileName = "Result_3Parts_Short.PNG"
plt.savefig(PlotFileName)


import scipy as sp
[s,sp12] = sp.stats.ttest_ind(new_ef1_S, new_ef2_S)
print (sp12)

[s,sp13] = sp.stats.ttest_ind(new_ef1_S, new_ef3_S)
print (sp13)

[s,sp23] = sp.stats.ttest_ind(new_ef2_S, new_ef3_S)
print (sp23)


#---------------------------------------------------------------------
  
ef1_L = np.zeros(22) 
ef2_L = np.zeros(22) 
ef3_L = np.zeros(22) 
    

for pt in IDs:
        print ("pt= " + str(pt+1))
        FileName_BestModel = PATH + 'BestModel_full_excl' + str(pt) + '.h5'
        part = 1
        ef1_L[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'L')
        
        part = 2
        ef2_L[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'L')
        
        part = 3
        ef3_L[pt] = MyUtils.Test_Temporal3Parts_6bars_ShortLong(FileName_BestModel, subband, pt, part, 'L')
      
        

new_ef1_L = [n for n in ef1_L if n > 0]
new_ef2_L = [n for n in ef2_L if n > 0]
new_ef3_L = [n for n in ef3_L if n > 0]


data = [new_ef1_L, new_ef2_L, new_ef3_L]
fig, ax = plt.subplots()
ax.set_title('(Averaged over Longest Trials of all Subjects)')
ax.boxplot(data)
ax.set_xticklabels(['Standby', 'Preparatory', 'Execution'])
plt.ylabel("MSE")
PlotFileName = "Result_3Parts_Long.PNG"
plt.savefig(PlotFileName)


import scipy as sp
[s,lp12] = sp.stats.ttest_ind(new_ef1_L, new_ef2_L)
print (lp12)

[s,lp13] = sp.stats.ttest_ind(new_ef1_L, new_ef3_L)
print (lp13)

[s,lp23] = sp.stats.ttest_ind(new_ef2_L, new_ef3_L)
print (lp23)
