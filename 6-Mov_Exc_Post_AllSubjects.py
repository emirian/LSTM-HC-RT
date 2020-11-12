
# def Test_Temporal3Parts (Model, subband, PatientID, part):
    
#     best_model = load_model(Model)
    

#     if subband == 'full':
#        data  = sio.loadmat('shamdata_tlgo')
#        HC    = data['shamhceeg'][:,PatientID] 
#     else:
#        data  = sio.loadmat('DatahcAlireza')
#        HC     = data[subband][:,PatientID] # Healthy Control

#     # Load the EEG data labels
#     labels  = sio.loadmat('task_gvssham')
#     HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
#     ALL_EEG = HC
#     ALL_Beh = HC_labels


#     num_chans,time_steps,num_trials = ALL_EEG[0].shape
#     X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
#     x = ALL_EEG[0]
#     X_ALL_EEG[:,:,:] = x
#     # Data transpose     
#     X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
#     # Data dimensions          
#     p,m,n,d = X_ALL_EEG.shape
#     # Data reshape     
#     X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  
#     X2_ALL_EEG =  np.zeros((p* m,n,d)) 

#     # # # # # Load the EEG data labels # # # # # 
#     num_trials, num_behvars = ALL_Beh[0].shape
#     movement_time_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
#     reaction_time_ALL = np.zeros((ALL_EEG.size, num_trials,1))                

#     y_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
#     ALLBeh = ALL_Beh[0]
#     for tr in range (num_trials):
#         i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
#         y_ALL[0,tr] = ALLBeh[tr,i]
        
        
#     for tr in range (num_trials):
#         i = 6 #movement time
#         movement_time_ALL[0,tr] = ALLBeh[tr,i]
        
#         i = 12
#         reaction_time_ALL[0,tr] = ALLBeh[tr,i]
        
#         if part == 1:
#             t1 = 499
#             t2 = int(500 + movement_time_ALL[0,tr])
#             temp_con = np.concatenate((np.zeros(t1), np.ones(t2-t1), np.zeros(2000-t2)), axis=0)
#         elif part == 2:
#             t2 = int(500 + movement_time_ALL[0,tr])
#             t4 = int(500 + movement_time_ALL[0,tr] + reaction_time_ALL[0,tr])
#             temp_con = np.concatenate((np.zeros(t2), np.ones(t4-t2), np.zeros(2000-t4)), axis=0)
#         else:
#              t4 = int(500 + movement_time_ALL[0,tr] + reaction_time_ALL[0,tr])
#              temp_con = np.concatenate((np.zeros(t4), np.ones(2000 - t4)), axis=0)
   
#         repetitions = 27
#         repeats_array = np.transpose(np.tile(temp_con, (repetitions,1)))
#         temp = X_ALL_EEG[tr,:,:]
#         X2_ALL_EEG[tr,:,:] = temp * repeats_array

    
#     y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


#     # # # # # Data augmentation (Down-sampling by 2) # # # # # 
#     X_ALL_EEG_DS1 = X2_ALL_EEG[:,0::2,:]
#     X_ALL_EEG_DS2 = X2_ALL_EEG[:,1::2,:]
#     X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
#     # labels
#     y_ALL_DS = np.concatenate((y_ALL,y_ALL))

#     # # # # # no Shuffle the data # # # # # 
#     N = X_ALL_EEG_DS.shape[0]
#     indices = [i for i in range(N)]
#     #shuffle(indices)
#     X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
#     y_ALL_DS     = y_ALL_DS[indices,]

#     # Sanity Check: Exclude labels with NaN values
#     # Indices of nan and inf values
#     idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
#     filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
#     filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
#     X = filtered_X_ALL_EEG_DS
#     y = filtered_y_ALL_DS
    
#     # Make X and y as float 32
#     X = X.astype('float32')
#     y = y.astype('float32')
    
#     # normalize the dataset
#     scaler = MinMaxScaler(feature_range=(0, 1))
    
#     # Standarizing X
#     m, n, d = X.shape
#     X = X.reshape(m, n*d)
#     X = scaler.fit_transform(X)
#     X = X.reshape(m, n, d)
#     # Standarizing y
#     y = scaler.fit_transform(y)
    

#     # make predictions
#     y_Predicted = best_model.predict(X)

#     MSE_Test = mean_squared_error(y, y_Predicted)
#     R2_score_Test = r2_score(y, y_Predicted)
#     #return (y-y_Predicted)    
#     return(MSE_Test)

import MyUtils
import matplotlib.pyplot as plt
import numpy as np


# Removed 8,9 i.e. 7, 8
IDs = [0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,19,21]


PATH = "C:/Elham/EEG_PatientIdentification/P1/P1_SHAM_TASK_AK/from GPU_OLD/Models/"
subband = 'full'  

ef1 = np.zeros(22) 
ef2 = np.zeros(22) 
ef3 = np.zeros(22) 
    
for pt in IDs:
        print ("pt= " + str(pt+1))
        FileName_BestModel = PATH + 'BestModel_full_excl' + str(pt) + '.h5'
        part = 1
        ef1[pt] = MyUtils.Test_Temporal3Parts(FileName_BestModel, subband, pt, part)
        
        part = 2
        ef2[pt] = MyUtils.Test_Temporal3Parts(FileName_BestModel, subband, pt, part)
    
        part = 3
        ef3[pt] = MyUtils.Test_Temporal3Parts(FileName_BestModel, subband, pt, part)
        
new_ef1 = [n for n in ef1 if n > 0]
new_ef2 = [n for n in ef2 if n > 0]
new_ef3 = [n for n in ef3 if n > 0]

data = [new_ef1, new_ef2, new_ef3]
fig, ax = plt.subplots()
ax.set_title('(Averaged over all Trials and Subjects)')
ax.boxplot(data)
ax.set_xticklabels(['Mot.Prep.', 'Mot.Exec.', 'Post-Mov'])
plt.ylabel("MSE")
PlotFileName = "Result_July_3Parts.PNG"
plt.savefig(PlotFileName)


import scipy as sp
[s,p12] = sp.stats.ttest_ind(new_ef1, new_ef2)
print (p12)
[s,p13] = sp.stats.ttest_ind(new_ef1, new_ef3)
print(p13)
[s,p23] = sp.stats.ttest_ind(new_ef2, new_ef3)
print(p23)


