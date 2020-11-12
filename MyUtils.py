import scipy.io as sio
import scipy as sp
from scipy.signal import savgol_filter
from scipy import signal
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Activation, Flatten, TimeDistributed, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import optimizers


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import openpyxl
import mne.viz as mv
from scipy.fftpack import rfft, irfft

import tensorflow as tf
from tensorflow.keras import backend as K

def subband2Index (subband):
    switcher = {
            'bandAlpha' : 2,
            'bandAlphaFiltered':3,
            'bandBeta': 4,
            'bandBetaFiltered':5,
            'bandDelta':6,
            'bandDeltaFiltered': 7,
            'bandGamma': 8,
            'bandGammaFiltered': 9,
            'bandTheta': 10,
            'bandThetaFiltered': 11,
            'full': 12,
            }
    return switcher.get(subband)

def TrainTest(SubjectID2Exclude, subband):
    ExcCol = subband2Index(subband)
    IDs = list(range(22))
    #IDs = np.delete(IDs, SubjectID2Exclude)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,IDs] 
    else:
 # Load the EEG data
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,IDs] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,IDs]

    fs_orig = 1000 # original sampling rate = 1000Hz
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0,1].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))

    for pt in IDs: # pt=patient
        x = ALL_EEG[0,pt]
        X_ALL_EEG[pt,:,:,:] = x
    
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0,1].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            
    for pt in IDs:
        #print ("pt= " + str(pt+1))
        for tr in range (num_trials):
            i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
            y_ALL[pt,tr] = ALL_Beh[0,pt][tr,i]
        
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 

    #------- Removing the Subject2Exclude--------------
    X_ALL_EEG_SAVED = np.delete(X_ALL_EEG, np.s_[SubjectID2Exclude*num_trials:(SubjectID2Exclude+1)*num_trials], 0)
    y_ALL_SAVED = np.delete(y_ALL, np.s_[SubjectID2Exclude*num_trials:(SubjectID2Exclude+1)*num_trials], 0)
    
    X_ALL_EEG = X_ALL_EEG_SAVED
    y_ALL = y_ALL_SAVED
    
    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # Shuffle the data - all patients # # # # # 
    from random import shuffle
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]


    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    X = np.nan_to_num(X)
    # # # # # Hold-out (Training-Testing: 80%-20%) # # # # # 
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=42)

    model = Sequential()
    model.add(LSTM(24, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, implementation=2))
    model.add(TimeDistributed(Dense(1)))
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    print(model.summary())
    
    # Callback
    
    # simple early stopping
    ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500) 
    
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(SubjectID2Exclude) + '.h5'
    checkpoint_name = FileName_BestModel
    MC = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [ES, MC]
    
    
    # Model Fitting
    history = model.fit(X_train, y_train, epochs= 1000, batch_size= 80, validation_data=(X_test, y_test), shuffle=False, callbacks=callbacks_list, verbose=1)
    
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    
    LossPlotFileName = 'Loss' + subband + '_excl' + str(SubjectID2Exclude) + '.png'

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(LossPlotFileName)
    
    # load the saved model
    best_model = load_model(FileName_BestModel)
    
    # make predictions
    y_train_Predicted = best_model.predict(X_train)
    y_test_Predicted = best_model.predict(X_test)
    
    TrainPlotFileName = 'Train' + subband + '_excl' + str(SubjectID2Exclude) + '.png'
    TestPlotFileName  = 'Test'  + subband + '_excl' + str(SubjectID2Exclude) + '.png'
    
    # Plot Original and Predicted Train Labels
    plt.clf()
    plt.plot(y_train)
    plt.plot(y_train_Predicted)
    plt.title('Original and Predicted Train Labels based only on' + subband)
    plt.ylabel('Response Time')
    plt.xlabel('epoch')
    plt.legend(['Original', 'Predicted'], loc='upper left')
    plt.savefig(TrainPlotFileName)

    
    # Plot Original and Predicted Test Labels
    plt.clf()
    plt.plot(y_test)
    plt.plot(y_test_Predicted)
    plt.title('Original and Predicted Test Labels based only on' + subband)
    plt.ylabel('Response Time')
    plt.xlabel('epoch')
    plt.legend(['Original', 'Predicted'], loc='upper left')
    plt.savefig(TestPlotFileName)
    
    
    # Evaluate the prediction performanceof on the Standarized Data
    
    MSE_Train = mean_squared_error(y_train, y_train_Predicted)
    MSE_Test = mean_squared_error(y_test, y_test_Predicted)
    R2_score_Train = r2_score(y_train, y_train_Predicted)
    R2_score_Test = r2_score(y_test, y_test_Predicted)

    
    srcfile = openpyxl.load_workbook('Results.xlsm',read_only=False, keep_vba= True)#to open the excel sheet and if it has macros

    sheetname = srcfile.get_sheet_by_name('Train_MSE')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(MSE_Train,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    sheetname = srcfile.get_sheet_by_name('Train_R2')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(R2_score_Train,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops

    sheetname = srcfile.get_sheet_by_name('Test_MSE')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(MSE_Test,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    sheetname = srcfile.get_sheet_by_name('Test_R2')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(R2_score_Test,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    
    
    # Evaluate the prediction performanceof on the Original Data
    
    # invert predictions
    y_train_Predicted = scaler.inverse_transform(y_train_Predicted)
    y_train_Original = scaler.inverse_transform(y_train)
    
    y_test_Predicted = scaler.inverse_transform(y_test_Predicted)
    y_test_Original  = scaler.inverse_transform(y_test)
    
    
    TrainPlotFileName_T = 'T_Train' + subband + '_excl' + str(SubjectID2Exclude) + '.png'
    TestPlotFileName_T  = 'T_Test'  + subband + '_excl' + str(SubjectID2Exclude) + '.png'
    
    # Plot Original and Predicted Train Labels
    plt.clf()
    plt.plot(y_train_Original)
    plt.plot(y_train_Predicted)
    plt.title('Original and Predicted Train Labels based only on' + subband)
    plt.ylabel('Response Time')
    plt.xlabel('epoch')
    plt.legend(['Original', 'Predicted'], loc='upper left')
    plt.savefig(TrainPlotFileName_T)

    
    # Plot Original and Predicted Test Labels
    plt.clf()
    plt.plot(y_test_Original)
    plt.plot(y_test_Predicted)
    plt.title('Original and Predicted Test Labels based only on' + subband)
    plt.ylabel('Response Time')
    plt.xlabel('epoch')
    plt.legend(['Original', 'Predicted'], loc='upper left')
    plt.savefig(TestPlotFileName_T)
   
    
    
    MSE_Train = mean_squared_error(y_train_Original, y_train_Predicted)
    MSE_Test = mean_squared_error(y_test_Original, y_test_Predicted)
    
    R2_score_Train = r2_score(y_train_Original, y_train_Predicted)
    R2_score_Test = r2_score(y_test_Original, y_test_Predicted)
    
    sheetname = srcfile.get_sheet_by_name('Train_MSE_T')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(MSE_Train,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    sheetname = srcfile.get_sheet_by_name('Train_R2_T')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(R2_score_Train,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops

    sheetname = srcfile.get_sheet_by_name('Test_MSE_T')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(MSE_Test,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    sheetname = srcfile.get_sheet_by_name('Test_R2_T')
    sheetname.cell(row=SubjectID2Exclude+2,column=ExcCol).value = str(round(R2_score_Test,2)) #write to row 1,col 1 explicitly, this type of writing is useful to write something in loops
    
    srcfile.save('Results.xlsm')
      
    
#---------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
    
def NewTest (Model, subband, PatientID):
    
    ExcCol = subband2Index(subband)
    best_model = load_model(Model)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            

    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_Predicted = best_model.predict(X)

    
    MSE_Test = mean_squared_error(y, y_Predicted)
    R2_score_Test = r2_score(y, y_Predicted)
    
#    srcfile = openpyxl.load_workbook('Results.xlsm',read_only=False, keep_vba= True)
#
#    sheetname = srcfile.get_sheet_by_name('NewTest_MSE')
#    sheetname.cell(row=PatientID+2,column=ExcCol).value = str(round(MSE_Test,2)) 
#    sheetname = srcfile.get_sheet_by_name('NewTest_R2')
#    sheetname.cell(row=PatientID+2,column=ExcCol).value = str(round(R2_score_Test,2)) 
#
#    srcfile.save('Results.xlsm')
    return (y-y_Predicted)

from scipy.interpolate import interp2d

def CheckSpecGram_New (Model, PatientID):

    data  = sio.loadmat('shamdata_tlgo')
    HC    = data['shamhceeg'][:,PatientID] 

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    best_model = load_model(Model)
  
    ALL_EEG = HC
    ALL_Beh = HC_labels
    
    # # # # # Load the EEG data of PD_on # # # # # 
    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  
    
    
    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            
    
    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 
    
    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))
    
    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]
    
    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    
    
    # make predictions
    y_Predicted = best_model.predict(X)
    
    MSE_Test = mean_squared_error(y, y_Predicted)
    print('MSE_Test = %.3f' % (MSE_Test))
    R2_score_Test = r2_score(y, y_Predicted)
    print('R^2score_Test = %.3f' % (R2_score_Test))
    
    
    bin_labels = ['L', 'M', 'H']
    RT = pd.qcut(y[:,0], q=3, labels = bin_labels)
    
    Residue_test = abs(y_Predicted - y)
    RES = Residue_test[:,0]
    bin_labels = ['LE', 'ME', 'HE']
    Err = pd.qcut(RES, q=3, labels = bin_labels)
    
    LL = list(np.where((RT == 'L') & (Err == 'LE')))
    LM = list(np.where((RT == 'L') & (Err == 'ME')))
    LH = list(np.where((RT == 'L') & (Err == 'HE')))
    ML = list(np.where((RT == 'M') & (Err == 'LE')))
    MM = list(np.where((RT == 'M') & (Err == 'ME')))
    MH = list(np.where((RT == 'M') & (Err == 'HE')))
    HL = list(np.where((RT == 'H') & (Err == 'LE')))
    HM = list(np.where((RT == 'H') & (Err == 'ME')))
    HH = list(np.where((RT == 'H') & (Err == 'HE')))
    
    
    ALL1 = 0
    ALL2 = 0
    cnt1 = 0
    cnt2 = 0
    for sample_ind in np.nditer(LL):
        #for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
        # important channels
        for ch in (2,3,7,10, 13,14, 16, 17,21,26):

        #for ch in range (0,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt1 = cnt1 + 1
    ALL2 = ALL2 + ALL1
    cnt2 = cnt2 + 1
    
    ALL2_L = ALL2
  
    
    ALL1 = 0
    ALL2 = 0
    cnt3 = 0
    cnt4 = 0
    for sample_ind in np.nditer(np.concatenate((HL, ML), axis = 1)):
        for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
         #for ch in (2,3,7,10, 13,14, 16, 17,21,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt3 = cnt3 + 1
    ALL2 = ALL2 + ALL1
    cnt4 = cnt4 + 1
    
    ALL2_H = ALL2
    
    fig = plt.figure()    
    
    winSize = 100 #or 200 or 500
    winOverlap = round(winSize*.9) # or round(winSize*.8) or round(winSize*.5)
    f, t, Sxx = signal.spectrogram(ALL2_H/(cnt3*cnt4) - ALL2_L/(cnt1*cnt2), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])

    '''
    f = interp2d(t, f, Sxx, kind='linear')
    xnew = np.arange(0, 2.5, .01)
    ynew = np.arange(0, 60,.01)
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    img = plt.pcolormesh(Xn, Yn, data1, cmap='RdBu_r')
    '''
 
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.title("Diff of Spectrograms of Epochs with High/Med vs. Low ReactionTime for New Test Patient = " + str(PatientID + 1))

    fig.colorbar(img)
    SpecGram_FileName = "SpecGram_Diff" + str(PatientID)+ '.png'
    plt.savefig(SpecGram_FileName)

    plt.clf()
    f, t, Sxx = signal.spectrogram(ALL2_H/(cnt3*cnt4), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])
    fig.colorbar(img)
    SpecGram_FileName = "SpecGram_H" + str(PatientID)+ '.png'
    plt.savefig(SpecGram_FileName)

    plt.clf()
    f, t, Sxx = signal.spectrogram(ALL2_L/(cnt1*cnt2), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])
    fig.colorbar(img)
    SpecGram_FileName = "SpecGram_L" + str(PatientID)+ '.png'
    plt.savefig(SpecGram_FileName)
    
    plt.clf()
    img = plt.plot((ALL2_H/(cnt3*cnt4) - ALL2_L/(cnt1*cnt2)))
    RawSig_FileName = "DiffSig" + str(PatientID)+ '.png'
    plt.savefig(RawSig_FileName)

def CheckSpecGram_SubPlot (Model, PatientID):

    data  = sio.loadmat('shamdata_tlgo')
    HC    = data['shamhceeg'][:,PatientID] 

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    best_model = load_model(Model)
  
    ALL_EEG = HC
    ALL_Beh = HC_labels
    
    # # # # # Load the EEG data of PD_on # # # # # 
    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  
    
    
    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            
    
    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 
    
    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))
    
    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]
    
    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    
    
    # make predictions
    y_Predicted = best_model.predict(X)
    
    MSE_Test = mean_squared_error(y, y_Predicted)
    print('MSE_Test = %.3f' % (MSE_Test))
    R2_score_Test = r2_score(y, y_Predicted)
    print('R^2score_Test = %.3f' % (R2_score_Test))
    
    
    bin_labels = ['L', 'M', 'H']
    RT = pd.qcut(y[:,0], q=3, labels = bin_labels)
    
    Residue_test = abs(y_Predicted - y)
    RES = Residue_test[:,0]
    bin_labels = ['LE', 'ME', 'HE']
    Err = pd.qcut(RES, q=3, labels = bin_labels)
    
    LL = list(np.where((RT == 'L') & (Err == 'LE')))
    LM = list(np.where((RT == 'L') & (Err == 'ME')))
    LH = list(np.where((RT == 'L') & (Err == 'HE')))
    ML = list(np.where((RT == 'M') & (Err == 'LE')))
    MM = list(np.where((RT == 'M') & (Err == 'ME')))
    MH = list(np.where((RT == 'M') & (Err == 'HE')))
    HL = list(np.where((RT == 'H') & (Err == 'LE')))
    HM = list(np.where((RT == 'H') & (Err == 'ME')))
    HH = list(np.where((RT == 'H') & (Err == 'HE')))
    
    
    ALL1 = 0
    ALL2 = 0
    cnt1 = 0
    cnt2 = 0
    for sample_ind in np.nditer(LL):
        #for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
        # important channels
        for ch in (2,3,7,10, 13,14, 16, 17,21,26):

        #for ch in range (0,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt1 = cnt1 + 1
    ALL2 = ALL2 + ALL1
    cnt2 = cnt2 + 1
    
    ALL2_L = ALL2
  
    
    ALL1 = 0
    ALL2 = 0
    cnt3 = 0
    cnt4 = 0
    for sample_ind in np.nditer(np.concatenate((HL, ML), axis = 1)):
        for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
         #for ch in (2,3,7,10, 13,14, 16, 17,21,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt3 = cnt3 + 1
    ALL2 = ALL2 + ALL1
    cnt4 = cnt4 + 1
    
    ALL2_H = ALL2
    
    fig = plt.figure()    
    
    winSize = 80 #or 200 or 500
    winOverlap = round(winSize*.99) # or round(winSize*.8) or round(winSize*.5)
    f, t, Sxx = signal.spectrogram(ALL2_H/(cnt3*cnt4) - ALL2_L/(cnt1*cnt2), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    plt.subplot(3, 1, 1)
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])
    plt.ylabel('Diff')

    #plt.title("Diff of Spectrograms of Epochs with High/Med vs. Low ReactionTime for New Test Patient = " + str(PatientID + 1))
    fig.colorbar(img)

    plt.subplot(3, 1, 2)
    f, t, Sxx = signal.spectrogram(ALL2_H/(cnt3*cnt4), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])
    plt.ylabel('High')
    fig.colorbar(img)

    plt.subplot(3, 1, 3)
    f, t, Sxx = signal.spectrogram(ALL2_L/(cnt1*cnt2), 1000, signal.windows.hamming(winSize, sym=True), winSize, winOverlap, 1000, 'constant', True,'density', -1, 'psd')
    img = plt.pcolormesh(t, f, Sxx, cmap='RdBu_r')
    plt.ylim([0, 60])
    plt.ylabel('Low')
    fig.colorbar(img)

#    plt.subplot(6, 1, 4)
#    plt.ylabel('Diff')
#    img = plt.plot((ALL2_H/(cnt3*cnt4) - ALL2_L/(cnt1*cnt2)))
#
#    plt.subplot(6, 1, 5)
#    plt.ylabel('High')
#    img = plt.plot((ALL2_H/(cnt3*cnt4)))
#
#    plt.subplot(6, 1, 6)
#    plt.ylabel('Low')
#    img = plt.plot((ALL2_L/(cnt1*cnt2)))
    
    
    FileName = "File" + str(PatientID)+ '.png'
    plt.savefig(FileName)
 
def CheckWelch (Model, PatientID):

    data  = sio.loadmat('shamdata_tlgo')
    HC    = data['shamhceeg'][:,PatientID] 

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    best_model = load_model(Model)
  
    ALL_EEG = HC
    ALL_Beh = HC_labels
    
    # # # # # Load the EEG data of PD_on # # # # # 
    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  
    
    
    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            
    
    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 
    
    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))
    
    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]
    
    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    
    
    # make predictions
    y_Predicted = best_model.predict(X)
    
    MSE_Test = mean_squared_error(y, y_Predicted)
    print('MSE_Test = %.3f' % (MSE_Test))
    R2_score_Test = r2_score(y, y_Predicted)
    print('R^2score_Test = %.3f' % (R2_score_Test))
    
    
    bin_labels = ['L', 'M', 'H']
    RT = pd.qcut(y[:,0], q=3, labels = bin_labels)
    
    Residue_test = abs(y_Predicted - y)
    RES = Residue_test[:,0]
    bin_labels = ['LE', 'ME', 'HE']
    Err = pd.qcut(RES, q=3, labels = bin_labels)
    
    LL = list(np.where((RT == 'L') & (Err == 'LE')))
    LM = list(np.where((RT == 'L') & (Err == 'ME')))
    LH = list(np.where((RT == 'L') & (Err == 'HE')))
    ML = list(np.where((RT == 'M') & (Err == 'LE')))
    MM = list(np.where((RT == 'M') & (Err == 'ME')))
    MH = list(np.where((RT == 'M') & (Err == 'HE')))
    HL = list(np.where((RT == 'H') & (Err == 'LE')))
    HM = list(np.where((RT == 'H') & (Err == 'ME')))
    HH = list(np.where((RT == 'H') & (Err == 'HE')))
    
    
    ALL1 = 0
    ALL2 = 0
    cnt1 = 0
    cnt2 = 0
    for sample_ind in np.nditer(LL):
        #for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
        # important channels
        for ch in (2,3,7,10, 13,14, 16, 17,21,26):

        #for ch in range (0,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt1 = cnt1 + 1
    ALL2 = ALL2 + ALL1
    cnt2 = cnt2 + 1
    
    ALL2_L = ALL2
  
    
    ALL1 = 0
    ALL2 = 0
    cnt3 = 0
    cnt4 = 0
    for sample_ind in np.nditer(np.concatenate((HL, ML), axis = 1)):
        #for ch in (5,6,7,9,10, 12, 13,14, 16, 17,20):
         for ch in (2,3,7,10, 13,14, 16, 17,21,26):

        #for ch in range (0,26):
            
            one_x = X[sample_ind,:,ch]
            ALL1 = ALL1 + one_x
            cnt3 = cnt3 + 1
    ALL2 = ALL2 + ALL1
    cnt4 = cnt4 + 1
    
    ALL2_H = ALL2
    
    fig = plt.figure()
    f, Pxx = signal.welch(abs(ALL2_H/(cnt3*cnt4) - ALL2_L/(cnt1*cnt2)), 1000)
    
    plt.xlim([0, 60])
    plt.ylim([0.005, 0.05])

    plt.semilogy(f, np.sqrt(Pxx))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    Welch_FileName = "Welch_Diff" + str(PatientID)+ '.png'
    plt.savefig(Welch_FileName)



'''    
    plt.clf()
    f = interp2d(t, f, Sxx, kind='linear')
    xnew = np.arange(0, 2.5, .01)
    ynew = np.arange(0, 60,.01)
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    img = plt.pcolormesh(Xn, Yn, data1, cmap='RdBu_r')
    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.title("Diff of Spectrograms of Epochs with High/Med vs. Low ReactionTime for New Test Patient = " + str(PatientID + 1))
    
    fig.colorbar(img)
    SpecGram_FileName = "SpecGram_H" + str(PatientID)+ '.png'
    plt.savefig(SpecGram_FileName)


    f, t, Sxx = signal.spectrogram(ALL2_L/(cnt1*cnt2), fs = 500)
''' 


'''    
    plt.clf()
    f = interp2d(t, f, Sxx, kind='linear')
    xnew = np.arange(0, 2.5, .01)
    ynew = np.arange(0, 60,.01)
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    img = plt.pcolormesh(Xn, Yn, data1, cmap='RdBu_r')
    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.title("Diff of Spectrograms of Epochs with High/Med vs. Low ReactionTime for New Test Patient = " + str(PatientID + 1))
    
    fig.colorbar(img)
    SpecGram_FileName = "SpecGram_L" + str(PatientID)+ '.png'
    plt.savefig(SpecGram_FileName)
 '''
 
def headplot(effects):
    pos = [[173, 52],[211, 48],[250 ,50],[86 ,95],[149, 120],[211, 124],[272, 121],[335, 94],[99 ,153],[322, 152],[58 ,203],[133, 203],[210 ,200],[288 ,202],[100 ,246],[320, 250],[406, 221],[86 ,307],[150 ,280],[210, 280],[270 ,280],[337, 306],[142 ,328],[281, 326],[173, 354],[212, 355],[247, 352]]
    Names = ['FP1', 'FPZ', 'FP2','F7', 'F3', 'Fz', 'F4', 'F8', 'Fc5', 'Fc6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP6', 'A2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Po5', 'Po6', 'O1', 'Oz', 'O2']
    mypos = np.array(pos, ndmin=2)
    mv.plot_topomap(effects,mypos, cmap ='RdBu_r', names = Names, show_names = True)
    return (mv)
def phaseScramble_Shuffle(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave the real and imaginary components. 
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr
def phaseScramble_AddNoise(ts, amount):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    phase_fsr = phase_fsr + np.random.normal(0,np.pi*1/amount)
    # use broadcasting and ravel to interleave the real and imaginary components. 
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr

def Test_Phase_Perturb (Model, subband, PatientID, amount):
    
    best_model = load_model(Model)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            

    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_test_Predicted = best_model.predict(X)

    effects_ypred = []
    #effects_yorig = []
    

    for ch in range(27):  # iterate over 27 channels
        for ep in range(X.shape[0]):
            new_x = X.copy()
            new_x[ep, :, ch] = phaseScramble_AddNoise(new_x[ep, :, ch], amount)  
            
        perturbed_out = best_model.predict(new_x)
        perturbed_out = np.nan_to_num(perturbed_out)
        y_test_Predicted = np.nan_to_num(y_test_Predicted)
        
        effect_ypred = ((y_test_Predicted - perturbed_out) ** 2).mean() ** 0.5    
        effects_ypred.append(effect_ypred)
        #effects_yorig.append(effect_yorig)
    return(effects_ypred)    
 
    '''   
 
        effect_yorig = ((y - perturbed_out) ** 2).mean() ** 0.5
    
        effects_ypred.append(effect_ypred)
        effects_yorig.append(effect_yorig)
    
    return(effects_ypred,effects_yorig)    
   '''
        
def Test_Elect_Perturb (Model, subband, PatientID, amount):
    
    best_model = load_model(Model)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            

    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_test_Predicted = best_model.predict(X)

    effects_ypred = []
    #effects_yorig = []
    

    for ch in range(27):  # iterate over 27 channels
        new_x = X.copy()
        perturbation = np.random.normal(0.0, amount, size=new_x.shape[:2])
        new_x[:, :, ch] = new_x[:, :, ch] + perturbation
            
        perturbed_out = best_model.predict(new_x)
        
        diff = y_test_Predicted - perturbed_out
        diff1 = diff[~np.isnan(diff)]
        effect_ypred = (diff1 ** 2).mean() ** 0.5
        
        #effect_yorig = ~np.isnan(y - perturbed_out)) ** 2).mean() ** 0.5
    
        effects_ypred.append(effect_ypred)
        #effects_yorig.append(effect_yorig)
    return(effects_ypred)    
 
def Test_Amp_Perturb (Model, subband, PatientID, amount):
    
    best_model = load_model(Model)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            

    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_test_Predicted = best_model.predict(X)

    effects_ypred = []
    #effects_yorig = []
    

    for ch in range(27):  # iterate over 27 channels
        new_x = X.copy()
        new_x[:, :, ch] = new_x[:, :, ch] * amount
            
        perturbed_out = best_model.predict(new_x)
        
        diff = y_test_Predicted - perturbed_out
        diff1 = diff[~np.isnan(diff)]
        effect_ypred = (diff1 ** 2).mean() ** 0.5
        
        #effect_yorig = ~np.isnan(y - perturbed_out)) ** 2).mean() ** 0.5
    
        effects_ypred.append(effect_ypred)
        #effects_yorig.append(effect_yorig)
    return(effects_ypred) 
  
    
 
def Test_Temporal3Parts (Model, subband, PatientID, part):
    
    best_model = load_model(Model)
    

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  
    X2_ALL_EEG =  np.zeros((p* m,n,d)) 

    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    movement_time_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
    reaction_time_ALL = np.zeros((ALL_EEG.size, num_trials,1))                

    y_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
        
        
    for tr in range (num_trials):
        i = 6 #movement time
        movement_time_ALL[0,tr] = ALLBeh[tr,i]
        
        i = 12
        reaction_time_ALL[0,tr] = ALLBeh[tr,i]

    rt_df = pd.DataFrame(reaction_time_ALL[0])
    rt_df.fillna((rt_df.mean()), inplace=True) 
    rt_df.to_numpy()
    mt_df = pd.DataFrame(movement_time_ALL[0])
    mt_df.fillna((mt_df.mean()), inplace=True)
    mt_df.to_numpy()

    for  tr in range (num_trials):
        if part == 1:
            t1 = 500
            t2 = int(500 + mt_df[0][tr])
            temp_con = np.concatenate((np.zeros(t1), np.ones(t2-t1), np.zeros(2000-t2)), axis=0)
        elif part == 2:
            t2 = int(500 + mt_df[0][tr])
            t4 = int(500 + mt_df[0][tr] + rt_df[0][tr])
            temp_con = np.concatenate((np.zeros(t2), np.ones(t4-t2), np.zeros(2000-t4)), axis=0)
        else:
             t4 = int(500 + mt_df[0][tr] + rt_df[0][tr])
             temp_con = np.concatenate((np.zeros(t4), np.ones(2000 - t4)), axis=0)
   
        repetitions = 27
        repeats_array = np.transpose(np.tile(temp_con, (repetitions,1)))
        temp = X_ALL_EEG[tr,:,:]
        X2_ALL_EEG[tr,:,:] = temp * repeats_array

    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X2_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X2_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_Predicted = best_model.predict(X)

    MSE_Test = mean_squared_error(y, y_Predicted)
    R2_score_Test = r2_score(y, y_Predicted)
    #return (y-y_Predicted)    
    return(MSE_Test)
    
def compile_saliency_function(model):
  """
  Compiles a function to compute the saliency maps and predicted classes
  for a given minibatch of input images.
  """
  inp = model.layers[0].input
  outp = model.layers[-1].output
  max_outp = K.max(outp, axis=1)
  saliency = K.gradients(max_outp[0], inp)[0]
  #saliency = K.tf.GradientTape(max_outp[0], inp)[0]
  max_class = K.argmax(outp, axis=1)
  return K.function([inp], [saliency])
    
def Test_Saliency1 (FileName_BestModel, subband, pt):
 best_model = load_model(FileName_BestModel)
 if subband == 'full':
    data  = sio.loadmat('shamdata_tlgo')
    HC    = data['shamhceeg'][:,pt] 
 else:
    data  = sio.loadmat('DatahcAlireza')
    HC     = data[subband][:,pt] # Healthy Control

  # Load the EEG data labels
 labels  = sio.loadmat('task_gvssham')
 HC_labels     = labels['hcoffmed'][:,pt]
    
    
 ALL_EEG = HC
 ALL_Beh = HC_labels
    
 num_chans,time_steps,num_trials = ALL_EEG[0].shape
 X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
 x = ALL_EEG[0]
 X_ALL_EEG[:,:,:] = x
 # Data transpose     
 X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
 # Data dimensions          
 p,m,n,d = X_ALL_EEG.shape
 # Data reshape     
 X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  

 # # # # # Load the EEG data labels # # # # # 
 num_trials, num_behvars = ALL_Beh[0].shape
 movement_time_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
 reaction_time_ALL = np.zeros((ALL_EEG.size, num_trials,1))                

 y_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
 ALLBeh = ALL_Beh[0]
 for tr in range (num_trials):
    i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
    y_ALL[0,tr] = ALLBeh[tr,i]
    
        
 for tr in range (num_trials):
    i = 6 #movement time
    movement_time_ALL[0,tr] = ALLBeh[tr,i]
        
    i = 12
    reaction_time_ALL[0,tr] = ALLBeh[tr,i]
    

 rt_df = pd.DataFrame(reaction_time_ALL[0])
 rt_df.fillna((rt_df.mean()), inplace=True) 
    
 #find min and max RT among 10 trials and check their Saliency
 max_RT_ind = rt_df.idxmax()
 max_RT     = rt_df.max()
 min_RT_ind = rt_df.idxmin()
 min_RT     = rt_df.min()
    
 rt_df.to_numpy()
 mt_df = pd.DataFrame(movement_time_ALL[0])
 mt_df.fillna((mt_df.mean()), inplace=True)
 mt_df.to_numpy()
 
 cnt = 1
 for  tr in (max_RT_ind, min_RT_ind):
 
    tpre1 = 500/2
    tpre2 = int(500 + mt_df[0][tr])/2
    texc2 = int(500 + mt_df[0][tr])/2
    texc4 = int(500 + mt_df[0][tr] + rt_df[0][tr])/2
    tpos4 = int(500 + mt_df[0][tr] + rt_df[0][tr])/2
    
    inp = X_ALL_EEG[tr,0::2,:]
    sal = compile_saliency_function(best_model)(inp)
    sal_gray = np.maximum(sal[0], 0)
    sal_gray /= np.max(sal_gray)
    sal_gray2 = np.transpose(np.reshape(sal_gray, [1000 , 27]))
    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    ax = sns.heatmap(sal_gray2)
    plt.axvline(x = tpre1, color='y', linestyle='--')
    plt.axvline(x = tpre2, color='y', linestyle='--')
    
    plt.axvline(x = texc2, color='b', linestyle='--')
    plt.axvline(x = texc4, color='b', linestyle='--')
    
    plt.axvline(x = tpos4, color='r', linestyle='--')
    

    PlotFileName = what + ".PNG"
    plt.savefig(PlotFileName)
    
    #------------------- SUM over time OR channels ------------------
    
    res1000 = np.sum(sal_gray2,0)
    res27 = np.sum(sal_gray2,1)

    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    res1000_smooth = savgol_filter(res1000, window_length=101, polyorder=5)
    ax = plt.plot(res1000_smooth)
    plt.axvline(x = tpre1, color='y', linestyle='--')
    plt.axvline(x = tpre2, color='y', linestyle='--')
    
    plt.axvline(x = texc2, color='b', linestyle='--')
    plt.axvline(x = texc4, color='b', linestyle='--')
    
    plt.axvline(x = tpos4, color='r', linestyle='--')
    
    PlotFileName = what + "_sum1000.PNG"
    plt.savefig(PlotFileName)

    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    
    plt.clf()
    img = headplot(res27)
    PlotFileName = what + "_sum27.PNG"
    plt.savefig(PlotFileName)
    
    cnt = cnt + 1
def Test_Saliency2 (FileName_BestModel, subband, pt):
 best_model = load_model(FileName_BestModel)
 if subband == 'full':
    data  = sio.loadmat('shamdata_tlgo')
    HC    = data['shamhceeg'][:,pt] 
 else:
    data  = sio.loadmat('DatahcAlireza')
    HC     = data[subband][:,pt] # Healthy Control

  # Load the EEG data labels
 labels  = sio.loadmat('task_gvssham')
 HC_labels     = labels['hcoffmed'][:,pt]
    
    
 ALL_EEG = HC
 ALL_Beh = HC_labels
    
 num_chans,time_steps,num_trials = ALL_EEG[0].shape
 X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
 x = ALL_EEG[0]
 X_ALL_EEG[:,:,:] = x
 # Data transpose     
 X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
 # Data dimensions          
 p,m,n,d = X_ALL_EEG.shape
 # Data reshape     
 X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  

 # # # # # Load the EEG data labels # # # # # 
 num_trials, num_behvars = ALL_Beh[0].shape
 movement_time_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
 reaction_time_ALL = np.zeros((ALL_EEG.size, num_trials,1))                

 y_ALL = np.zeros((ALL_EEG.size, num_trials,1)) 
 ALLBeh = ALL_Beh[0]
 for tr in range (num_trials):
    i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
    y_ALL[0,tr] = ALLBeh[tr,i]
    
        
 for tr in range (num_trials):
    i = 6 #movement time
    movement_time_ALL[0,tr] = ALLBeh[tr,i]
        
    i = 12
    reaction_time_ALL[0,tr] = ALLBeh[tr,i]
    

 rt_df = pd.DataFrame(reaction_time_ALL[0])
 rt_df.fillna((rt_df.mean()), inplace=True) 
    
 #find min and max RT among 10 trials and check their Saliency
 max_RT_ind = rt_df.idxmax()
 max_RT     = rt_df.max()
 min_RT_ind = rt_df.idxmin()
 min_RT     = rt_df.min()
    
 rt_df.to_numpy()
 mt_df = pd.DataFrame(movement_time_ALL[0])
 mt_df.fillna((mt_df.mean()), inplace=True)
 mt_df.to_numpy()
 
 cnt = 1
 for  tr in (max_RT_ind, min_RT_ind):
 
    tpre1 = 500/2
    tpre2 = int(500 + mt_df[0][tr])/2
    texc2 = int(500 + mt_df[0][tr])/2
    texc4 = int(500 + mt_df[0][tr] + rt_df[0][tr])/2
    tpos4 = int(500 + mt_df[0][tr] + rt_df[0][tr])/2
    
    inp = X_ALL_EEG[tr,0::2,:]
    sal = compile_saliency_function(best_model)(inp)
    sal_gray = np.maximum(sal[0], 0)
    sal_gray /= np.max(sal_gray)
    sal_gray2 = np.transpose(np.reshape(sal_gray, [1000 , 27]))
    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    ax = sns.heatmap(sal_gray2)
    plt.axvline(x = tpre1, color='y', linestyle='--')
    plt.axvline(x = tpre2, color='y', linestyle='--')
    
    plt.axvline(x = texc2, color='b', linestyle='--')
    plt.axvline(x = texc4, color='b', linestyle='--')
    
    plt.axvline(x = tpos4, color='r', linestyle='--')
    

    PlotFileName = what + ".PNG"
    plt.savefig(PlotFileName)
    
    #------------------- SUM over time OR channels ------------------
    
    res1000 = np.sum(sal_gray2,0)
    res27 = np.sum(sal_gray2,1)

    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    res1000_smooth = savgol_filter(res1000, window_length=101, polyorder=5)
    ax = plt.plot(res1000_smooth)
    plt.axvline(x = tpre1, color='y', linestyle='--')
    plt.axvline(x = tpre2, color='y', linestyle='--')
    
    plt.axvline(x = texc2, color='b', linestyle='--')
    plt.axvline(x = texc4, color='b', linestyle='--')
    
    plt.axvline(x = tpos4, color='r', linestyle='--')
    
    PlotFileName = what + "_sum1000.PNG"
    plt.savefig(PlotFileName)

    if (cnt == 1):
        what = "Longest RT" + str(int(max_RT[0])) + "-subject " + str(pt+1)
    else:
        what = "Shortest RT" + str(int(min_RT[0])) + "-subject" + str(pt+1)
    plt.figure(), plt.title(what)
    
    plt.clf()
    img = headplot(res27)
    PlotFileName = what + "_sum27.PNG"
    plt.savefig(PlotFileName)
    
    cnt = cnt + 1
    
    #--------------------------------------------
def NewTest_HistMartin (Model, subband, PatientID):
    
    ExcCol = subband2Index(subband)
    best_model = load_model(Model)

    if subband == 'full':
       data  = sio.loadmat('shamdata_tlgo')
       HC    = data['shamhceeg'][:,PatientID] 
    else:
       data  = sio.loadmat('DatahcAlireza')
       HC     = data[subband][:,PatientID] # Healthy Control

    # Load the EEG data labels
    labels  = sio.loadmat('task_gvssham')
    HC_labels     = labels['hcoffmed'][:,PatientID]
    
    
    ALL_EEG = HC
    ALL_Beh = HC_labels


    num_chans,time_steps,num_trials = ALL_EEG[0].shape
    X_ALL_EEG = np.zeros((ALL_EEG.size,num_chans,time_steps,num_trials))
    
    
    x = ALL_EEG[0]
    X_ALL_EEG[:,:,:] = x
    # Data transpose     
    X_ALL_EEG = np.transpose(X_ALL_EEG, (0, 3, 2, 1))        
    # Data dimensions          
    p,m,n,d = X_ALL_EEG.shape
    # Data reshape     
    X_ALL_EEG =  X_ALL_EEG.reshape(p* m,n,d)  


    # # # # # Load the EEG data labels # # # # # 
    num_trials, num_behvars = ALL_Beh[0].shape
    y_ALL = np.zeros((ALL_EEG.size, num_trials,1))            

    ALLBeh = ALL_Beh[0]
    for tr in range (num_trials):
        i = 12 # index of Reaction Time (Change the index to look other Behavior measures)
        y_ALL[0,tr] = ALLBeh[tr,i]
    
    y_ALL = np.reshape(y_ALL, (ALL_EEG.size*num_trials, 1)) 


    # # # # # Data augmentation (Down-sampling by 2) # # # # # 
    X_ALL_EEG_DS1 = X_ALL_EEG[:,0::2,:]
    X_ALL_EEG_DS2 = X_ALL_EEG[:,1::2,:]
    X_ALL_EEG_DS  = np.concatenate((X_ALL_EEG_DS1,X_ALL_EEG_DS2))
    # labels
    y_ALL_DS = np.concatenate((y_ALL,y_ALL))

    # # # # # no Shuffle the data # # # # # 
    N = X_ALL_EEG_DS.shape[0]
    indices = [i for i in range(N)]
    #shuffle(indices)
    X_ALL_EEG_DS = X_ALL_EEG_DS[indices, :,:]
    y_ALL_DS     = y_ALL_DS[indices,]

    # Sanity Check: Exclude labels with NaN values
    # Indices of nan and inf values
    idx = np.where((np.isnan(y_ALL_DS)==False) & (np.isinf(y_ALL_DS)==False))
    filtered_X_ALL_EEG_DS = X_ALL_EEG_DS[idx[0],:,:]
    filtered_y_ALL_DS = y_ALL_DS[idx[0]]
    
    
    X = filtered_X_ALL_EEG_DS
    y = filtered_y_ALL_DS
    
    # Make X and y as float 32
    X = X.astype('float32')
    y = y.astype('float32')
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Standarizing X
    m, n, d = X.shape
    X = X.reshape(m, n*d)
    X = scaler.fit_transform(X)
    X = X.reshape(m, n, d)
    # Standarizing y
    y = scaler.fit_transform(y)
    

    # make predictions
    y_Predicted = best_model.predict(X)

    
    MSE_Test = mean_squared_error(y, y_Predicted)
    R2_score_Test = r2_score(y, y_Predicted)
    
#    srcfile = openpyxl.load_workbook('Results.xlsm',read_only=False, keep_vba= True)
#
#    sheetname = srcfile.get_sheet_by_name('NewTest_MSE')
#    sheetname.cell(row=PatientID+2,column=ExcCol).value = str(round(MSE_Test,2)) 
#    sheetname = srcfile.get_sheet_by_name('NewTest_R2')
#    sheetname.cell(row=PatientID+2,column=ExcCol).value = str(round(R2_score_Test,2)) 
#
#    srcfile.save('Results.xlsm')
    return (y, y_Predicted)