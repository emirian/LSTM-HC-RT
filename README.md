# LSTM-HC-RT
# The files for the paper about "How does end to end learning for the prediction of reaction time Help to understand the healthy brain?"
1) Build filtered data using MainScript.m which can be found here: https://drive.google.com/drive/folders/1l_w32Trp7dX1NFbrZf-qRCaU6nQNatxL. It will need the data found here: https://drive.google.com/drive/folders/1IOLGO1HVodH91q_kss8i5vNQiao_Z_iK
This makes DatahcAlireza.mat which is used by the rest of the codes.
2) Build the models by running codes below. You would need MyUtils.py 
1-Subbands_AllSubjects.py
2-MoreTests_AllSubjects.py
2-MoreTests_AllSubjects_TbyT.py
25-MoreTests_AllSubjects_TbyT.py
3) Run the statistical analysis on the result excel files: Statistical Tests and Visualization_AK.R
4) Also run Phase_Amplitude_High_Low_RT.m, d_PAC_Cohen_Granger.m, c_PAC_Cohen_MainBody.m
To summarize features and calculate PAC + g-cause with RT
5) Run DCP_EEG_PhAm13.R to see the ADC results
6) Run 6-BrainStates_ShortLong.py to see the importance of each brain state (standby, preparatory, execution) in RT prediction
