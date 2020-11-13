clear all
clc

load('shamdata_tlgo.mat');
load ('task_gvssham.mat');

Xs_L = [];
Xs_H = [];

All_X = [];
All_Y = [];

%beh_ind = 5; %Vigour
beh_ind = 13; %RT

Alow = 8; Ahigh  = 13;
Blow = 13; Bhigh = 32;
Dlow = 0.5; Dhigh = 4;
Tlow = 4; Thigh = 8;
Glow = 32; Ghigh = 100;

G2low = 32; G2high = 60;
DTlow = 3; DThigh =7; 

LRTss = [];
HRTss = [];

LPACss = [];
HPACss = [];

LResults_epochs_DTG2 = [];
HResults_epochs_DTG2 = [];

for s = [1 : 7, 10:22]
   
   beh = cell2mat(hcoffmed(s));
   rt = beh(:, beh_ind);
   [Y,E] = discretize(rt,3);
   
   % find epochs with High vs. Low RT
   inds_H = find (Y == 3);
   inds_L = find (Y == 1);
   
   %LResults_epochs_DTG2 = [];
   
   for e = 1 : size(inds_L, 1) 
      LPACs = [];
      
      epoch = inds_L(e);
      
      %--------------------------------------------
      sig_27200010 = shamhceeg{s};
     
      for ch = 1 : 27
         if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
              sig_2000 = sig_27200010(ch,:,epoch);
              hasAnyNaN = sum(isnan(sig_2000(:)))
              %[pac, dpac] = PAC_Cohen_func(sig_2000, DTlow,DThigh,G2low,G2high);
              [pac, dpac] = PAC_Cohen_func(sig_2000, Alow,Ahigh,G2low,G2high);
              LPACs  = cat(1,LPACs,dpac);
         end
      end
      LResults_overchannel = mean(LPACs,1);
      LResults_epochs_DTG2 = cat(1,LResults_epochs_DTG2,LResults_overchannel);
      %-------------------------------------------
      one_rt = rt(epoch);
      LRTss   = cat(1, LRTss, one_rt);
   end
  
  %-------------------------H i g h ------------------------------
  %--------------------------------------------------------------
  %--------------------------------------------------------------
   
   %HResults_epochs_DTG2 = [];
   
   for e = 1 : size(inds_H, 1) 
      HPACs = [];
     
      epoch = inds_H(e);
      
      %--------------------------------------------
      sig_27200010 = shamhceeg{s};
      
      for ch = 1 : 27
         if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
              sig_2000 = sig_27200010(ch,:,epoch);
              hasAnyNaN = sum(isnan(sig_2000(:)))
              %[pac, dpac] = PAC_Cohen_func(sig_2000, DTlow,DThigh,G2low,G2high);
              [pac, dpac] = PAC_Cohen_func(sig_2000, Alow,Ahigh,G2low,G2high);
              HPACs  = cat(1,HPACs,dpac);
         end
      end
      HResults_overchannel = mean(HPACs,1);
      HResults_epochs_DTG2 = cat(1,HResults_epochs_DTG2,HResults_overchannel);
      
      %------------------------------------------
      %HPACss = cat(1, HPACss, HResults_epochs_DTG2);
      one_rt = rt(epoch);
      HRTss  = cat(1, HRTss, one_rt);
   end
   
 
end % for each subject

%HResults_epochs_DTG2(13) = [];
%HRTss(13) = [];
 
figure, yyaxis left; plot (LResults_epochs_DTG2); yyaxis right; plot(LRTss);legend ("PAC", "RT"); title ("Low RT: Fast Action");
figure, yyaxis left; plot (HResults_epochs_DTG2); yyaxis right; plot(HRTss);legend ("PAC", "RT"); title ("High RT: Slow Action");

corr(LResults_epochs_DTG2, LRTss)
corr(HResults_epochs_DTG2, HRTss)

