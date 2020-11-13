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

RTss = [];

PACss = zeros(22, 10);
RTss = zeros(22, 10);



for s = [1 : 7, 10:22]
   %disp ("subject:"); disp(num2str(s))
   beh = cell2mat(hcoffmed(s));
   rt = beh(:, beh_ind);
   RT(s, :) = rt; 
   Results_epochs_DTG2 = [];
   for e = 1 : 10 
      %disp ("epoch:"); disp(num2str(e))
      PACs = [];
      
      %--------------------------------------------
      sig_27200010 = shamhceeg{s};
     
      for ch = 1 : 27
         if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
              sig_2000 = sig_27200010(ch,:,e);
              %hasAnyNaN = sum(isnan(sig_2000(:)))
              [pac, dpac] = PAC_Cohen_func(sig_2000, DTlow,DThigh,G2low,G2high);
              PACs  = cat(1,PACs,pac);
         end
      end
      Results_overchannel = mean(PACs,1);
      Results_epochs_DTG2 = cat(1,Results_epochs_DTG2,Results_overchannel);
      
   end
   PACss(s,:) = Results_epochs_DTG2;  
   RTss(s,:) = rt;
 
end % for each subject

RTss(8:9,:) = [];
PACss(8:9,:) = [];

RTss2 = knnimpute(RTss);
PACss2 = knnimpute(PACss);

fs = zeros(20,1);
cvs = zeros (20,1);

for s = 1:20
    r = RTss2(s,:);
    p = PACss2(s,:);
    figure(s); yyaxis left; plot (r); yyaxis right; plot(p);legend ("RT", "PAC"); title (num2str(s));
    [f,cv] = granger_cause(r, p, 0.1, 2);
    fs(s) = f;
    cvs(s) = cv;
end
find(fs > cvs)