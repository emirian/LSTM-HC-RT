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


for s = 1 : 22
    if (s ~= 8 && s ~= 9 && s~=10 && s~=14)
   
       beh = cell2mat(hcoffmed(s));
       rt = beh(:, beh_ind);
       rt2 = fillmissing(rt,'nearest');
       [Y,E] = discretize(rt2,3);
       % find epochs with High vs. Low RT
       inds_H = find (Y == 3);
       inds_L = find (Y == 1);

       LResults_del_epochs = [];
       LResults_gam_epochs = [];
       LResults_alp_epochs = [];
       LResults_bet_epochs = [];
       LResults_tet_epochs = [];

       for e = 1 : size(inds_L, 1) 
          LResults_del = [];
          LResults_gam = [];
          LResults_alp = [];
          LResults_bet = [];
          LResults_tet = [];

          epoch = inds_L(e);

          %--------------------------------------------
          del_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  del_1000 = del_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(del_1000, 2);
                  LResults_del = cat(1,LResults_del,result2000);
             end
          end
          LResults_del_overchannel = mean(LResults_del,1);
          LResults_del_epochs = cat(1,LResults_del_epochs,LResults_del_overchannel);
          %-------------------------------------------
          gam_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  gam_1000 = gam_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(gam_1000, 66);
                  LResults_gam = cat(1,LResults_gam,result2000);
             end
          end
          LResults_gam_overchannel = mean(LResults_gam,1);
          LResults_gam_epochs = cat(1,LResults_gam_epochs,LResults_gam_overchannel);
          %---------------------------------------------
          alp_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  alp_1000 = alp_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(alp_1000, 11);
                  LResults_alp = cat(1,LResults_alp,result2000);
             end
          end
          LResults_alp_overchannel = mean(LResults_alp,1);
          LResults_alp_epochs = cat(1,LResults_alp_epochs,LResults_alp_overchannel);
          %---------------------------------------------
          bet_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  bet_1000 = bet_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(bet_1000, 20);
                  LResults_bet = cat(1,LResults_bet,result2000);
             end
          end
          LResults_bet_overchannel = mean(LResults_bet,1);
          LResults_bet_epochs = cat(1,LResults_bet_epochs,LResults_bet_overchannel);
         %----------------------------------------------
          tet_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  tet_1000 = tet_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(tet_1000, 6);
                  LResults_tet = cat(1,LResults_tet,result2000);
             end
          end
          LResults_tet_overchannel = mean(LResults_tet,1);
          LResults_tet_epochs = cat(1,LResults_tet_epochs,LResults_tet_overchannel);


       end

       %----------- Sum over Epochs of Low RT of one subject ------------
       as_del = mean(LResults_del_epochs,1);
       LAmp_del = mean(abs(as_del));
       LAng_del = abs(mean(exp(1i*angle (as_del))));

       as_gam = mean(LResults_gam_epochs,1);
       LAmp_gam = mean(abs(as_gam));
       LAng_gam = abs(mean(exp(1i*angle (as_gam))));

       as_alp = mean(LResults_alp_epochs,1);
       LAmp_alp = mean(abs(as_alp));
       LAng_alp = abs(mean(exp(1i*angle (as_alp))));

       as_bet = mean(LResults_bet_epochs,1);
       LAmp_bet = mean(abs(as_bet));
       LAng_bet = abs(mean(exp(1i*angle (as_bet))));

       as_tet = mean(LResults_tet_epochs,1);
       LAmp_tet = mean(abs(as_tet));
       LAng_tet = abs(mean(exp(1i*angle (as_tet))));


       % Concat all for different subjects
       %temp = [LAmp_del LAng_del LAmp_gam LAng_gam LAmp_alp LAng_alp LAmp_bet LAng_bet LAmp_tet LAng_tet];
       temp = [LAmp_del LAng_del LAmp_tet LAng_tet LAmp_alp LAng_alp LAmp_bet LAng_bet LAmp_gam LAng_gam];
       Xs_L = cat(1,Xs_L,temp);


      %-------------------------H i g h ------------------------------
      %--------------------------------------------------------------
      %--------------------------------------------------------------

       HResults_del_epochs = [];
       HResults_gam_epochs = [];
       HResults_alp_epochs = [];
       HResults_bet_epochs = [];
       HResults_tet_epochs = [];

       for e = 1 : size(inds_H, 1) 
          HResults_del = [];
          HResults_gam = [];
          HResults_alp = [];
          HResults_bet = [];
          HResults_tet = [];

          epoch = inds_H(e);

          %--------------------------------------------
          del_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  del_1000 = del_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(del_1000, 2);
                  HResults_del = cat(1,HResults_del,result2000);
             end
          end
          HResults_del_overchannel = mean(HResults_del,1);
          HResults_del_epochs = cat(1,HResults_del_epochs,HResults_del_overchannel);
          %-------------------------------------------
          gam_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  gam_1000 = gam_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(gam_1000, 66);
                  HResults_gam = cat(1,HResults_gam,result2000);
             end
          end
          HResults_gam_overchannel = mean(HResults_gam,1);
          HResults_gam_epochs = cat(1,HResults_gam_epochs,HResults_gam_overchannel);
          %---------------------------------------------
          alp_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  alp_1000 = alp_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(alp_1000, 11);
                  HResults_alp = cat(1,HResults_alp,result2000);
             end
          end
          HResults_alp_overchannel = mean(HResults_alp,1);
          HResults_alp_epochs = cat(1,HResults_alp_epochs,HResults_alp_overchannel);
          %---------------------------------------------
          bet_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  bet_1000 = bet_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(bet_1000, 20);
                  HResults_bet = cat(1,HResults_bet,result2000);
             end
          end
          HResults_bet_overchannel = mean(HResults_bet,1);
          HResults_bet_epochs = cat(1,HResults_bet_epochs,HResults_bet_overchannel);
         %----------------------------------------------
          tet_27100010 = shamhceeg{s};
          for ch = 1 : 27
             if (ch == 5 || ch == 6 || ch == 7 || ch == 9 || ch == 10 || ch == 12 || ch == 13 || ch == 14 || ch == 16 || ch == 17 || ch == 20)
                  tet_1000 = tet_27100010(ch,:,epoch);
                  result2000 = Mean_Phase_Amp_Extraction_func2(tet_1000, 6);
                  HResults_tet = cat(1,HResults_tet,result2000);
             end
          end
          HResults_tet_overchannel = mean(HResults_tet,1);
          HResults_tet_epochs = cat(1,HResults_tet_epochs,HResults_tet_overchannel);


       end

       %----------- Sum over Epochs of High RT of one subject ------------
       as_del = mean(HResults_del_epochs,1);
       HAmp_del = mean(abs(as_del));
       HAng_del = abs(mean(exp(1i*angle (as_del))));

       as_gam = mean(HResults_gam_epochs,1);
       HAmp_gam = mean(abs(as_gam));
       HAng_gam = abs(mean(exp(1i*angle (as_gam))));

       as_alp = mean(HResults_alp_epochs,1);
       HAmp_alp = mean(abs(as_alp));
       HAng_alp = abs(mean(exp(1i*angle (as_alp))));

       as_bet = mean(HResults_bet_epochs,1);
       HAmp_bet = mean(abs(as_bet));
       HAng_bet = abs(mean(exp(1i*angle (as_bet))));

       as_tet = mean(HResults_tet_epochs,1);
       HAmp_tet = mean(abs(as_tet));
       HAng_tet = abs(mean(exp(1i*angle (as_tet))));


       % Concat all for different subjects
       %temp = [HAmp_del HAng_del HAmp_gam HAng_gam HAmp_alp HAng_alp HAmp_bet HAng_bet HAmp_tet HAng_tet];
       temp = [HAmp_del HAng_del HAmp_tet HAng_tet HAmp_alp HAng_alp HAmp_bet HAng_bet HAmp_gam HAng_gam];
       Xs_H = cat(1,Xs_H,temp);



       %----------------------------- ALL TOGETHER ------------------

       All_X = [Xs_L; Xs_H];   
    end
end % for each subject


 Lows = zeros(18, 1);
 Highs = ones (18, 1);
 all_Y = [Lows; Highs];
 

FeatureNames = ["del-A", "del-P", "tet-A" ,"tet-P" ,"alp-A", "alp-P", "bet-A" ,"bet-P", "gam-A", "gam-P"];
for i = 1 : 10
    figure(i), 
    for j = 1 : 10
       
            subplot(2, 5, j), gscatter(All_X(:,i), All_X(:,j), all_Y), xlabel (FeatureNames(i)), ylabel (FeatureNames(j));
       
    end
end

Z13 = [All_X all_Y];
save('Z_13.mat', 'Z13');


figure, gplotmatrix(All_X,All_X,all_Y,'br','..',[],'on',[],FeatureNames,FeatureNames)

%--- TTEST added Sep 22, 2020
[h, p] = ttest2(Xs_H(:,1:10), Xs_L(:,1:10))