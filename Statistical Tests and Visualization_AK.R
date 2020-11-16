library(pacman)
p_load(reshape2,
       ez,
       lme4,
       lmerTest,
       ggplot2,
       grid,
       tidyr,
       plyr,
       dplyr,
       effects,
       gridExtra,
       # psych,
       Cairo, #alternate image writing package with superior performance.
       corrplot,
       knitr,
       PerformanceAnalytics,
       afex,
       ggpubr,
       readxl,
       export)

########################### Load Data 2 RT April 15th All ##########################
BadIDs = c("8","9")
TrainOnFullFileName = "TrainOnFull2_Results4AK_RT.xlsx"
TrainOnSBFileName = "TrainOnSB2_Results4AK_RT.xlsx"
dat = NULL
sheetNames = excel_sheets(TrainOnFullFileName)
for(ShName in sheetNames){
        temp      = read_xlsx(TrainOnFullFileName, sheet = ShName)
        temp = melt(temp, id.vars = c("Trial"), variable.name = "ID",value.name = "DeltaRT")
        temp$Train = unique("Full")
        temp$Test = unique(ShName)
        dat = rbind(dat,temp)
}

sheetNames = excel_sheets(TrainOnSBFileName)
for(ShName in sheetNames){
        temp      = read_xlsx(TrainOnSBFileName, sheet = ShName)
        temp = melt(temp, id.vars = c("Trial"), variable.name = "ID",value.name = "DeltaRT")
        temp$Train = unique("Subband")
        temp$Test = unique(ShName)
        dat = rbind(dat,temp)
}

dat$ID = factor(dat$ID)
dat = dat[!(dat$ID %in% BadIDs),]
dat = dat[complete.cases(dat),]

names(dat)
trialLevelDat = dat[,c("ID","Trial","Train","Test","DeltaRT")]
trialLevelDat$Trial = as.factor(trialLevelDat$Trial)
trialLevelDat$Train = as.factor(trialLevelDat$Train)
trialLevelDat$Test = as.factor(trialLevelDat$Test)
write.csv(trialLevelDat,"AK_TriallevelData_RT.csv",row.names = FALSE)

########################################### ANova Test ################################
Dat = trialLevelDat
names(Dat) = c("ID","Trial","Model","Test","DeltaRT")
Dat$Trial = gsub("Trial ", "", Dat$Trial)

Dat$Test = as.character(Dat$Test)
Dat$Test = gsub("_Filtered", "_Phase_perturbed", Dat$Test, ignore.case = FALSE, perl = FALSE,
                    fixed = FALSE, useBytes = FALSE)
# Dat$Test = as.factor(Dat$Test)
Dat$Phase = case_when(grepl("Phase",Dat$Test, ignore.case = TRUE) ~ "Perturbed",
                     TRUE ~ "Original") 


Dat$Test = case_when(grepl("Alpha",Dat$Test, ignore.case = TRUE) ~ "Alpha",
                    grepl("Beta",Dat$Test, ignore.case = TRUE) ~ "Beta",
                    grepl("Delta",Dat$Test, ignore.case = TRUE) ~ "Delta",
                    grepl("Theta",Dat$Test, ignore.case = TRUE) ~ "Theta",
                    grepl("Gamma",Dat$Test, ignore.case = TRUE) ~ "Gamma",
                    grepl("Full",Dat$Test, ignore.case = TRUE) ~ "Full")

Dat$Test = as.factor(Dat$Test)
Dat$Phase = as.factor(Dat$Phase)
Dat$Model = as.factor(Dat$Model)

DataAvg = as.data.frame(summarise(group_by(Dat,ID,Model,Test,Phase),MeanRTDiff = mean(DeltaRT^2,na.rm=T)))

ggplot(DataAvg,aes(x=Model, y=MeanRTDiff, fill = Phase)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))
        # stat_compare_means( comparisons = list(c("Full","Subband")),
        #                     label = "p.signif",label.y = .5)

# -----------------------> Between Models
DatAnova = DataAvg[! DataAvg$Test %in% c("Full","Theta","Alpha","Beta"),]

ggplot(DatAnova,aes(x=Model, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Phase)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))

        # stat_compare_means( comparisons = list(c("Original","Perturbed")),
        #                         label = "p.signif",paired = T)

results=as.data.frame(ezANOVA(data=DatAnova
                              , dv=MeanRTDiff,wid=.(ID),within=.(Model,Test,Phase),
                              ,type=3,detailed=T)$ANOVA)
results$pareta=results$SSn/(results$SSn+results$SSd)
is.num=sapply(results, is.numeric)
results[is.num] =lapply(results[is.num], round, 3)
results

ggplot(DatAnova,aes(x=Model, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))

# -----------------------> Within Full Model
DatAnova = DataAvg[! DataAvg$Test %in% c("Full"),]
DatAnova = DatAnova[DatAnova$Model %in% c("Full"),]

ggplot(DatAnova,aes(x=Phase, y=MeanRTDiff, fill = Phase)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))
 # stat_compare_means( comparisons = list(c("Original","Perturbed")),
 #                         label = "p.signif",paired = T)

results=as.data.frame(ezANOVA(data=DatAnova
                              , dv=MeanRTDiff,wid=.(ID),within=.(Test,Phase),
                              ,type=3,detailed=T)$ANOVA)
results$pareta=results$SSn/(results$SSn+results$SSd)
is.num=sapply(results, is.numeric)
results[is.num] =lapply(results[is.num], round, 3)
results

DatAnova = DataAvg[DataAvg$Model %in% c("Full"),]
DatAnova = DataAvg[DatAnova$Phase %in% c("Original"),]

ListofComparisons = list(c("Full","Gamma"),
                         c("Full","Alpha"),
                         c("Full","Beta"),
                         c("Full","Delta"),
                         c("Full","Theta"))

ggplot(DatAnova,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        # facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))
# stat_compare_means( comparisons = ListofComparisons,
#                          label = "p.format",paired = F,method = "t.test")
graph2ppt(file="Fig2.pptx",width = 8, height = 5)

Model = aov(MeanRTDiff~Test
            ,data = DatAnova)
summary(Model)

# -----------------------> Within Subband Model
DatAnova = DataAvg[! DataAvg$Test %in% c("Full"),]
DatAnova = DatAnova[DatAnova$Model %in% c("Subband"),]

ggplot(DatAnova,aes(x=Phase, y=MeanRTDiff, fill = Phase)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))

# stat_compare_means( comparisons = list(c("Original","Perturbed")),
#                         label = "p.signif",paired = T)

results=as.data.frame(ezANOVA(data=DatAnova
                              , dv=MeanRTDiff,wid=.(ID),within=.(Test,Phase),
                              ,type=3,detailed=T)$ANOVA)
results$pareta=results$SSn/(results$SSn+results$SSd)
is.num=sapply(results, is.numeric)
results[is.num] =lapply(results[is.num], round, 3)
results

ggplot(DatAnova,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        # facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))

# -----------------------> Within Original
DatAnova = DataAvg[! DataAvg$Test %in% c("Full"),]
DatAnova = DatAnova[DatAnova$Phase %in% c("Original"),]

ggplot(DatAnova,aes(x=Model, y=MeanRTDiff, fill = Model)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))

# stat_compare_means( comparisons = list(c("Original","Perturbed")),
#                         label = "p.signif",paired = T)

results=as.data.frame(ezANOVA(data=DatAnova
                              , dv=MeanRTDiff,wid=.(ID),within=.(Test,Model),
                              ,type=3,detailed=T)$ANOVA)
results$pareta=results$SSn/(results$SSn+results$SSd)
is.num=sapply(results, is.numeric)
results[is.num] =lapply(results[is.num], round, 3)
results

# -----------------------> Within Perturbed
DatAnova = DataAvg[! DataAvg$Test %in% c("Full","Alpha","Beta","Theta"),]
# DatAnova = DatAnova[DatAnova$Phase %in% c("Perturbed"),]

ggplot(DatAnova,aes(x=Model, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        # geom_jitter(position = position_jitterdodge(),aes(colour = Model))+
        facet_wrap(~Phase)+
        theme(strip.text.x = element_text(size=10, face="bold"))+
        labs(x="Trained on",y="MSE", size=10)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 10))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=10, face="bold"),
              strip.text.y = element_text(size=10, face="bold"))
# stat_compare_means( comparisons = list(c("Full","Subband")),
#                         label = "p.format",paired = F,label.y = 0.5)
graph2ppt(file="Fig3.pptx",width = 8, height = 5)
results=as.data.frame(ezANOVA(data=DatAnova
                              , dv=MeanRTDiff,wid=.(ID),within=.(Test,Model,Phase),
                              ,type=3,detailed=T)$ANOVA)
results$pareta=results$SSn/(results$SSn+results$SSd)
is.num=sapply(results, is.numeric)
results[is.num] =lapply(results[is.num], round, 3)
results

as.data.frame(summarise(group_by(DatAnova,Model),M = round(mean(MeanRTDiff),digits = 2),SD = round(sd(MeanRTDiff),digits = 2)))
t.test(DatAnova$MeanRTDiff[DatAnova$Model=="Full"],DatAnova$MeanRTDiff[DatAnova$Model=="Subband"])


# A[abs(scale(A, center=TRUE, scale=TRUE))>3]=NA
# B[abs(scale(B, center=TRUE, scale=TRUE))>3]=NA
wilcox.test(MeanRTDiff~Model , data = DatAnova[DatAnova$Test=="Gamma" & DatAnova$Phase == "Perturbed",],paired = F)
wilcox.test(MeanRTDiff~Model , data = DatAnova[DatAnova$Test=="Gamma" & DatAnova$Phase == "Original",],paired = F)
wilcox.test(MeanRTDiff~Model , data = DatAnova[DatAnova$Test=="Delta" & DatAnova$Phase == "Perturbed",],paired = F)
wilcox.test(MeanRTDiff~Model , data = DatAnova[DatAnova$Test=="Delta" & DatAnova$Phase == "Original",],paired = F)

as.data.frame(summarise(group_by(DatAnova,Model,Test,Phase),M = round(mean(MeanRTDiff),digits = 2),SD = round(sd(MeanRTDiff),digits = 2)))
t.test(DatAnova$MeanRTDiff[DatAnova$Model=="Full" & DatAnova$Test=="Delta" & DatAnova$Phase == "Original"],
       DatAnova$MeanRTDiff[DatAnova$Model=="Subband"& DatAnova$Test=="Delta" & DatAnova$Phase == "Original"],paired = T)
t.test(DatAnova$MeanRTDiff[DatAnova$Model=="Full" & DatAnova$Test=="Gamma" & DatAnova$Phase == "Original"],
       DatAnova$MeanRTDiff[DatAnova$Model=="Subband"& DatAnova$Test=="Gamma" & DatAnova$Phase == "Original"],paired = T)
t.test(DatAnova$MeanRTDiff[DatAnova$Model=="Full" & DatAnova$Test=="Delta" & DatAnova$Phase == "Perturbed"],
       DatAnova$MeanRTDiff[DatAnova$Model=="Subband"& DatAnova$Test=="Delta" & DatAnova$Phase == "Perturbed"], paired = T)


# ----------->  ttests
Fulldat = DataAvg$MeanRTDiff[DataAvg$Test=="Full"]
BandDat_Orig = DataAvg$MeanRTDiff[DataAvg$Model=="Full" & DataAvg$Test=="Alpha" & DataAvg$Phase == "Original"]
BandDat_Pert = DataAvg$MeanRTDiff[DataAvg$Model=="Full" & DataAvg$Test=="Alpha" & DataAvg$Phase == "Perturbed"]
t.test(Fulldat,BandDat_Orig)
t.test(BandDat_Orig,BandDat_Pert)
BandDat_Orig = DataAvg$MeanRTDiff[DataAvg$Model=="Full" & DataAvg$Test=="Gamma" & DataAvg$Phase == "Original"]
BandDat_Pert = DataAvg$MeanRTDiff[DataAvg$Model=="Full" & DataAvg$Test=="Gamma" & DataAvg$Phase == "Perturbed"]
t.test(BandDat_Orig,BandDat_Pert)

########################################### Plotting Data ####################################
DataAvg = summarise(group_by(trialLevelDat,ID,Train,Test),MeanRTDiff = mean(DeltaRT^2,na.rm=T))

ggplot(DataAvg,aes(x=Train, y=MeanRTDiff, fill = Train)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("Full","Subband")),
                            label = "p.signif",label.y = .5)

ggplot(DataAvg,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Train,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( method = "anova",
                            label = "p.signif",label.y = .5)

ggplot(DataAvg[DataAvg$Train=="Subband",],aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        # facet_wrap(~Train,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 10))+
+        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( method = "anova",
                            label = "p.signif",label.y = .5)

DataAvg$Test = as.character(DataAvg$Test)
DataAvg$Test = gsub("_Filtered", "_Phase_perturbed", DataAvg$Test, ignore.case = FALSE, perl = FALSE,
                    fixed = FALSE, useBytes = FALSE)
DataAvg$Test = as.factor(DataAvg$Test)

Dat = DataAvg[DataAvg$Train == "Subband",]
Dat$Sub = case_when(grepl("Alpha",Dat$Test, ignore.case = TRUE) ~ "Alpha",
                    grepl("Beta",Dat$Test, ignore.case = TRUE) ~ "Beta",
                    grepl("Delta",Dat$Test, ignore.case = TRUE) ~ "Delta",
                    grepl("Theta",Dat$Test, ignore.case = TRUE) ~ "Theta",
                    grepl("Gamma",Dat$Test, ignore.case = TRUE) ~ "Gamma") 

Dat$Test = case_when(grepl("Phase",Dat$Test, ignore.case = TRUE) ~ "Perturbed",
                     TRUE ~ "Original") 


ggplot(Dat,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Sub,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT_ Train on subband")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("Original","Perturbed")),
                            method = "t.test", label = "p.signif",label.y = .5, paired = T)

Dat = DataAvg[DataAvg$Train == "Full",]
Dat$Sub = case_when(grepl("Alpha",Dat$Test, ignore.case = TRUE) ~ "Alpha",
                    grepl("Beta",Dat$Test, ignore.case = TRUE) ~ "Beta",
                    grepl("Delta",Dat$Test, ignore.case = TRUE) ~ "Delta",
                    grepl("Theta",Dat$Test, ignore.case = TRUE) ~ "Theta",
                    grepl("Gamma",Dat$Test, ignore.case = TRUE) ~ "Gamma") 

Dat$Test = case_when(grepl("Phase",Dat$Test, ignore.case = TRUE) ~ "Perturbed",
                     TRUE ~ "Original") 


ggplot(Dat,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Sub,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT_ Train on full")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("Original","Perturbed")),
                            method = "t.test", label = "p.signif",label.y = .5, paired = T)

########################################### Plotting Data in new Format 14 July ####################################
DataAvg = summarise(group_by(trialLevelDat,ID,Train,Test),MeanRTDiff = mean(DeltaRT^2,na.rm=T))

DataAvg$Test = as.character(DataAvg$Test)
DataAvg$Test = gsub("_Filtered", "_Phase_perturbed", DataAvg$Test, ignore.case = FALSE, perl = FALSE,
     fixed = FALSE, useBytes = FALSE)
DataAvg$Test = as.factor(DataAvg$Test)

Desired = c("Alpha","Beta", "Delta", "Full", "Gamma", "Theta")

ggplot(DataAvg[!(DataAvg$Test %in% Desired),],aes(x=Train, y=MeanRTDiff, fill = Train)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 10))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=10, face="bold"),
              strip.text.y = element_text(size=10, face="bold"))+
        stat_compare_means( comparisons = list(c("Full","Subband")),
                            label = "p.signif",label.y = .5)
graph2ppt(file="Fig1.pptx",width = 8, height = 5)

Desired = c("Alpha","Beta", "Delta", "Full", "Gamma", "Theta")
ggplot(DataAvg[DataAvg$Test %in% Desired,],aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Train,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 10))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=10, face="bold"),
              strip.text.y = element_text(size=10, face="bold"))+
        stat_compare_means( method = "anova",
                            label = "p.signif",label.y = .5)
graph2ppt(file="Fig2.pptx",width = 8, height = 5)

########################### Load Vigour ##########################
BadIDs = c("8","9")
TrainOnFullFileName = "TrainOnFull2_Results4AK_Vigour.xlsx"
TrainOnSBFileName = "TrainOnSB2_Results4AK_Vigour.xlsx"
dat = NULL
sheetNames = excel_sheets(TrainOnFullFileName)
for(ShName in sheetNames){
        temp      = read_xlsx(TrainOnFullFileName, sheet = ShName)
        temp = melt(temp, id.vars = c("Trial"), variable.name = "ID",value.name = "DeltaRT")
        temp$Train = unique("Full")
        temp$Test = unique(ShName)
        dat = rbind(dat,temp)
}

sheetNames = excel_sheets(TrainOnSBFileName)
for(ShName in sheetNames){
        temp      = read_xlsx(TrainOnSBFileName, sheet = ShName)
        temp = melt(temp, id.vars = c("Trial"), variable.name = "ID",value.name = "DeltaRT")
        temp$Train = unique("Subband")
        temp$Test = unique(ShName)
        dat = rbind(dat,temp)
}

dat$ID = factor(dat$ID)
dat = dat[!(dat$ID %in% BadIDs),]
dat = dat[complete.cases(dat),]

names(dat)
trialLevelDat = dat[,c("ID","Trial","Train","Test","DeltaRT")]
trialLevelDat$Trial = as.factor(trialLevelDat$Trial)
trialLevelDat$Train = as.factor(trialLevelDat$Train)
trialLevelDat$Test = as.factor(trialLevelDat$Test)
write.csv(trialLevelDat,"AK_TriallevelData_Vigour.csv",row.names = FALSE)

########################################### Plotting Data ####################################
DataAvg = summarise(group_by(trialLevelDat,ID,Train,Test),MeanRTDiff = mean(DeltaRT^2,na.rm=T))

ggplot(DataAvg,aes(x=Train, y=MeanRTDiff, fill = Train)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data Vigour")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("Full","Subband")),
                            label = "p.signif",label.y = .85)

ggplot(DataAvg,aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Train,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data Vigour")+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( method = "anova",
                            label = "p.signif",label.y = .85)

########################################### Plotting Data in new Format 14 July ####################################
DataAvg = summarise(group_by(trialLevelDat,ID,Train,Test),MeanRTDiff = mean(DeltaRT^2,na.rm=T))

DataAvg$Test = as.character(DataAvg$Test)
DataAvg$Test = gsub("_Filtered", "_Phase_perturbed", DataAvg$Test, ignore.case = FALSE, perl = FALSE,
                    fixed = FALSE, useBytes = FALSE)
DataAvg$Test = as.factor(DataAvg$Test)

Desired = c("Alpha","Beta", "Delta", "Full", "Gamma", "Theta")

ggplot(DataAvg[!(DataAvg$Test %in% Desired),],aes(x=Train, y=MeanRTDiff, fill = Train)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 10))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=10, face="bold"),
              strip.text.y = element_text(size=10, face="bold"))+
        stat_compare_means( comparisons = list(c("Full","Subband")),
                            label = "p.signif",label.y = .5)
graph2ppt(file="Fig1_Vigour.pptx",width = 8, height = 5)

Desired = c("Alpha","Beta", "Delta", "Full", "Gamma", "Theta")
ggplot(DataAvg[DataAvg$Test %in% Desired,],aes(x=Test, y=MeanRTDiff, fill = Test)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(~Train,nrow = 2)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="Test on",y="", size=16)+
        ggtitle("Data RT")+
        theme(axis.title.y = element_text(size = 10))+
        theme(axis.text.x = element_text(size = 10))+
        theme(strip.text.x = element_text(size=10, face="bold"),
              strip.text.y = element_text(size=10, face="bold"))+
        stat_compare_means( method = "anova",
                            label = "p.signif",label.y = .5)
graph2ppt(file="Fig2_Vigour.pptx",width = 8, height = 5)


############################################Compare Vigour and RT #############################
VigDat = read.csv("AK_TriallevelData_Vigour.csv")
RTDat = read.csv("AK_TriallevelData_RT.csv")
VigDat$Var = unique("Vigour")
RTDat$Var = unique("RT")
Dat = rbind(VigDat,RTDat)
DataAvg = summarise(group_by(Dat,ID,Train,Test,Var),MeanRTDiff = mean(DeltaRT^2,na.rm=T))


ggplot(DataAvg[DataAvg$Train=="Full",],aes(x=Var, y=MeanRTDiff, fill = Var)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(.~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Tained on Full")+
        coord_cartesian(ylim=c(0, 1))+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("RT","Vigour")),
                            label = "p.signif",label.y = .85)

ggplot(DataAvg[DataAvg$Train=="Subband",],aes(x=Var, y=MeanRTDiff, fill = Var)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(.~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Tained on Subband")+
        coord_cartesian(ylim=c(0, 1))+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("RT","Vigour")),
                            label = "p.signif",label.y = .85)


ggplot(DataAvg[DataAvg$Train=="Full",],aes(x=Var, y=MeanRTDiff, fill = Var)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(.~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Tained on Full")+
        coord_cartesian(ylim=c(0, 1))+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("RT","Vigour")),
                            label = "p.signif",label.y = .85)

ggplot(DataAvg[DataAvg$Train=="Subband",],aes(x=Var, y=MeanRTDiff, fill = Var)) + 
        geom_bar(stat="summary",fun.y="mean",position="dodge")+
        stat_summary(fun.data = "mean_se", geom="errorbar",position="dodge")+
        facet_wrap(.~Test)+
        theme(strip.text.x = element_text(size=16, face="bold"))+
        labs(x="",y="", size=16)+
        ggtitle("Tained on Subband")+
        coord_cartesian(ylim=c(0, 1))+
        theme(axis.title.y = element_text(size = 18))+
        theme(axis.text.x = element_text(size = 16))+
        theme(strip.text.x = element_text(size=16, face="bold"),
              strip.text.y = element_text(size=16, face="bold"))+
        stat_compare_means( comparisons = list(c("RT","Vigour")),
                            label = "p.signif",label.y = .85)

