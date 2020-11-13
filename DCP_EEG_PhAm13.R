setwd("C:/Elham/EEG_PatientIdentification/P1/P1_SHAM_TASK_AK_GoodResult4Paper")
library (R.matlab)
library(bnlearn)
library (tidyr)

XY <- readMat("Z_13.mat")


ALL <- XY$Z13

A <- c("del-A", "del-P", "tet-A" ,"tet-P" ,"alp-A", "alp-P", "bet-A" ,"bet-P", "gam-A", "gam-P", "RT_HL")
colnames(ALL) <- A
data <- as.data.frame(ALL)

f = dim(data)[2]

data = drop_na(data)

bn_hc  <- hc        (data)
fittedbn <- bn.fit(bn_hc, data = data)
strength = arc.strength(bn_hc, data)
strength.plot(bn_hc, strength, layout = "fdp", shape = "ellipse", main = "BN-RT for phase/amp at sub-bands of Task EEG data")

library(fpc)
dp = discrproj(data[,1:10], data[,11],  method = "adc", clnum = 0)
print(dp$ev[1:5])
#eigenvalues in descending order, usually indicating portion of information in the corresponding direction.

plotcluster(data[,1:10], data[,11], method = "adc", clnum = 0, main = "RT:H/L")

plotcluster(data[,1:10], data[,11], method = "arc", clnum = 0, main = "RT:H/L")


#columns are coordinates of projection basis vectors. New points x can be projected onto the projection basis vectors by x %*% units
U = dp$units
barplot(U[1:10,1], ylim=c(-100,100))
barplot(U[1:10,2], ylim=c(-100,100))
