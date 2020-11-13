import MyUtils


IDs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

for pt in IDs: 

#    subband = 'full'
#    print ("pt= " + str(pt+1))
#    MyUtils.TrainTest(pt, subband)

    subband = 'bandAlpha'
    print ("pt= " + str(pt+1))
    MyUtils.TrainTest(pt, subband)
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(pt) + '.h5'
    MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)
    
    subband = 'bandBeta'
    print ("pt= " + str(pt+1))
    MyUtils.TrainTest(pt, subband)
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(pt) + '.h5'
    MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)
  
    subband = 'bandDelta'
    print ("pt= " + str(pt+1))
    MyUtils.TrainTest(pt, subband)
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(pt) + '.h5'
    MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)
  
    subband = 'bandGamma'
    print ("pt= " + str(pt+1))
    MyUtils.TrainTest(pt, subband)
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(pt) + '.h5'
    MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)

    subband = 'bandTheta'
    print ("pt= " + str(pt+1))
    MyUtils.TrainTest(pt, subband)
    FileName_BestModel = 'BestModel_' + subband + '_excl' + str(pt) + '.h5'
    MyUtils.NewTest(FileName_BestModel, subband+'Filtered', pt)
  
   
  