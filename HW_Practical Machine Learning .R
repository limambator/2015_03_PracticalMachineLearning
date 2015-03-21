setwd("D:/!Courseria/2015_03 Practical Machine Learning/HW")
dataset_test <- read.table("pml-testing.csv", header = TRUE, sep = ",", quote = "\"",
                           dec = ".", fill = TRUE)

dataset_train <- read.table("pml-training.csv", header = TRUE, sep = ",", quote = "\"",
                            dec = ".", fill = TRUE)

library(caret);
inTrain <- createDataPartition(y=dataset_train$classe,
                               p=0.75, list=FALSE)
set.seed(123)
training <- dataset_train[inTrain,]
testing <-  dataset_train[-inTrain,]

#dataset_train <-  read.csv("pml-training.csv", header = TRUE, sep = ",", quote = "\"",
#         dec = ".", fill = TRUE, comment.char = "")

#library(dplyr)
#tt = arrange(dataset_test , desc(num_window))

## remove some zero variance predictors and linear dependencies
mdrrDescr1 <- training[, -nearZeroVar(training)]

# exclude variables
myvars <- names(mdrrDescr1 ) %in% c("classe", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window")
mdrrDescr1.1  <- mdrrDescr1[!myvars]

# eliminate any column that has 'NA' greater than 15% of the column length.                                     )
mdrrDescr1.2 = mdrrDescr1.1[, colMeans(is.na(mdrrDescr1.1)) <= .15]

#  Hight Correlation
mdrrDescr2 <- mdrrDescr1.2[, -findCorrelation(cor(mdrrDescr1.2), .95)]


mdrrDescr1.3 = mdrrDescr1[c("X","classe")]
mdrrDescr2.1 = merge(mdrrDescr2,mdrrDescr1.3,by="X")

svmLFit <- train(classe ~ ., data = mdrrDescr2.1[,c(-1)],method = "svmLinear")
svmPFit <- train(classe ~ ., data = mdrrDescr2.1[,c(-1)],method = "svmPoly")
svmRFit <- train(classe ~ ., data = mdrrDescr2.1[,c(-1)],method = "svmRadial")

rfFit <- train(classe ~ .,method="rf", data=mdrrDescr2.1[,c(-1)])
svmPFit <- train(classe ~ ., data = mdrrDescr2.1[,c(-1)],method = "svmPoly")

rfFit

# Var Importance RF
vI <- varImp(rfFit$finalModel)
vI$sample <- row.names(vI); vI <- vI[order(vI$Overall, decreasing = T),]
vI

confusionMatrix(testing$classe,predict(rfFit,testing))
save.image("D:/!DiskD/!Courseria/2015_03 Practical Machine Learning/HW/rfFit.RData")

featurePlot(x = testing[, vI$sample[1:2]],
            y = testing$classe,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

library(psych)
describe(testing[, vI$sample[1:20]])

PredictorVariables <- vI$sample[1:20]

Formula <- formula(paste("classe ~ ",
                         paste(PredictorVariables, collapse=" + ")))
Formula

rfFit_top20 <- train(Formula,method="rf", data=mdrrDescr2.1[,c(-1)])
rfFit_top20

save.image("D:/!Courseria/2015_03 Practical Machine Learning/HW/rfFit_frFit20.RData")

rfFit_top20$finalModel
#
# fitControl <- trainControl(## 10-fold CV
#   method = "repeatedcv",
#   number = 10,
#   ## repeated ten times
#   repeats = 10)

rfFit_top20rrf <- train(Formula,method='RRF', data=mdrrDescr2.1[,c(-1)])



1preProcRf <- preProcess(mdrrDescr2.1[,c(-1,-51)],method=c("BoxCox", "scale"))
trainPCRf <- predict(preProcRf,mdrrDescr2.1[,c(-1,-51)])

trainPC_Rfclasse <-  cbind(trainPCRf, mdrrDescr2.1[c(51)])

rfFit_top20.1 <- train(Formula,method="rf", data=trainPC_Rfclasse )
rfFit_top20.1

confusionMatrix(testing$classe,predict(rfFit_top20,testing))

save.image("D:/!DiskD/!Courseria/2015_03 Practical Machine Learning/HW/rfFitAndFit20.RData")

#PCA
preProc <- preProcess(mdrrDescr2.1[,c(-1,-51)],method=c("BoxCox", "scale","pca"), thresh = 0.9)
trainPC <- predict(preProc,mdrrDescr2.1[,c(-1,-51)])

# exclude variables
names1 <-names(mdrrDescr2.1[,c(-1,-51)])
testingPC  <- testing[names1]

trainPC_classe <-  cbind(trainPC, mdrrDescr2.1[c(51)])
rfFit_PC <- train(classe ~ .,method="rf", data=trainPC_classe)

#
testPC <- predict(preProc,testingPC)
testPC_classe <-  cbind(testPC , testing[c(160)])

confusionMatrix(testPC_classe$classe,predict(rfFit_PC,testPC_classe))

save.image("D:/!DiskD/!Courseria/2015_03 Practical Machine Learning/HW/rfFitAndFit20andPCA.RData")

#
preProcValues <- preProcess(training["magnet_belt_y"], method = c("BoxCox", "scale","pca"))
trainTransformed <- predict(preProcValues, training["magnet_belt_y"])
training_lg <- log10(training["magnet_belt_y"])

describe(trainTransformed)
describe(training_lg)

hist(trainTransformed$magnet_belt_y)
hist(training_lg$magnet_belt_y)
#

testTransformed <- predict(preProcValues, test)

# Random Forest rfFit_PC
# Confusion Matrix and Statistics
#
# Reference
# Prediction    A    B    C    D    E
# A 1386    2    2    2    3
# B   12  921   14    0    2
# C    5   12  831    5    2
# D    0    2   37  765    0
# E    1    5    2    4  889
#
# Overall Statistics
#
# Accuracy : 0.9772          
# 95% CI : (0.9726, 0.9812)
# No Information Rate : 0.2863          
# P-Value [Acc > NIR] : < 2.2e-16       
#
# Kappa : 0.9711          
# Mcnemar's Test P-Value : 4.488e-06       
#
# Statistics by Class:
#
# Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9872   0.9777   0.9379   0.9858   0.9922
# Specificity            0.9974   0.9929   0.9940   0.9906   0.9970
# Pos Pred Value         0.9935   0.9705   0.9719   0.9515   0.9867
# Neg Pred Value         0.9949   0.9947   0.9864   0.9973   0.9983
# Prevalence             0.2863   0.1921   0.1807   0.1582   0.1827
# Detection Rate         0.2826   0.1878   0.1695   0.1560   0.1813
# Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
# Balanced Accuracy      0.9923   0.9853   0.9660   0.9882   0.9946



# Random Forest rfFit_top20
#
# 14718 samples
# 49 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E'
#
# No pre-processing
# Resampling: Bootstrapped (25 reps)
#
# Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ...
#
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
# 2    0.9872683  0.9838870  0.001907278  0.002411886
# 11    0.9866239  0.9830723  0.002121336  0.002685727
# 20    0.9771993  0.9711475  0.004594563  0.005812273
#
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 2.
# > confusionMatrix(testing$classe,predict(rfFit_top20,testing))
# Confusion Matrix and Statistics
#
# Reference
# Prediction    A    B    C    D    E
# A 1391    2    1    0    1
# B   10  931    5    1    2
# C    0    9  843    3    0
# D    0    0   13  789    2
# E    0    0    1    0  900
#
# Overall Statistics
#
# Accuracy : 0.9898          
# 95% CI : (0.9866, 0.9924)
# No Information Rate : 0.2857          
# P-Value [Acc > NIR] : < 2.2e-16       
#
# Kappa : 0.9871          
# Mcnemar's Test P-Value : NA              
#
# Statistics by Class:
#
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9929   0.9883   0.9768   0.9950   0.9945
# Specificity            0.9989   0.9955   0.9970   0.9964   0.9997
# Pos Pred Value         0.9971   0.9810   0.9860   0.9813   0.9989
# Neg Pred Value         0.9972   0.9972   0.9951   0.9990   0.9988
# Prevalence             0.2857   0.1921   0.1760   0.1617   0.1845
# Detection Rate         0.2836   0.1898   0.1719   0.1609   0.1835
# Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
# Balanced Accuracy      0.9959   0.9919   0.9869   0.9957   0.9971


# Var Importance svmLFit
varImp(svmLFit)

svmLFit

# Var Importance svmPFit
varImp(svmPFit)
svmPFit

# Var Importance svmPFit
varImp(svmRFit)
svmRFit


# Random Forest
# Confusion Matrix and Statistics
#
# Reference
# Prediction    A    B    C    D    E
# A 1394    0    0    0    1
# B    4  942    2    0    1
# C    0    4  844    7    0
# D    0    0    8  796    0
# E    0    0    1    0  900
#
# Overall Statistics
#
# Accuracy : 0.9943          
# 95% CI : (0.9918, 0.9962)
# No Information Rate : 0.2851          
# P-Value [Acc > NIR] : < 2.2e-16       
#
# Kappa : 0.9928          
# Mcnemar's Test P-Value : NA              
#
# Statistics by Class:
#
# Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9971   0.9958   0.9871   0.9913   0.9978
# Specificity            0.9997   0.9982   0.9973   0.9980   0.9998
# Pos Pred Value         0.9993   0.9926   0.9871   0.9900   0.9989
# Neg Pred Value         0.9989   0.9990   0.9973   0.9983   0.9995
# Prevalence             0.2851   0.1929   0.1743   0.1637   0.1839
# Detection Rate         0.2843   0.1921   0.1721   0.1623   0.1835
# Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
# Balanced Accuracy      0.9984   0.9970   0.9922   0.9947   0.9988
#
# Overall               sample
# yaw_belt             1216.01149             yaw_belt
# pitch_forearm        1009.56773        pitch_forearm
# pitch_belt            828.18133           pitch_belt
# magnet_dumbbell_z     768.50149    magnet_dumbbell_z
# magnet_dumbbell_y     624.80694    magnet_dumbbell_y
# roll_forearm          571.11319         roll_forearm
# magnet_belt_y         492.05002        magnet_belt_y
# magnet_belt_z         368.50862        magnet_belt_z
# gyros_belt_z          358.74612         gyros_belt_z
# magnet_dumbbell_x     322.44144    magnet_dumbbell_x
# roll_dumbbell         312.77972        roll_dumbbell
# accel_dumbbell_y      298.92260     accel_dumbbell_y
# accel_forearm_x       258.49386      accel_forearm_x
# accel_dumbbell_z      245.15819     accel_dumbbell_z
# total_accel_dumbbell  243.22883 total_accel_dumbbell
# accel_forearm_z       235.67125      accel_forearm_z
# magnet_belt_x         231.80374        magnet_belt_x
# total_accel_belt      227.71492     total_accel_belt
# magnet_forearm_z      213.34576     magnet_forearm_z
# yaw_arm               195.90723              yaw_arm
# roll_arm              185.73774             roll_arm
# yaw_dumbbell          149.50586         yaw_dumbbell
# gyros_dumbbell_y      138.22885     gyros_dumbbell_y
# magnet_arm_x          137.89429         magnet_arm_x
# magnet_forearm_x      131.76648     magnet_forearm_x
# magnet_arm_y          123.83654         magnet_arm_y
# magnet_forearm_y      119.92177     magnet_forearm_y
# magnet_arm_z          114.70783         magnet_arm_z
# yaw_forearm           106.68653          yaw_forearm
# accel_dumbbell_x      105.13015     accel_dumbbell_x
# accel_arm_x           104.58496          accel_arm_x
# pitch_arm              97.47849            pitch_arm
# accel_arm_y            95.83920          accel_arm_y
# gyros_arm_y            94.92206          gyros_arm_y
# pitch_dumbbell         79.30832       pitch_dumbbell
# gyros_dumbbell_x       75.24372     gyros_dumbbell_x
# accel_belt_y           73.88358         accel_belt_y
# gyros_arm_x            72.28306          gyros_arm_x
# accel_forearm_y        71.16230      accel_forearm_y
# total_accel_forearm    68.62425  total_accel_forearm
# gyros_belt_y           68.33931         gyros_belt_y
# gyros_forearm_y        67.45304      gyros_forearm_y
# accel_arm_z            63.98242          accel_arm_z
# gyros_belt_x           58.69263         gyros_belt_x
# total_accel_arm        54.30017      total_accel_arm
# gyros_dumbbell_z       47.53267     gyros_dumbbell_z
# gyros_forearm_z        43.49702      gyros_forearm_z
# gyros_forearm_x        33.73254      gyros_forearm_x
# gyros_arm_z            28.44260          gyros_arm_z

#######################################################
# svmLFit

# ROC curve variable importance
#
# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 48)
#
# A     B     C      D     E
# pitch_forearm     100.00 62.85 70.84 100.00 67.13
# roll_dumbbell      51.27 62.38 83.86  83.86 57.39
# accel_forearm_x    81.06 49.49 62.68  81.06 45.62
# magnet_arm_x       78.04 52.95 54.92  78.04 64.89
# magnet_arm_y       76.49 38.65 53.35  76.49 67.35
# accel_arm_x        73.01 50.74 47.09  73.01 61.38
# pitch_dumbbell     52.70 71.21 71.21  61.17 46.33
# magnet_forearm_x   71.03 49.73 38.70  71.03 41.95
# magnet_belt_y      67.25 59.89 61.36  61.86 67.25
# magnet_dumbbell_y  46.09 65.27 65.27  46.73 52.21
# magnet_dumbbell_x  65.23 65.23 64.37  50.30 51.53
# accel_dumbbell_x   56.78 57.14 57.14  47.35 39.29
# magnet_dumbbell_z  56.18 23.49 56.18  36.53 53.79
# magnet_arm_z       52.30 52.30 36.53  39.90 49.16
# magnet_belt_z      50.44 49.20 49.01  50.82 50.82
# pitch_arm          48.88 26.52 36.92  41.99 48.88
# magnet_forearm_y   37.22 24.29 44.90  44.90 34.85
# accel_dumbbell_z   42.20 42.20 40.03  21.24 29.28
# yaw_dumbbell       18.91 41.09 41.09  17.84 27.20
# accel_dumbbell_y   34.73 40.91 40.91  37.54 26.55
# Warning message:
#   package ‘pROC’ was built under R version 3.0.3
#  svmLFit
# Support Vector Machines with Linear Kernel
#
# 19622 samples
# 48 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E'
#
# No pre-processing
# Resampling: Bootstrapped (25 reps)
#
# Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ...
#
# Resampling results
#
# Accuracy   Kappa      Accuracy SD  Kappa SD   
# 0.7626239  0.6981214  0.003571158  0.004412452
#
# Tuning parameter 'C' was held constant at a value of 1

#########################################################################

# varImp(svmPFit)
# ROC curve variable importance
#
# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 48)
#
# A     B     C      D     E
# pitch_forearm     100.00 62.85 70.84 100.00 67.13
# roll_dumbbell      51.27 62.38 83.86  83.86 57.39
# accel_forearm_x    81.06 49.49 62.68  81.06 45.62
# magnet_arm_x       78.04 52.95 54.92  78.04 64.89
# magnet_arm_y       76.49 38.65 53.35  76.49 67.35
# accel_arm_x        73.01 50.74 47.09  73.01 61.38
# pitch_dumbbell     52.70 71.21 71.21  61.17 46.33
# magnet_forearm_x   71.03 49.73 38.70  71.03 41.95
# magnet_belt_y      67.25 59.89 61.36  61.86 67.25
# magnet_dumbbell_y  46.09 65.27 65.27  46.73 52.21
# magnet_dumbbell_x  65.23 65.23 64.37  50.30 51.53
# accel_dumbbell_x   56.78 57.14 57.14  47.35 39.29
# magnet_dumbbell_z  56.18 23.49 56.18  36.53 53.79
# magnet_arm_z       52.30 52.30 36.53  39.90 49.16
# magnet_belt_z      50.44 49.20 49.01  50.82 50.82
# pitch_arm          48.88 26.52 36.92  41.99 48.88
# magnet_forearm_y   37.22 24.29 44.90  44.90 34.85
# accel_dumbbell_z   42.20 42.20 40.03  21.24 29.28
# yaw_dumbbell       18.91 41.09 41.09  17.84 27.20
# accel_dumbbell_y   34.73 40.91 40.91  37.54 26.55
# > svmPFit
# Support Vector Machines with Polynomial Kernel
#
# 19622 samples
# 48 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E'
#
# No pre-processing
# Resampling: Bootstrapped (25 reps)
#
# Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ...
#
# Resampling results across tuning parameters:
#   
#   degree  scale  C     Accuracy   Kappa      Accuracy SD  Kappa SD   
# 1       0.001  0.25  0.5389580  0.4052956  0.008905963  0.011524742
# 1       0.001  0.50  0.5841731  0.4686397  0.007253337  0.009094218
# 1       0.001  1.00  0.6123272  0.5059307  0.007237715  0.009062534
# 1       0.010  0.25  0.6536070  0.5591144  0.005446099  0.006719270
# 1       0.010  0.50  0.6821518  0.5953318  0.004944753  0.006038190
# 1       0.010  1.00  0.7078706  0.6280070  0.005693919  0.007050106
# 1       0.100  0.25  0.7304612  0.6567983  0.005868516  0.007241896
# 1       0.100  0.50  0.7439420  0.6740703  0.005482082  0.006786811
# 1       0.100  1.00  0.7529827  0.6856361  0.005444543  0.006742119
# 2       0.001  0.25  0.5911690  0.4777395  0.007223862  0.009091526
# 2       0.001  0.50  0.6261971  0.5237605  0.007272013  0.009106709
# 2       0.001  1.00  0.6684443  0.5781791  0.005592510  0.006864423
# 2       0.010  0.25  0.8464465  0.8052358  0.007490528  0.009391931
# 2       0.010  0.50  0.8790323  0.8465835  0.005902828  0.007420487
# 2       0.010  1.00  0.9058460  0.8806402  0.005687637  0.007168790
# 2       0.100  0.25  0.9736361  0.9666264  0.001920750  0.002422416
# 2       0.100  0.50  0.9796866  0.9742914  0.002053657  0.002593938
# 2       0.100  1.00  0.9838354  0.9795455  0.001636636  0.002066422
# 3       0.001  0.25  0.6208218  0.5166015  0.007230575  0.008986036
# 3       0.001  0.50  0.6691292  0.5790016  0.006335116  0.007858916
# 3       0.001  1.00  0.7196098  0.6434212  0.008119560  0.010053505
# 3       0.010  0.25  0.9008378  0.8742731  0.006229368  0.007844732
# 3       0.010  0.50  0.9278146  0.9085188  0.006353728  0.008021569
# 3       0.010  1.00  0.9491552  0.9355895  0.005306940  0.006699551
# 3       0.100  0.25  0.9911151  0.9887589  0.001297661  0.001643880
# 3       0.100  0.50  0.9919820  0.9898555  0.001205728  0.001529779
# 3       0.100  1.00  0.9922036  0.9901357  0.001343643  0.001704374
#
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were degree = 3, scale = 0.1 and C = 1.

#########################################################################

# ROC curve variable importance
#
# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 48)

# A     B     C      D     E
# pitch_forearm     100.00 62.85 70.84 100.00 67.13
# roll_dumbbell      51.27 62.38 83.86  83.86 57.39
# accel_forearm_x    81.06 49.49 62.68  81.06 45.62
# magnet_arm_x       78.04 52.95 54.92  78.04 64.89
# magnet_arm_y       76.49 38.65 53.35  76.49 67.35
# accel_arm_x        73.01 50.74 47.09  73.01 61.38
# pitch_dumbbell     52.70 71.21 71.21  61.17 46.33
# magnet_forearm_x   71.03 49.73 38.70  71.03 41.95
# magnet_belt_y      67.25 59.89 61.36  61.86 67.25
# magnet_dumbbell_y  46.09 65.27 65.27  46.73 52.21
# magnet_dumbbell_x  65.23 65.23 64.37  50.30 51.53
# accel_dumbbell_x   56.78 57.14 57.14  47.35 39.29
# magnet_dumbbell_z  56.18 23.49 56.18  36.53 53.79
# magnet_arm_z       52.30 52.30 36.53  39.90 49.16
# magnet_belt_z      50.44 49.20 49.01  50.82 50.82
# pitch_arm          48.88 26.52 36.92  41.99 48.88
# magnet_forearm_y   37.22 24.29 44.90  44.90 34.85
# accel_dumbbell_z   42.20 42.20 40.03  21.24 29.28
# yaw_dumbbell       18.91 41.09 41.09  17.84 27.20
# accel_dumbbell_y   34.73 40.91 40.91  37.54 26.55

# Support Vector Machines with Radial Basis Function Kernel
#
# 19622 samples
# 48 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E'
#
# No pre-processing
# Resampling: Bootstrapped (25 reps)
#
# Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ...
#
# Resampling results across tuning parameters:
#   
#   C     Accuracy   Kappa      Accuracy SD  Kappa SD   
# 0.25  0.8844206  0.8534541  0.005793988  0.007323124
# 0.50  0.9118927  0.8883231  0.004957081  0.006248296
# 1.00  0.9386758  0.9223042  0.004416621  0.005567022
#
# Tuning parameter 'sigma' was held constant at a value of 0.01432956
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were sigma = 0.01432956 and C = 1.

save.image("D:/!DiskD/!Courseria/2015_03 Practical Machine Learning/HW/2.RData")

#

#summary(mdrrDescr2.1)
library(e1071)
fitsvm <- svm( classe ~ ., data=mdrrDescr2.1)

#print(fit)
#plot(fit, mdrrDescr2.1)
tune.out = tune(svm, classe ~ ., data = mdrrDescr2.1, kernel = "linear", ranges = list(cost = c(0.001,
                                                                                                0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)

# We can perform cross-validation using tune() to select the best choice of and cost for an SVM with a radial kernel:

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           #summaryFunction = twoClassSummary
)

svmFit <- train(classe ~ ., data = mdrrDescr2.1,
                method = "svmLinear"#,
                #trControl = fitControl,
                #preProc = c("center", "scale"),
                #tuneLength = 8,
                #metric = "ROC"
)
svmFit


head(dataset_train )
dataset_train$classe

library(psych)
describe(dataset_train)

library(caret); library(kernlab);

inTrain <- createDataPartition(y=dataset_train$classe,
                               p=0.1, list=FALSE)

training <- dataset_train[inTrain,]
testing <- dataset_train[-inTrain,]

modelFit <- train(classe ~.,data=training, method="svmLinear")

warnings()

#######Q2
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

qplot(wage,colour=education,data=training,geom="density")
hist(training$Superplasticizer,main="",xlab="ave. capital run length")
hist(log10(training$Superplasticizer+1),main="",xlab="ave. capital run length")

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

library(dplyr)
tt = select(training, starts_with("IL"))

library(caret)
preProc <- preProcess(tt,method="pca", thresh = 0.9)
summary(preProc)
preProc
head(preProc)

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

library(dplyr)
library(caret)
training = select(training,diagnosis, starts_with("IL"))
modelFit <- train(training$diagnosis ~ .,method="glm",data=training)

summary(training)

glm1 <- glm(training$diagnosis ~.,family="binomial",data=training)
summary(glm1)


library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.1, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

modelFit <- train(as.factor(type) ~.,data=training, method="glm")


#head(tt1[,-1])
#head(tt1)
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
library(dplyr)
library(caret)
training = select(training,diagnosis, starts_with("IL"))
testing = select(testing,diagnosis, starts_with("IL"))

preProc <- preProcess(training[,-1],method="pca", thresh = 0.8)
trainPC <- predict(preProc,training[,-1])
modelFit <- train(training$diagnosis ~ .,method="glm",data=trainPC)

preProc1 <- preProcess(testing[,-1],method="pca", thresh = 0.8)
trainPC1 <- predict(preProc,testing[,-1])

confusionMatrix(testing$diagnosis,predict(modelFit,trainPC1))



predict(modelFit,testing)
confusionMatrix(testing$diagnosis,predict(modelFit,testing))


modelFit1 <- train(training$diagnosis ~ .,method="glm",preProcess="pca",
                   preProcOptions = list(thresh = 0.8),data=training)
modelFit
confusionMatrix(testing$diagnosis,predict(modelFit,testing))

####

library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
trainPC <- predict(preProc,log10(training[,-58]+1))
modelFit <- train(training$type ~ .,method="glm",data=trainPC)
testPC <- predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))


modelFit <- train(training$type ~ .,method="glm",preProcess="pca",preProcOptions = list(thresh = 0.8),data=training)

warnings()


library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]


library(Hmisc)
training$fctrAge <- cut2(training$Age,g=5)
plot(training$CompressiveStrength, col=training$fctrAge, main="Age")
training$fctrFlyAsh <- cut2(training$FlyAsh,g=5)
plot(training$CompressiveStrength, col=training$fctrFlyAsh, main="Age")


CompressiveStrength
##################################################################
#####Q3
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

inTrain <- createDataPartition(y=segmentationOriginal$Case,
                               p=0.8, list=FALSE)
training <- segmentationOriginal[inTrain,]
testing <- segmentationOriginal[-inTrain,]
set.seed(125)

modFit <- train(Class ~ .,method="rpart",data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE,
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

library(rattle)
library(rpart)
fancyRpartPlot(modFit$finalModel)
rpart

test = testing
df1 <- subset(testing, Cell == 207827637)
df1[1:119] <- NA
#TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2
df1$TotalIntench2 = 23.000
df1$FiberWidthCh1 = 10
df1$PerimStatusCh1 = 2
predict(modFit,newdata=df1)

#TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100
df1$TotalIntench2 = 50.000
df1$FiberWidthCh1 = 10
df1$PerimStatusCh4 = 100
predict(modFit,newdata=df1)

#TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100
df1$TotalIntench2 = 57.000
df1$FiberWidthCh1 = 8
df1$PerimStatusCh4 = 100
predict(modFit,newdata=df1)

#FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2
df1$TotalIntench1 = 8
df1$VarIntenCh4 = 8
df1$PerimStatusCh1 = 2
predict(modFit,newdata=df1)


library(rattle)
summary(segmentationOriginal$Case)
inTrain <- grep("Train",segmentationOriginal$Case)
training <- segmentationOriginal[inTrain,]
testing <- segmentationOriginal[-inTrain,]
set.seed(125)
fit <- train(Class~.,data=training,method="rpart")
fancyRpartPlot(fit$finalModel)
predData <- training[1:3,]
which(colnames(training)=="TotalIntenCh2")
which(colnames(training)=="FiberWidthCh1")
which(colnames(training)=="PerimStatusCh1")
#TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2
#FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2
predData[1,c(103,50,85)]=c(23000,10,2)
predData[2,c(103,50,85)]=c(50000,10,100)
predData[3,c(103,50,85)]=c(57000,8,100)
predict(fit,predData)



# Classification Tree with rpart
library(rpart)

# grow tree
fit <- rpart(Class ~ .,
             method="class", data=training)

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# plot tree
plot(fit, uniform=TRUE,
     main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

fancyRpartPlot(fit)

#####Q3
library(pgmm)
data(olive)
olive1 = olive[,-1]

fit <- rpart(Area ~ .,
             method="class", data=olive1)
newdata = as.data.frame(t(colMeans(olive)))
predict(fit, newdata)


modFit <- train(Area ~ .,method="rpart",data=olive1)
predict(modFit,newdata=newdata)

library(tree)
tr = tree(Area ~ .,data=olive1)
predict(tr,newdata)


library(pgmm)
data(olive)
olive = olive[,-1]
newdata = as.data.frame(t(colMeans(olive)))
fit <- train(Area~.,data=olive,method="rpart")
pred <- predict(fit,newdata)
fancyRpartPlot(fit$finalModel)

###Q4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)

modelFit <- train(chd ~ age+alcohol+obesity+tobacco+typea+ldl,method="glm",family="binomial", data=trainSA)



predict(modelFit,trainSA, type = "response")

modelFit1 <- glm((chd) ~ age+alcohol+obesity+tobacco+typea+ldl,family="binomial", data=trainSA)

missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(trainSA$chd, predict(modelFit,trainSA))
missClass(testSA$chd, predict(modelFit,testSA))


missClass(testSA, predict(modelFit,testSA))

###Q5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y = factor(vowel.train$y)

modelFit <- train(y ~ .,method="rf", data=vowel.train)

v = varImp(modelFit$finalModel)
varImp(modelFit$finalModel)
order(v)


qt(0.95, df = 27)
round(-3 - qt(0.95, df = 27)*0.5, digits = 2)
round(-3 + qt(0.95, df = 27)*0.5, digits = 2)
