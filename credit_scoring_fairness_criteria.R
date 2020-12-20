# classification package
install.packages("class")
# decision tree package
install.packages("party")
# this one has an odd name but we do need it - SVM package
install.packages("e1071")
# this is the crossvalidation package - CARET
install.packages("caret")
# support vector machine kernel library
install.packages("kernlab")
#  naive bayes library
install.packages("klaR")
install.packages("rlang")


# now reference all the libraries
library(class)
library(party)
library(e1071)
library(caret)
library(klaR)

# working directory
setwd("/Users/xinyu/desktop/senior fall/Independent Study/dataset/")

# load data, this will look for the file in the working directory
bank <- read.csv("ml bank dataset/bankfullclean.csv")

bank$biY = as.factor(bank$biY)
bank$catAge = as.factor(bank$catAge)
bank$education = as.factor(bank$education)
bank$biDefault = as.factor(bank$biDefault)
bank$biHousing = as.factor(bank$biHousing)
bank$biLoan = as.factor(bank$biLoan)
bank$month = as.factor(bank$month)
# check header
head(bank)

# summary stats
summary(bank)


# TRAIN, TEST --> SPLIT (use TEST for fairness testing purposes)
set.seed(0)
split = sample(size = nrow(bank), c("train", "test"),  replace = TRUE, prob = c(0.8, 0.2))

train = bank[split=="train",] # use this for train control

test = bank[split=="test",] # use this for fairness tests (holdout sample)

nrow(bank)
nrow(train)
nrow(test) #nrow(train) + nrow(test) should add up to nrow(bank)

prop.table(table(bank$biY))

# this implements cross-validation
set.seed(0) # this sets the random generator to the same origin
# k-fold cross validation with 10 folds;
train_Control = trainControl(method = "cv", number = 10)

summary(bank)


########################################################################################################
###########################train data using LINEAR REGRESSION MODEL ####################################
########################################################################################################


########################################################################################################
############################## Protected Attribute 1: MARITAL STATUS ####################################

summary(bank)
linear_regression_caret = train(biY ~ catAge + job + marital + education + balance + day + month + duration + campaign + pdays + previous + Fmonth + biDefault + biHousing + biLoan , 
                                data = train, 
                                method = "glm", 
                                trControl = train_Control, 
                                tuneLength = 10)
linear_regression_caret #Accuracy: 0.892479


# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret, test)

# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277

accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Acurracy:0.8965941


# now let's separate based on marital status to test fairness criteria
prop.table(table(test$marital))

summary(bank$marital)
split_by_marital = split(test, test$marital)
test_divorced = split_by_marital$divorced
# View(test_divorced)

test_married = split_by_marital$married
# View(test_married)

test_single = split_by_marital$single
# View(test_single)


# Test on divorced subset
linear_regression_labels_divorced = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced # Accuracy:0.920398


# Test on married subset
linear_regression_labels_married = predict(linear_regression_caret, test_married)
confusion_matrix_married = table(linear_regression_labels_married, test_married$biY)
accuracy_linear_regression_married = (confusion_matrix_married[1,1] + confusion_matrix_married[2,2]) / sum(confusion_matrix_married)
accuracy_linear_regression_married # Accuracy: 0.902182

# Test on single subset
linear_regression_labels_single = predict(linear_regression_caret, test_single)
confusion_matrix_single = table(linear_regression_labels_single, test_single$biY)
accuracy_linear_regression_single = (confusion_matrix_single[1,1] + confusion_matrix_single[2,2]) / sum(confusion_matrix_single)
accuracy_linear_regression_single # Accuracy: 0.875295

####################################  Demographic Parity  #######################################
#positive rate = (TP + TN)
PR_linear_regression_divorced = confusion_matrix_divorced[2,2] + confusion_matrix_divorced[1,2]
PR_linear_regression_married = confusion_matrix_married[2,2] +confusion_matrix_married[1,2]
PR_linear_regression_single = confusion_matrix_single[2,2] +confusion_matrix_single[1,2]
PR_linear_regression_divorced # 85
PR_linear_regression_married # 561
PR_linear_regression_single # 350

####################################  Equalized Oppportunites  #######################################
#TPR = TP/TP+FN

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03243243
TPR_linear_regression_married # 0.02602992
TPR_linear_regression_single # 0.04988764

####################################  Equalized Odds  #######################################
#TPR = TP/TP+FN
#FNR = FN/FN+TP

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03243243
TPR_linear_regression_married #  0.02602992
TPR_linear_regression_single # 0.04988764


FNR_linear_regression_divorced = confusion_matrix_divorced[1,1] / (confusion_matrix_divorced[1,1] +confusion_matrix_divorced[2,2])
FNR_linear_regression_married = confusion_matrix_married[1,1] / (confusion_matrix_married[1,1] + confusion_matrix_married[2,2])
FNR_linear_regression_single = confusion_matrix_single[1,1] / (confusion_matrix_single[1,1] + confusion_matrix_single[2,2])
FNR_linear_regression_divorced # 0.9675676
FNR_linear_regression_married # 0.9739701
FNR_linear_regression_single # 0.9501124


####################################  Fairness Through unawareness  #######################################
linear_regression_caret2 = train(biY ~ catAge + job + education + balance + day + month + duration + pdays + previous + campaign + duration + campaign + biDefault + biHousing + biLoan , 
                                data = train, 
                                method = "glm", 
                                trControl = train_Control, 
                                tuneLength = 10)

linear_regression_caret2 # Accuracy: 0.8931441

# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret2, test)

confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277
accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Accuracy: 0.8964824


prop.table(table(test$marital))

summary(bank$marital)
split_by_marital = split(test, test$marital)
test_divorced = split_by_marital$divorced
# View(test_divorced2)

test_married = split_by_marital$married
# View(test_married2)

test_single = split_by_marital$single
# View(test_single2)


# Test on divorced subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_divorced = predict(linear_regression_caret2, test_divorced)
confusion_matrix_divorced = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced # Accuracy: 0.9223881

# Test on MARRIED subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_married = predict(linear_regression_caret2, test_married)
confusion_matrix_married = table(linear_regression_labels_married, test_married$biY)
accuracy_linear_regression_married = (confusion_matrix_married[1,1] + confusion_matrix_married[2,2]) / sum(confusion_matrix_married)
accuracy_linear_regression_married # Accuracy:0.9023669

# Test on single subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_single = predict(linear_regression_caret2, test_single)
confusion_matrix_single = table(linear_regression_labels_single, test_single$biY)
accuracy_linear_regression_single = (confusion_matrix_single[1,1] + confusion_matrix_single[2,2]) / sum(confusion_matrix_single)
accuracy_linear_regression_single # Accuracy: 0.8737215




########################################################################################################
############################## Protected Attribute 2: AGE Group ####################################
# <=30 30-40 40-50  >=50 
# 7011 17666 11178  9186 

summary(bank)
linear_regression_caret = train(biY ~ catAge + job + marital + education + balance + day + month + duration + campaign + pdays + previous + Fmonth + biDefault + biHousing + biLoan , 
                                data = train, 
                                method = "glm", 
                                trControl = train_Control, 
                                tuneLength = 10)
linear_regression_caret #Accuracy: 0.892479


# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret, test)

# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277

accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Acurracy:0.8965941


# now let's separate based on marital status to test fairness criteria
prop.table(table(test$catAge))

summary(bank$catAge)
split_by_age = split(test, test$catAge)
test_0_30 = split_by_age$"<=30"
# View(test_0_30)

test_30_40 = split_by_age$"30-40"
# View(test_30_40)

test_40_50 = split_by_age$"40-50"
# View(test_40_50)

test_50_100 = split_by_age$">=50"
# View(test_50_100)


# Test on <= 30 subset
linear_regression_labels_0_30 = predict(linear_regression_caret, test_0_30)
confusion_matrix_0_30 = table(linear_regression_labels_0_30, test_0_30$biY)
accuracy_linear_regression_0_30 = (confusion_matrix_0_30[1,1] + confusion_matrix_0_30[2,2]) / sum(confusion_matrix_0_30)
accuracy_linear_regression_0_30 # Accuracy:0.8679941


# Test on 30-40 subset
linear_regression_labels_30_40 = predict(linear_regression_caret, test_30_40)
confusion_matrix_30_40 = table(linear_regression_labels_30_40, test_30_40$biY)
accuracy_linear_regression_30_40 = (confusion_matrix_30_40[1,1] + confusion_matrix_30_40[2,2]) / sum(confusion_matrix_30_40)
accuracy_linear_regression_30_40 # Accuracy: 0.9041369

# Test on 40-50 subset
linear_regression_labels_40_50 = predict(linear_regression_caret, test_40_50)
confusion_matrix_40_50 = table(linear_regression_labels_40_50, test_40_50$biY)
accuracy_linear_regression_40_50 = (confusion_matrix_40_50[1,1] + confusion_matrix_40_50[2,2]) / sum(confusion_matrix_40_50)
accuracy_linear_regression_40_50 # Accuracy:0.9205712

# Test on >= 50 subset
linear_regression_labels_50_100 = predict(linear_regression_caret, test_50_100)
confusion_matrix_50_100 = table(linear_regression_labels_50_100, test_50_100$biY)
accuracy_linear_regression_50_100 = (confusion_matrix_50_100[1,1] + confusion_matrix_50_100[2,2]) / sum(confusion_matrix_50_100)
accuracy_linear_regression_50_100 # Accuracy:0.874258

####################################  Demographic Parity  #######################################
#positive rate = (TP + TN)
PR_linear_regression_0_30 = confusion_matrix_0_30[2,2] + confusion_matrix_0_30[1,2]
PR_linear_regression_30_40 = confusion_matrix_30_40[2,2] +confusion_matrix_30_40[1,2]
PR_linear_regression_40_50 = confusion_matrix_40_50[2,2] +confusion_matrix_40_50[1,2]
PR_linear_regression_50_100 = confusion_matrix_50_100[2,2] +confusion_matrix_50_100[1,2]
PR_linear_regression_0_30 # 203
PR_linear_regression_30_40 # 336
PR_linear_regression_40_50 # 202
PR_linear_regression_50_100 # 255

####################################  Equalized Oppportunites  #######################################
#TPR = TP/TP+FN
TPR_linear_regression_0_30 = confusion_matrix_0_30[2,2] / (confusion_matrix_0_30[2,2] + confusion_matrix_0_30[1,1]) 
TPR_linear_regression_30_40 = confusion_matrix_30_40[2,2] / (confusion_matrix_30_40[2,2] + confusion_matrix_30_40[1,1])
TPR_linear_regression_40_50 = confusion_matrix_40_50[2,2] / (confusion_matrix_40_50[2,2] + confusion_matrix_40_50[1,1])
TPR_linear_regression_50_100 = confusion_matrix_50_100[2,2] / (confusion_matrix_50_100[2,2] + confusion_matrix_50_100[1,1])
TPR_linear_regression_0_30 # 0.06202209
TPR_linear_regression_30_40 # 0.02177343
TPR_linear_regression_40_50 # 0.02375182
TPR_linear_regression_50_100 # 0.04753086




####################################  Equalized Odds  #######################################
#TPR = TP/TP+FN
#FNR = FN/FN+TP

TPR_linear_regression_0_30 = confusion_matrix_0_30[2,2] / (confusion_matrix_0_30[2,2] + confusion_matrix_0_30[1,1]) 
TPR_linear_regression_30_40 = confusion_matrix_30_40[2,2] / (confusion_matrix_30_40[2,2] + confusion_matrix_30_40[1,1])
TPR_linear_regression_40_50 = confusion_matrix_40_50[2,2] / (confusion_matrix_40_50[2,2] + confusion_matrix_40_50[1,1])
TPR_linear_regression_50_100 = confusion_matrix_50_100[2,2] / (confusion_matrix_50_100[2,2] + confusion_matrix_50_100[1,1])
TPR_linear_regression_0_30 # 0.06202209
TPR_linear_regression_30_40 # 0.02177343
TPR_linear_regression_40_50 # 0.02375182
TPR_linear_regression_50_100 # 0.04753086

FNR_linear_regression_0_30 = confusion_matrix_0_30[1,1] / (confusion_matrix_0_30[2,2] + confusion_matrix_0_30[1,1]) 
FNR_linear_regression_30_40 = confusion_matrix_30_40[1,1] / (confusion_matrix_30_40[2,2] + confusion_matrix_30_40[1,1])
FNR_linear_regression_40_50 = confusion_matrix_40_50[1,1] / (confusion_matrix_40_50[2,2] + confusion_matrix_40_50[1,1])
FNR_linear_regression_50_100 = confusion_matrix_50_100[1,1] / (confusion_matrix_50_100[2,2] + confusion_matrix_50_100[1,1])
FNR_linear_regression_0_30 # 0.9379779
FNR_linear_regression_30_40 # 0.9782266
FNR_linear_regression_40_50 # 0.9762482
FNR_linear_regression_50_100 # 0.9524691




####################################  Fairness Through unawareness  #######################################
linear_regression_caret2 = train(biY ~ job + marital + education + balance + day + month + duration + pdays + previous + campaign + duration + campaign + biDefault + biHousing + biLoan , 
                                 data = train, 
                                 method = "glm", 
                                 trControl = train_Control, 
                                 tuneLength = 10)

linear_regression_caret2 # Accuracy: 0.8927011

# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret2, test)

confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277
accuracy_linear_regression = (confusion_matrix2[1,1] + confusion_matrix2[2,2]) / sum(confusion_matrix2)
accuracy_linear_regression # Accuracy: 0.8977108


prop.table(table(test$catAge))

summary(bank$catAge)
split_by_age = split(test, test$catAge)
test_0_30 = split_by_age$"<=30"
# View(test_0_30)

test_30_40 = split_by_age$"30-40"
# View(test_30_40)

test_40_50 = split_by_age$"40-50"
# View(test_40_50)

test_50_100 = split_by_age$">=50"
# View(test_50_100)


# Test on <= 30 subset
linear_regression_labels_0_30 = predict(linear_regression_caret2, test_0_30)
confusion_matrix_0_30 = table(linear_regression_labels_0_30, test_0_30$biY)
accuracy_linear_regression_0_30 = (confusion_matrix_0_30[1,1] + confusion_matrix_0_30[2,2]) / sum(confusion_matrix_0_30)
accuracy_linear_regression_0_30 # Accuracy:0.869469

# Test on 30-40 subset
linear_regression_labels_30_40 = predict(linear_regression_caret2, test_30_40)
confusion_matrix_30_40 = table(linear_regression_labels_30_40, test_30_40$biY)
accuracy_linear_regression_30_40 = (confusion_matrix_30_40[1,1] + confusion_matrix_30_40[2,2]) / sum(confusion_matrix_30_40)
accuracy_linear_regression_30_40 # Accuracy: 0.9049929

# Test on 40-50 subset
linear_regression_labels_40_50 = predict(linear_regression_caret2, test_40_50)
confusion_matrix_40_50 = table(linear_regression_labels_40_50, test_40_50$biY)
accuracy_linear_regression_40_50 = (confusion_matrix_40_50[1,1] + confusion_matrix_40_50[2,2]) / sum(confusion_matrix_40_50)
accuracy_linear_regression_40_50 # Accuracy:0.9201249

# Test on >= 50 subset
linear_regression_labels_50_100 = predict(linear_regression_caret2, test_50_100)
confusion_matrix_50_100 = table(linear_regression_labels_50_100, test_50_100$biY)
accuracy_linear_regression_50_100 = (confusion_matrix_50_100[1,1] + confusion_matrix_50_100[2,2]) / sum(confusion_matrix_50_100)
accuracy_linear_regression_50_100 # Accuracy:0.8731786




########################################################################################################
###########################train data using NAIVE BAYES MODEL ####################################
########################################################################################################
summary(test)
NB_caret = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + campaign + Fmonth + biDefault + biHousing + biLoan, 
                                data = train, 
                                method = "nb", 
                                trControl = train_Control, 
                                tuneLength = 10)
warnings()
NB_caret #Accuracy: 0.8927283


# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret, test)

# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277

accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Acurracy:0.8977108


# now let's separate based on gender to test fairness criteria
prop.table(table(test$marital))

summary(bank$marital)
split_by_marital = split(test, test$marital)
test_divorced = split_by_marital$divorced
# View(test_divorced)

test_married = split_by_marital$married
# View(test_married)

test_single = split_by_marital$single
# View(test_single)


# Test on divorced subset
linear_regression_labels_divorced = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced # Accuracy:0.9263682


# Test on married subset
linear_regression_labels_married = predict(linear_regression_caret, test_married)
confusion_matrix_married = table(linear_regression_labels_married, test_married$biY)
accuracy_linear_regression_married = (confusion_matrix_married[1,1] + confusion_matrix_married[2,2]) / sum(confusion_matrix_married)
accuracy_linear_regression_married # Accuracy: 0.9034763

# Test on single subset
linear_regression_labels_single = predict(linear_regression_caret, test_single)
confusion_matrix_single = table(linear_regression_labels_single, test_single$biY)
accuracy_linear_regression_single = (confusion_matrix_single[1,1] + confusion_matrix_single[2,2]) / sum(confusion_matrix_single)
accuracy_linear_regression_single # Accuracy: 0.8741149

####################################  Demographic Parity  #######################################
#positive rate = (TP + TN)
PR_linear_regression_divorced = confusion_matrix_divorced[2,2] + confusion_matrix_divorced[1,2]
PR_linear_regression_married = confusion_matrix_married[2,2] +confusion_matrix_married[1,2]
PR_linear_regression_single = confusion_matrix_single[2,2] +confusion_matrix_single[1,2]
PR_linear_regression_divorced # 85
PR_linear_regression_married # 561
PR_linear_regression_single # 350

####################################  Equalized Oppportunites  #######################################
#TPR = TP/TP+FN

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married # 0.02762996
TPR_linear_regression_single # 0.04860486

####################################  Equalized Odds  #######################################
#TPR = TP/TP+FN
#FNR = FN/FN+TP

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married #  0.02762996
TPR_linear_regression_single # 0.04860486


TPR_linear_regression_divorced = confusion_matrix_divorced[1,1] / (confusion_matrix_divorced[1,1] +confusion_matrix_divorced[2,2])
TPR_linear_regression_married = confusion_matrix_married[1,1] / (confusion_matrix_married[1,1] + confusion_matrix_married[2,2])
TPR_linear_regression_single = confusion_matrix_single[1,1] / (confusion_matrix_single[1,1] + confusion_matrix_single[2,2])
TPR_linear_regression_divorced # 0.9634801
TPR_linear_regression_married # 0.97237
TPR_linear_regression_single # 0.9513951


####################################  Fairness Through unawareness  #######################################
linear_regression_caret2 = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + duration + campaign + duration + campaign + biDefault + biHousing + biLoan , 
                                 data = train, 
                                 method = "glm", 
                                 trControl = train_Control, 
                                 tuneLength = 10)

linear_regression_caret2 # Accuracy: 0.8924794

confusion_matrix2 = table(linear_regression_labels, test$biY) 
confusion_matrix2

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277
accuracy_linear_regression2 = (confusion_matrix2[1,1] + confusion_matrix2[2,2]) / sum(confusion_matrix2)
accuracy_linear_regression2 # Accuracy: 0.8977108


prop.table(table(test$marital))

summary(bank$marital)
split_by_marital2 = split(test, test$marital)
test_divorced2 = split_by_marital2$divorced
# View(test_divorced2)

test_married2 = split_by_marital2$married
# View(test_married2)

test_single2 = split_by_marital2$single
# View(test_single2)


# Test on divorced subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_divorced2 = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced2 = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced2 = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced2 # Accuracy: 0.9263682







########################################################################################################
###########################train data using RANDOM FOREST MODEL ####################################
########################################################################################################
summary(test)
RF_caret = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + campaign + Fmonth + biDefault + biHousing + biLoan, 
                 data = train, 
                 method = "rf", 
                 trControl = train_Control, 
                 tuneLength = 10)
warnings()
RF_caret #Accuracy: 0.8927283


# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret, test)

# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277

accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Acurracy:0.8977108


# now let's separate based on gender to test fairness criteria
prop.table(table(test$marital))

summary(bank$marital)
split_by_marital = split(test, test$marital)
test_divorced = split_by_marital$divorced
# View(test_divorced)

test_married = split_by_marital$married
# View(test_married)

test_single = split_by_marital$single
# View(test_single)


# Test on divorced subset
linear_regression_labels_divorced = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced # Accuracy:0.9263682


# Test on married subset
linear_regression_labels_married = predict(linear_regression_caret, test_married)
confusion_matrix_married = table(linear_regression_labels_married, test_married$biY)
accuracy_linear_regression_married = (confusion_matrix_married[1,1] + confusion_matrix_married[2,2]) / sum(confusion_matrix_married)
accuracy_linear_regression_married # Accuracy: 0.9034763

# Test on single subset
linear_regression_labels_single = predict(linear_regression_caret, test_single)
confusion_matrix_single = table(linear_regression_labels_single, test_single$biY)
accuracy_linear_regression_single = (confusion_matrix_single[1,1] + confusion_matrix_single[2,2]) / sum(confusion_matrix_single)
accuracy_linear_regression_single # Accuracy: 0.8741149

####################################  Demographic Parity  #######################################
#positive rate = (TP + TN)
PR_linear_regression_divorced = confusion_matrix_divorced[2,2] + confusion_matrix_divorced[1,2]
PR_linear_regression_married = confusion_matrix_married[2,2] +confusion_matrix_married[1,2]
PR_linear_regression_single = confusion_matrix_single[2,2] +confusion_matrix_single[1,2]
PR_linear_regression_divorced # 85
PR_linear_regression_married # 561
PR_linear_regression_single # 350

####################################  Equalized Oppportunites  #######################################
#TPR = TP/TP+FN

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married # 0.02762996
TPR_linear_regression_single # 0.04860486

####################################  Equalized Odds  #######################################
#TPR = TP/TP+FN
#FNR = FN/FN+TP

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married #  0.02762996
TPR_linear_regression_single # 0.04860486


TPR_linear_regression_divorced = confusion_matrix_divorced[1,1] / (confusion_matrix_divorced[1,1] +confusion_matrix_divorced[2,2])
TPR_linear_regression_married = confusion_matrix_married[1,1] / (confusion_matrix_married[1,1] + confusion_matrix_married[2,2])
TPR_linear_regression_single = confusion_matrix_single[1,1] / (confusion_matrix_single[1,1] + confusion_matrix_single[2,2])
TPR_linear_regression_divorced # 0.9634801
TPR_linear_regression_married # 0.97237
TPR_linear_regression_single # 0.9513951


####################################  Fairness Through unawareness  #######################################
linear_regression_caret2 = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + duration + campaign + duration + campaign + biDefault + biHousing + biLoan , 
                                 data = train, 
                                 method = "glm", 
                                 trControl = train_Control, 
                                 tuneLength = 10)

linear_regression_caret2 # Accuracy: 0.8924794

confusion_matrix2 = table(linear_regression_labels, test$biY) 
confusion_matrix2

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277
accuracy_linear_regression2 = (confusion_matrix2[1,1] + confusion_matrix2[2,2]) / sum(confusion_matrix2)
accuracy_linear_regression2 # Accuracy: 0.8977108


prop.table(table(test$marital))

summary(bank$marital)
split_by_marital2 = split(test, test$marital)
test_divorced2 = split_by_marital2$divorced
# View(test_divorced2)

test_married2 = split_by_marital2$married
# View(test_married2)

test_single2 = split_by_marital2$single
# View(test_single2)


# Test on divorced subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_divorced2 = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced2 = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced2 = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced2 # Accuracy: 0.9263682





########################################################################################################
###########################train data using Support Vector Machine MODEL ####################################
########################################################################################################
summary(test)
SVM_caret = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + campaign + Fmonth + biDefault + biHousing + biLoan, 
                 data = train, 
                 method = "svmLinear2", 
                 trControl = train_Control, 
                 tuneLength = 10)
warnings()
SVM_caret #Accuracy: 0.8927283


# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels = predict(linear_regression_caret, test)

# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(linear_regression_labels, test$biY) 
confusion_matrix

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277

accuracy_linear_regression = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_linear_regression # Acurracy:0.8977108


# now let's separate based on gender to test fairness criteria
prop.table(table(test$marital))

summary(bank$marital)
split_by_marital = split(test, test$marital)
test_divorced = split_by_marital$divorced
# View(test_divorced)

test_married = split_by_marital$married
# View(test_married)

test_single = split_by_marital$single
# View(test_single)


# Test on divorced subset
linear_regression_labels_divorced = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced # Accuracy:0.9263682


# Test on married subset
linear_regression_labels_married = predict(linear_regression_caret, test_married)
confusion_matrix_married = table(linear_regression_labels_married, test_married$biY)
accuracy_linear_regression_married = (confusion_matrix_married[1,1] + confusion_matrix_married[2,2]) / sum(confusion_matrix_married)
accuracy_linear_regression_married # Accuracy: 0.9034763

# Test on single subset
linear_regression_labels_single = predict(linear_regression_caret, test_single)
confusion_matrix_single = table(linear_regression_labels_single, test_single$biY)
accuracy_linear_regression_single = (confusion_matrix_single[1,1] + confusion_matrix_single[2,2]) / sum(confusion_matrix_single)
accuracy_linear_regression_single # Accuracy: 0.8741149

####################################  Demographic Parity  #######################################
#positive rate = (TP + TN)
PR_linear_regression_divorced = confusion_matrix_divorced[2,2] + confusion_matrix_divorced[1,2]
PR_linear_regression_married = confusion_matrix_married[2,2] +confusion_matrix_married[1,2]
PR_linear_regression_single = confusion_matrix_single[2,2] +confusion_matrix_single[1,2]
PR_linear_regression_divorced # 85
PR_linear_regression_married # 561
PR_linear_regression_single # 350

####################################  Equalized Oppportunites  #######################################
#TPR = TP/TP+FN

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married # 0.02762996
TPR_linear_regression_single # 0.04860486

####################################  Equalized Odds  #######################################
#TPR = TP/TP+FN
#FNR = FN/FN+TP

TPR_linear_regression_divorced = confusion_matrix_divorced[2,2] / (confusion_matrix_divorced[2,2] +confusion_matrix_divorced[1,1])
TPR_linear_regression_married = confusion_matrix_married[2,2] / (confusion_matrix_married[2,2] + confusion_matrix_married[1,1])
TPR_linear_regression_single = confusion_matrix_single[2,2] / (confusion_matrix_single[2,2] + confusion_matrix_single[1,1])
TPR_linear_regression_divorced # 0.03651987
TPR_linear_regression_married #  0.02762996
TPR_linear_regression_single # 0.04860486


TPR_linear_regression_divorced = confusion_matrix_divorced[1,1] / (confusion_matrix_divorced[1,1] +confusion_matrix_divorced[2,2])
TPR_linear_regression_married = confusion_matrix_married[1,1] / (confusion_matrix_married[1,1] + confusion_matrix_married[2,2])
TPR_linear_regression_single = confusion_matrix_single[1,1] / (confusion_matrix_single[1,1] + confusion_matrix_single[2,2])
TPR_linear_regression_divorced # 0.9634801
TPR_linear_regression_married # 0.97237
TPR_linear_regression_single # 0.9513951


####################################  Fairness Through unawareness  #######################################
linear_regression_caret2 = train(biY ~ catAge + job + marital + education + balance + day + month + duration + pdays + previous + duration + campaign + duration + campaign + biDefault + biHousing + biLoan , 
                                 data = train, 
                                 method = "glm", 
                                 trControl = train_Control, 
                                 tuneLength = 10)

linear_regression_caret2 # Accuracy: 0.8924794

confusion_matrix2 = table(linear_regression_labels, test$biY) 
confusion_matrix2

# linear_regression_labels    0    1
#                         0 7762  719
#                         1  197  277
accuracy_linear_regression2 = (confusion_matrix2[1,1] + confusion_matrix2[2,2]) / sum(confusion_matrix2)
accuracy_linear_regression2 # Accuracy: 0.8977108


prop.table(table(test$marital))

summary(bank$marital)
split_by_marital2 = split(test, test$marital)
test_divorced2 = split_by_marital2$divorced
# View(test_divorced2)

test_married2 = split_by_marital2$married
# View(test_married2)

test_single2 = split_by_marital2$single
# View(test_single2)


# Test on divorced subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
linear_regression_labels_divorced2 = predict(linear_regression_caret, test_divorced)
confusion_matrix_divorced2 = table(linear_regression_labels_divorced, test_divorced$biY)
accuracy_linear_regression_divorced2 = (confusion_matrix_divorced[1,1] + confusion_matrix_divorced[2,2]) / sum(confusion_matrix_divorced)
accuracy_linear_regression_divorced2 # Accuracy: 0.9263682











