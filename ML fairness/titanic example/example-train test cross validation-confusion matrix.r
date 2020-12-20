####################################
####      TITANIC EXAMPLE       ####
####################################
# type CTRL+L
rm(list=ls())

# do the installs just once
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

# now reference all the libraries
library(class)
library(party)
library(e1071)
library(caret)

# working directory
setwd("/Users/xinyu/Desktop/senior fall/Independent Study/dataset/titanic example")

# load data, this will look for the file in the working directory
titanic <- read.csv("titanic.csv")

# check header
head(titanic)
# gender was correctly converted to factor variable; passenger class seems to not be converted though it should be

# summary stats
summary(titanic)

# check for missing values
sum(is.na(titanic$Fare)) #OK -- make sure to take care of the missing values before running classification prediction

# let's see proportion survived
prop.table(table(titanic$Survived))

# don't need the name for this exercise (one name per row, not useful as a predictor) so drop the column
titanic$Name = NULL

# ticket class should be a factor variable
summary(titanic$Pclass)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Survived = as.factor(titanic$Survived)

# TRAIN, TEST --> SPLIT (use TEST for fairness testing purposes)
set.seed(0)
# replace means sample with replacement (FALSE = no replacement)
split = sample( size = nrow(titanic), c("train", "test"),  replace = TRUE, prob = c(0.8, 0.2))

train = titanic[split=="train",] # use this for train control

test = titanic[split=="test",] # use this for fairness tests (holdout sample)

nrow(titanic)
nrow(train)
nrow(test) #nrow(train) + nrow(test) should add up to nrow(titanic)


# let's see other measures - is this a balanced dataset?
# let's see proportion survived
prop.table(table(titanic$Survived)) # should be 50-50 if balanced (50% survived, 50% perished -- not balanced)

# this implements cross-validation
set.seed(0) # this sets the random generator to the same origin
# k-fold cross validation with 10 folds;
train_Control = trainControl(method = "cv", number = 10)

# this trains the knn model using the cross-validation splits and automates the selection of train and test
# each run with a different value of k and reports back the best version of k; tuneLength is the number of
# different values of k (of course, k's are odd numbers)
#train$Survived = as.factor(train$Survived)
knn_caret = train(Survived~Age + Siblings.Spouses.Aboard + Fare + Parents.Children.Aboard,
                  data = train, method = "knn", trControl = train_Control,
                  tuneLength = 20)
summary(train)
knn_caret
# fun: let's see accuracy versus number of neighbors
plot(knn_caret)

## Fairness tests

# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
knn_labels = predict(knn_caret, test)
# here we are comparing the *predicted* outcomes to the actual
confusion_matrix = table(knn_labels, test$Survived) 
confusion_matrix

accuracy_knn = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_knn # this is the accuracy (0 - 1) 69% overall accuracy without separation on gender; pretty bad actually

# now let's separate based on gender to test fairness criteria
prop.table(table(test$Sex))

split_by_gender = split(test, test$Sex)
test_female = split_by_gender$female
View(test_female)
test_male = split_by_gender$male
View(test_male)

# Test on male subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
knn_labels_male = predict(knn_caret, test_male)
confusion_matrix_male = table(knn_labels_male, test_male$Survived)
accuracy_knn_male = (confusion_matrix_male[1,1] + confusion_matrix_male[2,2]) / sum(confusion_matrix_male)
accuracy_knn_male # this is the accuracy (0 - 1) 74%  accuracy for males

# Test on female subset
# Predict function will run the model from parameter 1 onto the test set (parameter 2) 
knn_labels_female = predict(knn_caret, test_female)
confusion_matrix_female = table(knn_labels_female, test_female$Survived)
accuracy_knn_female = (confusion_matrix_female[1,1] + confusion_matrix_female[2,2]) / sum(confusion_matrix_female)
accuracy_knn_female # this is the accuracy (0 - 1) 60%  accuracy for females (worse)

## Fails accuracy parity by gender

# Let's now test equality of opportunity (equal true positive rate)
#TPR=TP/(TP+FN)
TPR_knn_female = confusion_matrix_female[2,2] / (confusion_matrix_female[2,2]+confusion_matrix_female[1,2])
TPR_knn_male = confusion_matrix_male[2,2] / (confusion_matrix_male[2,2]+confusion_matrix_male[1,2])
TPR_knn_female
TPR_knn_male
# We could say these are close enough (within 5%) though not strictly equal

# Positive Predictive Value Parity
#PPV = TP / (TP+FP)
PPV_knn_female = confusion_matrix_female[2,2] / (confusion_matrix_female[2,2]+confusion_matrix_female[2,1])
PPV_knn_male = confusion_matrix_male[2,2] / (confusion_matrix_male[2,2]+confusion_matrix_male[2,1])
PPV_knn_female
PPV_knn_male

# this definitely fails 0.89 to 0.39

# Negative Predictive Value Parity
# NPV = TN / (TN+FN)
NPV_knn_female = confusion_matrix_female[1,1] / (confusion_matrix_female[1,1]+confusion_matrix_female[1,2])
NPV_knn_male = confusion_matrix_male[1,1] / (confusion_matrix_male[1,1]+confusion_matrix_male[1,2])
NPV_knn_female
NPV_knn_male

# also fails

# Let's check if Equalized Odds is successful; we already tested Equality of Opportunity which is one half of the test
#FPR = FP/(FP+TN) We must equalize false positive rate too
FPR_knn_female = confusion_matrix_female[2,1] / (confusion_matrix_female[2,1]+confusion_matrix_female[1,1])
FPR_knn_male = confusion_matrix_male[2,1] / (confusion_matrix_male[2,1]+confusion_matrix_male[1,1])
FPR_knn_female
FPR_knn_male
# very close to 5% apart. We'll consider it passed

##### DIFFERENT MODEL

# Note: cross-validation gives you more robust results in terms of accuracy but takes more computing time
library("e1071")
svm_linear_kernel = train(Survived~.,
                          data=titanic,
                          method="svmLinear2", trControl=train_Control,
                          tuneLength=10) # this is slow, you can increase the tune length parameter to > 10 but will take aw hile

# CARET still depends on the packages which contain the model definitions
# the radial kernel SVM is in kernlab
library("kernlab")
svm_Radial_kernel = train(Survived~.,
                          data=titanic,
                          method="svmRadial", trControl=train_Control,
                          tuneLength=10)
svm_Radial_kernel
# Even within the same algorithm, the choice of the kernel affects accuracy substantially. Curious to read more? For a short overview see https://data-flair.training/blogs/svm-kernel-functions/
  # For a detailed mathematical analysis, see Marsland's Machine Learning An Algorithmic Perspective, 2015   Edition, Chapter 8 - Support Vector Machines
# the best previous candidate was the decision tree - let's do cross-validation on decision trees 

library("party")
tree_caret = train(Survived~.,
                   data = titanic, method = "ctree2", trControl = train_Control,
                   tuneLength = 25)
tree_caret
plot(tree_caret)

## Fairness test demos
# this is akin to the probabilities + predicted_class above which generated a survived/not survived prediction
# notice syntax different as SVM is not a probabilistic classifier like the logistic classifier
svm_labels = predict(svm_model, test)

head(svm_labels)
# confusion matrix is generated by comparing the predicted to the actual from the test dataset
confusion_matrix = table(svm_labels, test$Survived)
confusion_matrix

accuracy_svm = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_svm # better!



# let's say you wanted to run this on new data (for now, let's re-run it on the test set we created before as "new" data to see how to use the trained model)
cv_tree_labels = predict(tree_caret, test_Married)
confusion_matrix = table(cv_tree_labels, test$Survived)
accuracy_tree = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_tree

# REFERENCE MATERIAL: https://topepo.github.io/caret/available-models.html (lists all models in caret) and https://cran.r-project.org/web/packages/caret/caret.pdf