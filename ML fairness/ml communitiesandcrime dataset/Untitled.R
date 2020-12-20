data <- read.csv("/Users/xinyu/Desktop/dataset/ml communitiesandcrime dataset/communities.csv")
head(data)
summary(data)
nrow(data)

### clean data set
###add column name into file

###build logistic regression model
logit_res <- glm(ViolentCrimesPerPop ~ ., data, family = 'binomial')
summary(logit_res)

library(mfx)
logitmfx(numy ~ ., data)
















