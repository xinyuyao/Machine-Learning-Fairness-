#https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

setwd("/Users/xinyu/Desktop/senior fall/Independent Study/dataset/ml bank dataset")
data <- read.csv("bank/bank-full.csv")
summary(data)
nrow(data)

#####draw the distribution of data

hist(data$age)

barplot(summary(data$job))
       
barplot(summary(data$marital))

barplot(summary(data$education))

barplot(summary(data$default))

hist(data$balance)

barplot(summary(data$housing))

barplot(summary(data$loan))

barplot(summary(data$contact))

hist(data$day)

data$Fmonth <- factor(data$month,levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul","aug", "sept", "oct", "nov", "dec"))
barplot(summary(data$Fmonth))

hist(data$duration)

hist(data$campaign)

hist(data$pdays)

hist(data$previous)

barplot(summary(data$poutcome))

barplot(summary(data$y))



#####dealing with missing value by deleting the row if the row has equal to or more than two missing value. Otherwise, replce the missing value with medium value

#change the value of "unknown" to "NA"

d1 <- data
summary(d1)
nrow(d1)

#0/45211 comlums of NA's in age
nrow(d1[d1$age == "unknown",])


#288/45211 comlums of NA's in job
nrow(d1[d1$job == "unknown",])
d1$job[d1$job == "unknown"] <- "NA"
summary(d1)

#0/45211 columns of NA's in marital
nrow(d1[d1$martial == "unknown",])
d1$marital[data$marital == "unknown"] <- "NA"
summary(d1)

#1857/45211 columns of NA's in education
nrow(d1[d1$education == "unknown",])
d1$education[data$education == "unknown"] <- "NA"
summary(d1)

#0/45211 columns of NA's in default
nrow(d1[d1$default == "unknown",])
d1$default[data$default == "unknown"] <- "NA"
summary(d1)

#0/45211 columns of NA's in balance
nrow(d1[d1$balance == "unknown",])


#0/45211 columns of NA's in housing
nrow(d1[d1$housing == "unknown",])

#0/45211 columns of NA's in loan
nrow(d1[d1$loan == "unknown",])

#13020/45211 columns of NA's in contact
nrow(d1[d1$contact == "unknown",])
d1$contact <- NULL
summary(d1)

#0/45211 columns of NA's in day
nrow(d1[d1$day == "unknown",])

#0/45211 columns of NA's in month
nrow(d1[d1$month == "unknown",])



#0/45211 columns of NA's in duration
nrow(d1[d1$duration == "unknown",])


#0/45211 columns of NA's in campaign
nrow(d1[d1$campaign == "unknown",])

#0/45211 comlums of NA's in pdays
nrow(d1[d1$pdays == "unknown",])

#288/45211 comlums of NA's in previous
nrow(d1[d1$previous == "unknown",])


#36959/45211 columns of NA's in poutcome
nrow(d1[d1$poutcome == "unknown",])
d1$poutcome <- NULL
summary(d1)

#0/45211 columns of NA's in y
nrow(d1[d1$y == "unknown",])

#0/45211 columns of NA's in Fmonth
nrow(d1[d1$Fmonth == "unknown",])
d1$Fmonth[data$Fmonth == "unknown"] <- "NA"
summary(d1$Fmonth)



#####delete the row if the row has more than one missing value
summary(d1)
df1<- as.data.frame(d1)
cnt_na <- apply(df1, 1, function(d1) sum(is.na(d1)))
cnt_na

#remove 45211 - 45084 = 127 rows
d1 <- as.data.frame(df1[cnt_na < 2,])
nrow(d1)
summary(d1)
head(d1)


#replace rest missing value with median(numerical value), replacing missing value with max(categorical value)

#replace 9505 + (228) job missing value with "blue-collar"
nrow(d1$job[is.na(d1$job)])
d1$job[is.na(d1$job)] <- "blue-collar"
summary(d1)

#replace 23202 + (1857) education missing value with "secondary"
nrow(d1$education[is.na(d1$education)])
d1$education[is.na(d1$education)] <- "secondary"
summary(d1)

#replace 13766 + (579) education missing value with "secondary"
nrow(d1$Fmonth[is.na(d1$Fmonth)])
d1$Fmonth[is.na(d1$Fmonth)] <- "may"
summary(d1$Fmonth)


####observe distribution after cleaning the dataset(dealing with missing value)
# log-tranform balance, previous prvious a large amount of NAN, therefore I didn't transform this two attributes


###log-transform data

#log-transform duration
hist(d1$duration)
summary(d1$duration)
d1$log_duration <- log(d1$duration)
hist(d1$log_duration)
summary(d1$log_duration)

#log-transform campaign
hist(d1$campaign)
summary(d1$campaign)
d1$log_campaign<- log(d1$campaign)
hist(d1$log_campaign)
summary(d1$log_campaign)

summary(d1)



######set categorical value to numerical value/binary value


######bank client data

###categorize AGE
class(d1$age)
d1$catAge <- cut(d1$age, breaks=c(0, 30, 40, 50, 100), labels=c("<=30","30-40","40-50",">=50"))
summary(d1$catAge)


###change marital into binary value
summary(d1$marital)
d1$biMarital <- ifelse(d1$marital == "married", 1, 0)
summary(d1$biMarital)

###change default(credit) into binary value
summary(d1$default)
d1$biDefault <- ifelse(d1$default == "yes", 1, 0)
summary(d1$biDefault)

###change house into binary value
summary(d1$housing)
d1$biHousing <- ifelse(d1$housing == "yes", 1, 0)
summary(d1$housing)

###change loan(personal) into binary value
summary(d1$loan)
d1$biLoan <- ifelse(d1$loan == "yes", 1, 0)
summary(d1$biLoan)

###change house into binary value
summary(d1$housing)
d1$biHousing <- ifelse(d1$housing == "yes", 1, 0)
summary(d1$biHousing)


######last contact of the current campaign

######other attributes
summary(d1$y)
d1$biY <- ifelse(d1$y == "yes", 1, 0)
summary(d1$biY)

write.csv(d1, file="bankfullclean.csv")
# 
# ####build regression model
# library(mfx)
# #age
# logit_res1 <- glm(biY ~ catAge, data, family = 'binomial')
# summary(logit_res1)
# logitmfx(biY ~ catAge, data)
# 
# #job
# logit_res2 <- glm(biY ~ job, data, family = 'binomial')
# summary(logit_res2)
# logitmfx(biY ~ job, data)
# #education
# logit_res3 <- glm(biY ~ education, data, family = 'binomial')
# summary(logit_res3)
# logitmfx(biY ~ education, data)
# 
# #loan
# logit_res4 <- glm(biY ~ biDefault + biHousing + biLoan, data, family = 'binomial')
# summary(logit_res4)
# logitmfx(biY ~ biDefault + biHousing + biLoan, data)
# 
# 
# #previous compaign
# logit_res5 <- glm(biY ~ contact + month + day + duration + log_campaign + log_pdays + previous, data, family = 'binomial')
# summary(logit_res5)
# logitmfx(biY ~ contact + month + day + duration + log_campaign + log_pdays + previous, data)
# 
# 
# 
