data <- read.csv("/Users/xinyu/Desktop/dataset/ml communitiesandcrime dataset/communities.csv")
head(data)
summary(data)
nrow(data)

### clean data set
###add column name into file

###build logistic regression model
#explore crimes ~ race 
ols_res1 <- lm(ViolentCrimesPerPop ~ racePctWhite + racePctAsian + racePctHisp + racepctblack, data)
summary(ols_res1)

#explore crimes ~ number of people(higher population causes higher crimes)
ols_res2 <- lm(ViolentCrimesPerPop ~ population + householdsize, data)
summary(ols_res2)

#explore crimes ~ age(most of crimes happens at the age of 12-29 and 65+, more likely 21-29)
ols_res3 <- lm(ViolentCrimesPerPop ~ agePct12t21 + agePct12t29 + agePct16t24 + agePct65up, data)
summary(ols_res3)

#explore crimes ~ medIncome(medIncome has negative influence on crimes)
ols_res4 <- lm(ViolentCrimesPerPop ~ medIncome, data)
summary(ols_res4)

#explore crimes ~ household situation(large percentage of households with retirement income cause higher violence rate)
ols_res5 <- lm(ViolentCrimesPerPop ~ pctWWage + pctWFarmSelf + pctWInvInc + pctWSocSec + pctWPubAsst + pctWRetire, data)
summary(ols_res5)

#explore crimes ~ household situation(more number of poverty people causes higher crimes)
ols_res6 <- lm(ViolentCrimesPerPop ~ NumUnderPov, data)
summary(ols_res6)

#explore crimes ~ education level(people at 25 or more has the most crime rate if they are between 9th grade and high 
#school. People with Bachelor or more cause a positive influence on crime rate)
ols_res7 <- lm(ViolentCrimesPerPop ~ PctLess9thGrade + PctNotHSGrad + PctBSorMore, data)
summary(ols_res7)

#explore crimes ~ job area()
ols_res8 <- lm(ViolentCrimesPerPop ~ PctUnemployed + PctEmploy + PctEmplManu + PctEmplProfServ + PctOccupManu + PctOccupMgmtProf, data)
summary(ols_res8)

#explore crimes ~ male marital situation(man who does not have partner causes higher rate of crime)
ols_res9 <- lm(ViolentCrimesPerPop ~ MalePctDivorce + MalePctNevMarr, data)
summary(ols_res9)

#explore crimes ~ female marital situation(women who is divorced causes higher rate of crime)
ols_res10 <- lm(ViolentCrimesPerPop ~ FemalePctDiv, data)
summary(ols_res10)

#explore crimes ~ immigrant(Longer immigrant time cause higher crime rate)
ols_res11 <- lm(ViolentCrimesPerPop ~ PctImmigRecent + PctImmigRec5 + PctImmigRec8 + PctImmigRec10 + PctRecentImmig + PctRecImmig5 + PctRecImmig8 + PctRecImmig10, data)
summary(ols_res11)

#explore crimes ~ language(people who speak English only has a positive influence on crime rate)
ols_res12 <- lm(ViolentCrimesPerPop ~ PctSpeakEnglOnly + PctNotSpeakEnglWell, data)
summary(ols_res12)

#explore crimes ~ household
ols_res13 <- lm(ViolentCrimesPerPop ~ PctLargHouseFam + PctLargHouseOccup + PersPerOccupHous + PersPerOwnOccHous + PersPerRentOccHous + PctPersOwnOccup + PctPersDenseHous + PctHousLess3BR + MedNumBR + HousVacant +  PctHousOccup + PctHousOwnOcc + PctVacantBoarded + PctVacMore6Mos, data)
summary(ols_res13)

#explore crimes ~ police(higher crime higher police)
ols_res14 <- lm(ViolentCrimesPerPop ~  LemasSwornFT +  LemasSwFTPerPop + LemasSwFTFieldOps +LemasSwFTFieldPerPop + LemasTotalReq + LemasTotReqPerPop + PolicReqPerOffic + PolicPerPop + PolicCars + PolicOperBudg, data)
summary(ols_res14)



library(mfx)
logitmfx(numy ~ ., data)
















