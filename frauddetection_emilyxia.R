rm(list=ls())

library(caret)
library(data.table)
library(randomForest)
library(zoo)
library(ROCR)
library(DMwR)
library(ggplot2)

df <-fread('/Users/lulu/Downloads/creditcard.csv')

#data exploratory analysis and preprocessing
summary(df)
table(df$Class)  #492 positives out of 284807 (0.17%)
sum(is.na(df))   #no NA
sapply(df,var) 

#check each variable
ggplot(df,aes(x=V1)) + geom_density(aes(group=Class, colour= Class, fill=Class, alpha=0.3))

#remove variables with no predictive power
df$Time <- NULL

set.seed(100)

df$Class <- as.factor(df$Class)
index <- createDataPartition(df$Class, p=0.7,list=FALSE)
training <- df[index,]
test <- df[-index,]

#Logistic regression
logit <- glm(Class ~ ., data = training, family = 'binomial')
logit_pred <- predict(logit, test, type = 'response')

logit_prediction <- prediction(logit_pred, test$Class)
logit_recall <- performance(logit_prediction,'prec','rec')
logit_roc <- performance(logit_prediction,'tpr','fpr')
logit_auc <- performance(logit_prediction,'auc')

#Random Forest
rf.model <- randomForest(Class ~ ., data = training, ntree = 2000, nodesize = 20)
rf_pred <- predict(rf.model, test, type = 'prob')

rf_prediction <- prediction(rf_pred[,2], test$Class)
rf_recall <- performance(rf_prediction,'prec','rec')
rf_roc <- performance(rf_prediction,'tpr','fpr')
rf_auc <- performance(rf_prediction,'auc')

#Downsampling with SMOTE package
train_smote <- SMOTE(Class ~ ., training, perc.over = 3000, perc.under=100)
table(train_smote$Class)

ctrl <- trainControl(method ='cv', number = 10)

tb_model <- train(Class ~ ., data = train_smote, method = 'treebag', trControl = ctrl)

tb_pred <- predict(tb_model$finalModel, test, type = 'prob')

tb_prediction <- prediction(tb_pred[,2], test$Class)
tb_recall <- performance(tb_prediction,'prec','rec')
tb_roc <- performance(tb_prediction,'tpr','fpr')
tb_auc <- performance(tb_prediction,'auc')

plot(logit_recall, col='red')
plot(rf_recall, add = TRUE, col = 'blue')
plot(tb_recall, add = TRUE, col = 'green')

#create a function to calculate area under precision-recall curve
auprc <- function(pr_curve){
  x <- as.numeric(unlist(pr_curve@x.values))
  y <- as.numeric(unlist(pr_curve@y.values))
  y[is.nan(y)] <- 1
  id <- order(x)
  result <- sum(diff(x[id])*rollmean(y[id],2))
  return(result)
}

#compare the models with this metric
auprc(logit_recall)
auprc(rf_recall)
auprc(tb_recall)
