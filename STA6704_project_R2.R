## import the original dataset
setwd("E:/Desk/2019/STA6704-data mining II/proj")
rm(list=ls())
############################
#data2= read.table("parkinsons_updrs.data.txt",header=TRUE,sep=",",dec=".")
data2= read.table(file.choose(),header=TRUE,sep=",",dec=".")
dim(data2)
str(data2)
data2_sta=summary(data2)
#partion dataset into traininng and validation sets
set.seed(10)
train.index=sample(c(1:dim(data2)[1]),0.6*dim(data2)[1])
traind=data2[train.index,]
validd=data2[-train.index,]
names(traind)
names(validd)

#variable selection
##############################
library(glmnet)
data2.fit=lm(total_UPDRS~.,data=traind)
## forward
forward.fit=step(data2.fit,direction = "forward")
summary(forward.fit)   #AIC=8241.77  15 variables
##backward
backward.fit=step(data2.fit,direction = "backward")
summary(backward.fit)  #AIC=8232.99  15 variables
##both
both.fit=step(data2.fit,direction = "both")
summary(both.fit)  #AIC=8232.99   15 variables
library(dplyr)
data2=select(data2,-test_time,-Jitter_RAP,-Shimmer_dB,-Shimmer_APQ3,-Shimmer_DDA,-NHR)
dim(data2)

names(data2)
str(data2)
cor(data2)
# SVR with  kernels
##############################
library(e1071)

## SVR with linear kernel
##############################
data.svm =svm(total_UPDRS~.,data=traind,kernel="linear", shrinking=TRUE,cross=5)
summary(data.svm)
data.svmpred=predict(data.svm,validd) 
data.svmpred1=predict(data.svm,traind)        #without stepwise         with stepwise
sqrt(mean((data.svmpred-validd$total_UPDRS)^2))  #3.277364 test error 3.280485
sqrt(mean((data.svmpred1-traind$total_UPDRS)^2))  #3.26323 training error 3.261565
#Find value of W (slope)
W = t(data.svm$coefs) %*% data.svm$SV
write.csv(W,'E:/Desk/2019/STA6704-data mining II/proj/svr_slope_stepwise.csv')
#Find value of b (intercept)
b = data.svm$rho #0.04325446   0.04160305

# perform a grid search by tune()
##############################
tuneResult <- tune(svm,total_UPDRS~., data =traind,kernel="linear",
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
print(tuneResult)
plot(tuneResult)
## error-epsilon(0,0.1,0.2,...,1),in total 11 
## penalty-cost(2^2,2^3,...,2^9), in total 8 
## model number=11*8=88
#Parameter tuning of 'svm':  
#  - sampling method: 10-fold cross validation 
#- best parameters:
#epsilon cost
#0.2    16
#- best performance: 10.62802 , RMSE=sqrt(10.62802)=3.260064
# Draw the tuning graph
plot(tuneResult)
bestml=tuneResult$best.model
summary(bestml)
#Parameters:
#   SVM-Type:  eps-regression 
# SVM-Kernel:  linear 
#       cost:  16 
#      gamma:  0.04761905 
#    epsilon:  0.2 
#Number of Support Vectors:  1390
tunepre.t=predict(bestml,traind)
tunepre.v=predict(bestml,validd)
sqrt(mean(tunepre.t-traind$total_UPDRS)^2)  #0.375698
sqrt(mean(tunepre.v-validd$total_UPDRS)^2)  #0.3280411

## kernels: polynoimal,Gaussian(radial),signmoid
##radial kernel
##################################
data.svm.r =svm(total_UPDRS~.,data=traind,kernel="radial",
              shrinking=TRUE,cross=5)
summary(data.svm.r)
data.svm.rpred=predict(data.svm.r,validd)
data.svm.rpred1=predict(data.svm.r,traind)
sqrt(mean((data.svm.rpred-validd$total_UPDRS)^2))  #1.92791 test error 
sqrt(mean((data.svm.rpred1-traind$total_UPDRS)^2))  #1.619339 training error 

#Find value of W (slope)
W1 = t(data.svm.r$coefs) %*% data.svm.r$SV
write.csv(W1,'E:/Desk/2019/STA6704-data mining II/proj/svr_radial.csv')
#Find value of b (intercept)
b1 = data.svm.r$rho #0.001403138   
## tune with radial kernel
tuneResult <- tune(svm,total_UPDRS~., data =traind,kernel="radial",
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
print(tuneResult)
##Parameter tuning of 'svm':
#Parameter tuning of 'svm':
  
#  - sampling method: 10-fold cross validation 

#- best parameters:
#epsilon cost
#0    128
#- best performance: 1.528463  , RMSE=sqrt(1.528463 )=1.23631
# Draw the tuning graph
plot(tuneResult.r)
bestmr=tuneResult.r$best.model
summary(bestmr)
#Parameters:
#   SVM-Type:  eps-regression 
# SVM-Kernel:  radial 
#       cost:  128 
#      gamma:  0.04761905 
#    epsilon:  0 
#Number of Support Vectors:  3525
tunepre.rt=predict(bestmr,traind)
tunepre.rv=predict(bestmr,validd)
sqrt(mean(tunepre.rt-traind$total_UPDRS)^2) #0.01235608
sqrt(mean(tunepre.rv-validd$total_UPDRS)^2) #0.05400357


##polynomial kernel
data.svm.p =svm(total_UPDRS~.,data=traind,kernel="polynomial",
                shrinking=TRUE,cross=5)
summary(data.svm.p)
data.svm.ppred=predict(data.svm.p,validd)
data.svm.ppred1=predict(data.svm.p,traind)
sqrt(mean((data.svm.ppred-validd$total_UPDRS)^2))  #11.84785 test error 
sqrt(mean((data.svm.ppred1-traind$total_UPDRS)^2))  #3.211809 training error 
#Find value of W (slope)
W2 = t(data.svm.p$coefs) %*% data.svm.p$SV
write.csv(W2,'E:/Desk/2019/STA6704-data mining II/proj/svr_polynomial.csv')
#Find value of b (intercept)
b2 = data.svm.p$rho #
## tune with polynomial kernel
tuneResult <- tune(svm,total_UPDRS~., data =traind,kernel="polynomial",
                   ranges = list(epsilon = seq(0,0.5,0.1), cost = 2^(2:6)))

print(tuneResult)
#Parameter tuning of ‘svm’:
#sampling method: 10-fold cross validation 
#best parameters:
# epsilon cost
#     0.3   32
# best performance: 57.50688 
plot(tuneResult)
## error-epsilon(0,0.1,0.2,...,1),in total 6 
## penalty-cost(2^2,2^3,...,2^9), in total 5 
## model number=5*6=30
tunedModel <- tuneResult$best.model
tunedModelv=predict(tunedModel,validd)
tunedModelt <- predict(tunedModel, traind) 
sqrt(mean((tunedModelv-validd$total_UPDRS)^2))  #34.06732
sqrt(mean((tunedModelt-traind$total_UPDRS)^2))  #2.349622

##sigmoid kernel
data.svm.s =svm(total_UPDRS~.,data=traind,kernel="sigmoid",
                shrinking=TRUE,cross=5)
summary(data.svm.s)
data.svm.spred=predict(data.svm.s,validd)
data.svm.spred1=predict(data.svm.s,traind)
sqrt(mean((data.svm.spred-validd$total_UPDRS)^2))  #1155.724 test error 
sqrt(mean((data.svm.spred1-traind$total_UPDRS)^2))  #1148.464 training error 
#Find value of W (slope)
W3 = t(data.svm.s$coefs) %*% data.svm.s$SV
write.csv(W3,'E:/Desk/2019/STA6704-data mining II/proj/svr_sigmoid.csv')
#Find value of b (intercept)
b3 = data.svm.s$rho #
## tune with polynomial kernel
tuneResult1 <- tune(svm,total_UPDRS~., data =traind,kernel="sigmoid",
                   ranges = list(epsilon = seq(0,0.5,0.1), cost = 2^(2:6)))

print(tuneResult1)
#Parameter tuning of ‘svm’:
#sampling method: 10-fold cross validation 
#best parameters:
# epsilon cost
#     0.5    4
# best performance: 17203108  
plot(tuneResult1)
## error-epsilon(0,0.1,0.2,...,1),in total 6 
## penalty-cost(2^2,2^3,...,2^9), in total 5 
## model number=5*6=30
tunedModel <- tuneResult1$best.model
tunedModelv=predict(tunedModel,validd)
tunedModelt <- predict(tunedModel, traind) 
sqrt(mean((tunedModelv-validd$total_UPDRS)^2))  #4620.818
sqrt(mean((tunedModelt-traind$total_UPDRS)^2))  #4591.126



## ensemble tree by boosting with stepwise
###########################################
set.seed(10)
train.index=sample(c(1:dim(data2)[1]),0.6*dim(data2)[1])
train_df=data2[train.index,]
valid_df=data2[-train.index,]
library(dplyr)
x_train= train_df[,c(1:4,6:16)]
y_train= train_df[,5]
x_test=  valid_df[,c(1:4,6:16)]
y_test=  valid_df[,5]

names(data2)
str(x_train)
niter <- 400  # literation number
loss <- function(y,yhat){0.5*(y - yhat)^2} # mean loss function
#################
library(rpart)
v=0.05# learning rate/shrinkage parameter
#train_df=Boston[train,]
fit=rpart(total_UPDRS~.,data=train_df)
train_yp=predict(fit,x_train)       # y of training set
test_yp=predict(fit,x_test)         # y of test set
train_df$yr=train_df$total_UPDRS - v*train_yp # 
train_YP=v*train_yp
test_YP=v*test_yp
train_errors=rep(0,niter )
test_errors=rep(0,niter )
for(i in seq(niter))  {
  fit=rpart(yr~subject+age+sex+motor_UPDRS+Jitter+
              Jitter_Abs+Jitter_PPQ5+Jitter_DDP+Shimmer+
              Shimmer_APQ5+Shimmer_APQ11+
              +HNR+RPDE+DFA+PPE,data=train_df,
            control = list(minsplit = 10, maxdepth = 12, xval = 10))
#  fit=rpart(yr~subject+age+sex++test_time+motor_UPDRS+Jitter+
#              Jitter_Abs+Jitter_RAP+Jitter_PPQ5+Jitter_DDP+Shimmer+
#              Shimmer_dB+Shimmer_APQ3+Shimmer_APQ5+Shimmer_APQ11+
#              Shimmer_DDA+NHR+HNR+RPDE+DFA+PPE,data=train_df)
  train_yp=predict(fit,x_train)
  test_yp=predict(fit,x_test)
  train_df$yr=train_df$yr - v*train_yp
  train_YP=cbind(train_YP,v*train_yp)
  test_YP=cbind(test_YP,v*test_yp)
  ytrain_hat=apply(train_YP,1,sum)
  ytest_hat=apply(test_YP,1,sum)
  train_error = mean(loss(y_train,ytrain_hat))
  train_errors[i] <- train_error
  test_errors[i] <- mean(loss(y_test,ytest_hat))## without stepwise,  with stepwise
  cat(i,"error:",train_error,"\n")    # 200           ,0.214406       0.4596417 
  cat(i,"error:",test_errors[i],"\n")  # 200 iteration, 0.2700032     0.550301 
  print(i)
}
par(mfrow=c(1,2))
#plot(errors,main="Training errors of Decision Tree Boosting")
plot(seq(1,niter ),test_errors,type="l",xlim=c(0,400),ylim=c(0,500),ylab="Error
Rate",xlab="Iterations",lwd=2, main='Errors of Decision Tree
Boosting')
lines(train_errors,lwd=2,col="purple")
legend(200,500,c("Training Error","Test Error"),
       col=c("purple","black"),lwd=2)

## ensemble tree by boosting without stepwise
###########################################
data2= read.table(file.choose(),header=TRUE,sep=",",dec=".")
dim(data2)
str(data2)
data2_sta=summary(data2)
set.seed(10)
train.index=sample(c(1:dim(data2)[1]),0.6*dim(data2)[1])
train_df=data2[train.index,]
valid_df=data2[-train.index,]
library(dplyr)
x_train= train_df[,c(1:5,7:22)]
y_train= train_df[,6]
x_test=  valid_df[,c(1:5,7:22)]
y_test=  valid_df[,6]

names(data2)
str(x_train)
niter <- 400  # literation number
loss <- function(y,yhat){0.5*(y - yhat)^2} # mean loss function
#################
library(rpart)
v=0.05# learning rate/shrinkage parameter
#train_df=Boston[train,]
fit=rpart(total_UPDRS~.,data=train_df)
train_yp=predict(fit,x_train)       # y of training set
test_yp=predict(fit,x_test)         # y of test set
train_df$yr=train_df$total_UPDRS - v*train_yp # 
train_YP=v*train_yp
test_YP=v*test_yp
train_errors=rep(0,niter )
test_errors=rep(0,niter )
for(i in seq(niter))  {
        fit=rpart(yr~subject+age+sex++test_time+motor_UPDRS+Jitter+
                Jitter_Abs+Jitter_RAP+Jitter_PPQ5+Jitter_DDP+Shimmer+
                Shimmer_dB+Shimmer_APQ3+Shimmer_APQ5+Shimmer_APQ11+
                Shimmer_DDA+NHR+HNR+RPDE+DFA+PPE,data=train_df,
      control = list(minsplit = 10, maxdepth = 12, xval = 10))  ## tune by control()
  train_yp=predict(fit,x_train)
  test_yp=predict(fit,x_test)
  train_df$yr=train_df$yr - v*train_yp
  train_YP=cbind(train_YP,v*train_yp)
  test_YP=cbind(test_YP,v*test_yp)
  ytrain_hat=apply(train_YP,1,sum)
  ytest_hat=apply(test_YP,1,sum)
  train_error = mean(loss(y_train,ytrain_hat))
  train_errors[i] <- train_error
  test_errors[i] <- mean(loss(y_test,ytest_hat))## without stepwise,  with stepwise
  cat(i,"error:",train_error,"\n")    # 200           ,0.214406       0.4596417 
  cat(i,"error:",test_errors[i],"\n")  # 200 iteration, 0.2700032     0.550301 
  print(i)
}
par(mfrow=c(1,2))
#plot(errors,main="Training errors of Decision Tree Boosting")
plot(seq(1,niter ),test_errors,type="l",xlim=c(0,400),ylim=c(0,500),ylab="Error
     Rate",xlab="Iterations",lwd=2, main='Errors of Decision Tree
     Boosting')
lines(train_errors,lwd=2,col="purple")
legend(200,500,c("Training Error","Test Error"),
       col=c("purple","black"),lwd=2)

## gradient boosted tree  tuned by train()
######################################################
set.seed(10)
train.index=sample(c(1:dim(data2)[1]),0.6*dim(data2)[1])
train_df=data2[train.index,]
valid_df=data2[-train.index,]
library(dplyr)
x_train= train_df[,c(1:5,7:22)]
y_train= train_df[,6]
x_test=  valid_df[,c(1:5,7:22)]
y_test=  valid_df[,6]

## tune by train()
install.packages("tidyr")
install.packages("caret")
library(caret)
library(kernlab)
fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated 5 times
  repeats = 5)
## set a search grid
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (5:15)*100, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid) # 33 times

## tuning and using gradient boosting
set.seed(825)
gbmFit2 <- train(total_UPDRS~ ., data = train_df, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
gbmFit2
##plot depth vs iterations
trellis.par.set(caretTheme())
plot(gbmFit2) 

# get final best model
tunetreet=predict(gbmFit2,train_df)
tunetreev=predict(gbmFit2,valid_df)
sqrt(mean(tunetreet-train_df$total_UPDRS)^2)  #0.0003068737
sqrt(mean(tunetreev-valid_df$total_UPDRS)^2)  #0.00738085


####################
## random forest
###################
library(randomForest)
rfNews()
bagf=randomForest(total_UPDRS~.,data=train_df,mtry=21,improtance=TRUE)
bagf
#Type of random forest: regression
#Number of trees: 500
#No. of variables tried at each split: 21
#Mean of squared residuals: 0.167416
#% Var explained: 99.85
bagp=predict(bagf,newdata=valid_df)
bagp1=predict(bagf,newdata=train_df)
plot(bagp,valid_df$total_UPDRS)
abline(0,1)
sqrt(mean(bagp-valid_df$total_UPDRS)^2)    # 0.02727457
sqrt(mean(bagp1-train_df$total_UPDRS)^2)   # 0.001778002

importance(bagf)
# IncNodePurity
#subject         14125.11986
#age             11152.06323
#sex              1575.55684
#test_time        4935.00701
#motor_UPDRS    360388.78309
#Jitter             38.06340
#Jitter_Abs        455.29347
#Jitter_RAP         33.61760
#Jitter_PPQ5        53.50230
#Jitter_DDP         29.19413
#Shimmer            70.23246
#Shimmer_dB         56.96813
#Shimmer_APQ3      169.44403
#Shimmer_APQ5      259.24985
#Shimmer_APQ11    1249.76843
#Shimmer_DDA       148.26916
#NHR                63.77070
#HNR               305.17417
#RPDE              137.76243
#DFA              1220.70189
#PPE                96.15073
varImpPlot(bagf)
###tune the random forest by changing mtry from 1:21
trainE=rep(0,21)
validE=rep(0,21)
for (i in 1:21) 
  {
bagf=randomForest(total_UPDRS~.,data=train_df,mtry=i,improtance=TRUE)
bagp=predict(bagf,newdata=valid_df)
bagp1=predict(bagf,newdata=train_df)
validE[i]=sqrt(mean(bagp-valid_df$total_UPDRS)^2)    
trainE[i]=sqrt(mean(bagp1-train_df$total_UPDRS)^2)  
print(i)
print(trainE[i])
print(validE[i])
}

plot(seq(1,21),validE,type="l",xlim=c(0,21),ylim=c(0,0.15),ylab="test RMSE",
     xlab="number of variables",lwd=2, main='Errors of RandomForest')
lines(trainE,lwd=2,col="purple")
legend(13,0.15,c("Training Error","Test Error"),
       col=c("purple","black"),lwd=2)
##optimal mtry=10 random forest select 10 variables with the best outcome
##test RMSE 0.012201519
# training     RMSE 0.006184478
bagf=randomForest(total_UPDRS~.,data=train_df,mtry=10,improtance=TRUE)
bagp=predict(bagf,newdata=valid_df)
bagp1=predict(bagf,newdata=train_df)
sqrt(mean(bagp-valid_df$total_UPDRS)^2)  #0.0126947  
sqrt(mean(bagp1-train_df$total_UPDRS)^2)  # 0.002654275
importance(bagf)
## 
