###################################  CUTE-03  ###################################
## cleaning the working environment
rm(list=ls())
## setting the working directory
setwd("C:/Users/sirola/Downloads/CUTe")
dir()

### reading the data
bankdata<- read.csv("bankdata.csv",header = T)

##exploring  the data
str(bankdata)
summary(bankdata)
sum(is.na(bankdata))
names(bankdata)

## as the variables in the data are varying in different ranges we need to standardize the data
## removing thecolumns with na values more than 10% in their respective columns
## By observing we seen Attr37 have more than 10%
sum(is.na(bankdata$Attr37))
sum(is.na(bankdata$Attr21))
bankdata <- bankdata[,-c(21,37)]

## imputation of missing values
library(DMwR)
bankdata<- knnImputation(bankdata)
sum(is.na(bankdata))
## checking the values of the target
bankdata$target
levels(bankdata$target)
table(bankdata$target) 
##   No    Yes 
##  40921  2083 
## recode the levels of the target to 0 and 1's
bankdata$target <- ifelse(bankdata$target == "Yes",1, 0)
table(bankdata$target) 
##   0     1 
## 40921  2083 
bankdata$target<- as.factor(as.character(bankdata$target))
str(bankdata)

## seperating the target variable from the dataset 
bankdata1 <- bankdata[-63]
names(bankdata1)
summary(bankdata)

## Standardizing the data as the attributes having the data  
bankdatastd<- as.data.frame(scale(bankdata1))
summary(bankdatastd)

##no joining the target variable to the standardized dataset
bankdata2full <- as.data.frame(c(bankdatastd,bankdata[63])) 
names(bankdata2full)

## checking for zero variance attributes
library(caret)
nearZeroVar(bankdatastd)
## they are no zero variance attributes in the  data 

## Now checking for highly correlated vaiables in the data with 90% correlation
library(caret)
correlationmatrix= cor(bankdatastd)
print(correlationmatrix)
highcordata= findCorrelation(correlationmatrix,cutoff = 0.85)
highcordata
install.packages("corrplot")
library(corrplot)
corrplot(cor(bankdatastd),method="circle")
names(bankdata2full)
par(mfrow=c(1,1))
bankdatareduced=bankdata2full[-highcordata]

## checking for class imbalance in the data as it affects the accuracy of the model
prop.table(table(bankdatareduced$target))
###  0           1 
#### 0.95086705  0.04913295 
## we can see there is so 95% imbalance in the data so we try and smote it

## smoting the data

library(DMwR)
bankdatasmote <- SMOTE(target~., data=bankdatareduced, perc.over = 500, perc.under = 150 )
prop.table(table(bankdatasmote$target))
###        0         1 
##0.5555477 0.4444523 
## as the class imbalance is reduced to some extent we can build the classification model

######### Dividing the data into train and validation 
set.seed(123) 
# to take a random sample of  70% of the records for train data 
train = sample(1:nrow(bankdatasmote),nrow(bankdatasmote)*0.7) 
bankdata_train = bankdatasmote[train,] 
bankdata_test = bankdatasmote[-train,] 

write.csv(bankdata_train, file = "banktrain.csv")
write.csv(bankdata_test,file="banktest.csv")

## Buliding the Logistic Regression 
logreg<-glm(target~.,data=bankdata_train,family = "binomial")
summary(logreg)

prob_train<- predict(logreg,type = "response")
prob_valid<- predict(logreg,bankdata_test,type = "response")

library(ROCR) 
pred <- prediction(prob_train,bankdata_train$target)
perf <- performance(pred, measure="tpr", x.measure="fpr")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.1))
par(mfrow=c(1,1))

## by seeing the roc curve we can say by taking any threshold value we are getting equal ratio
## of TPR, FPR so it logistic regression is not better than random model

## error metrics on both train and validation
library(caret)
confusionMatrix(bankdata_train$target,prob_train)
## building the decision tree on the  train data

## buiding classification tree using r part
library(rpart)
library(rpart.plot)

## building the model on smoted data as the data has high class imbalance
DT_rpart_class <- rpart(target~.,data = bankdata_train,method = "class") 
DT_rpart_class

## predicting the target on both the train and validaion datasets
pred_Train1 <- predict(DT_rpart_class,newdata=bankdata_train,type = "class")

pred_valid1 <- predict(DT_rpart_class,newdata=bankdata_test,type = "class")

## error metrics on both train and validation
library(caret)
confusionMatrix(bankdata_train$target,pred_Train1)
##  Accuracy : 0.768 
##  Sensitivity : 0.8000         
##  Specificity : 0.7300
##  Kappa : 0.5314  
confusionMatrix(bankdata_test$target,pred_valid1)
##   Accuracy : 0.7602
##  Sensitivity : 0.7887   
##  Specificity : 0.7268
##   Kappa : 0.5165 


# Parameter tuning - Choosing Best CP

printcp(DT_rpart_class)

# it is an experimentation process try with different cp values grow the tree
#take cp=0.1,
DT_rpart_class1<-rpart(target~.,data=bankdata_train,method="class",control = rpart.control(cp=0.1))
printcp(DT_rpart_class1)

## as the x error indicating it still decreases  
##we experiment increaing it with cp 0.001
DT_rpart_class2<-rpart(target~.,data=bankdata_train,method="class",control = rpart.control(cp=0.001))
printcp(DT_rpart_class2)

## experiment it with cp = 0.00001
DT_rpart_class3<-rpart(target~.,data=bankdata_train,method="class",control = rpart.control(cp=0.00001))
printcp(DT_rpart_class3)
## at cp value 0.000034435 the x error decreased from it started increasing

## experiment it with cp = 0.0000001
DT_rpart_class4<-rpart(target~.,data=bankdata_train,method="class",control = rpart.control(cp=0.0000001))
printcp(DT_rpart_class4)
## at cp value 0.000026783 the x error started increasing
## now building th model on the cp value
DT_rpart_class5<-rpart(target~.,data=bankdata_train,method="class",control = rpart.control(cp=0.000026783))
plot(DT_rpart_class5)
text(DT_rpart_class5)

#Predict the target for train and test datasets
pred_Train2 = predict(DT_rpart_class5,newdata=bankdata_train, type="class")
pred_val2 = predict(DT_rpart_class5, newdata=bankdata_test, type="class")
#Error Metrics on train and test
confusionMatrix(bankdata_train$target,pred_Train2)
##  Accuracy : 0.9253 
##  Sensitivity : 0.9330          
##  Specificity : 0.9155
##  Kappa : 0.8485 
confusionMatrix(bankdata_test$target,pred_val2)
## Accuracy : 0.8426 
## Sensitivity : 0.8543         
## Specificity : 0.8279   
## Kappa       : 0.6815  


##### Building random forest 
library(randomForest)

model_rf <- randomForest(target ~ . , bankdata_train,ntree = 50,mtry = 5)
# We can also look at variable importance from the built model using the importance() function and visualise it using the varImpPlot() funcion
importance(model_rf)

varImpPlot(model_rf)

# Predict on the train data
preds_train_rf <- predict(model_rf)
confusionMatrix(preds_train_rf, bankdata_train$target)
## Accuracy : 0.9633 
## Sensitivity : 0.9629          
## Specificity : 0.9638 
# Store predictions from the model
F1_Score(bankdata_train, preds_train_rf, positive = NULL)
preds_rf <- predict(model_rf,newdata = bankdata_test)

confusionMatrix(preds_rf, bankdata_test$target)
##  Accuracy : 0.9363
##  Sensitivity : 0.9344         
##  Specificity : 0.9387
##  Kappa : 0.8716 
#### SVM MODEL
## Seperating the target and indepedent variables
x_train <- bankdata_train[-33]
x_test <- bankdata_test[-33]
## creating a dataframe with the target variable
y_train <- bankdata_train$target
y_test <- bankdata_test$target

## Building SVM Model On the train data
library(e1071)
svmModel <- svm(x = x_train,y = y_train, type = "C-classification",kernel = "linear", cost = 10) 
summary(svmModel)
# Predict on train and test using the model
pred_train3 =  predict(svmModel,x_train) # x is all the input variables
pred_val3  = predict(svmModel,x_test)

# Build Confusion matrix
table(y_train,pred_train3)
table(y_test,pred_val3)
library(caret)
confusionMatrix(pred_train3,y_train)
## Accuracy : 0.6615
## Sensitivity : 0.8851          
## Specificity : 0.3799
## Kappa : 0.2786 
confusionMatrix(pred_val3,y_test)
##Accuracy : 0.6565
##Sensitivity : 0.8867          
##Specificity : 0.3737  
##Kappa : 0.2728 
#######Build SVM model with RBF kernel#### 
RBFsvmModel = svm(x_train,y_train, method = "C-classification", kernel = "radial", cost = 10)
summary(RBFsvmModel)

# Predict on train and test using the model
pred_train  =  predict(RBFsvmModel, x_train) # x is all the input variables
pred_val    =  predict(RBFsvmModel,x_test)

# BuildingConfusion matrix
confusionMatrix(pred_train,y_train)
##Accuracy    : 0.7014
##Sensitivity : 0.7965         
##Specificity : 0.5817 
##Kappa       : 0.3846
confusionMatrix(pred_val,y_test)
##Accuracy     : 0.6983 
##Sensitivity  : 0.7978          
## Specificity : 0.5761
## Kappa       : 0.3801


## Building Another model on SVM
RBFsvmModel1 = svm(x_train,y_train, method = "C-classification", kernel = "radial", cost = 15)
summary(RBFsvmModel1)

# Predict on train and test using the model
pred_train4  =  predict(RBFsvmModel1,x_train) # x is all the input variables
pred_val4    =  predict(RBFsvmModel1,x_test)

# BuildingConfusion matrix
confusionMatrix(pred_train4,y_train)
##Accuracy    : 0.7081
##Sensitivity : 0.7903       
##Specificity : 0.6046
##Kappa       : 0.4003
confusionMatrix(pred_val4,y_test)
##Accuracy     : 0.7051
##Sensitivity  : 0.7978          
## Specificity : 0.5761
## Kappa       : 0.3801

## Building Another model on SVM
RBFsvmModel1 = svm(x_train,y_train, method = "C-classification", kernel = "radial", cost = 15)
summary(RBFsvmModel1)

# Predict on train and test using the model
pred_train4  =  predict(RBFsvmModel1,x_train) # x is all the input variables
pred_val4    =  predict(RBFsvmModel1,x_test)

# BuildingConfusion matrix
confusionMatrix(pred_train4,y_train)
##Accuracy    : 0.7081
##Sensitivity : 0.7903       
##Specificity : 0.6046
##Kappa       : 0.4003
confusionMatrix(pred_val4,y_test)
##Accuracy     : 0.7051
##Sensitivity  : 0.7978          
## Specificity : 0.5761
## Kappa       : 0.3801

### 
RBFsvmModel2 = svm(x_train,y_train, method = "C-classification", kernel = "radial", cost = 25)
summary(RBFsvmModel2)

# Predict on train and test using the model
pred_train5  =  predict(RBFsvmModel2,x_train) # x is all the input variables
pred_val5    =  predict(RBFsvmModel2,x_test)

# BuildingConfusion matrix
confusionMatrix(pred_train5,y_train)
##Accuracy    : 0.714
##Sensitivity : 0.7867       
##Specificity : 0.6224
##Kappa       : 0.4137
confusionMatrix(pred_val5,y_test)
##Accuracy     : 0.7112
##Sensitivity  : 0.7880          
## Specificity : 0.6170
## Kappa       : 0.4096






#Grid Search/Hyper-parameter tuning

tuneResult <- tune(svm,train.x = x_train, train.y = y_train,
                   ranges = list(cost = 2^(2:3)),class.weights= c("0" = 1, "1" = 1),
                   tunecontrol=tune.control(cross=3))print(tuneResult)
summary(tuneResult)

#Predict model and calculate errors
tunedModel <- tuneResult$best.model;tunedModel

# Predict on train and test using the model
pred_train6  =  predict(tunedModel, x_train) # x is all the input variables
pred_val6=predict(tunedModel,x_test)
# Building Confusion matrix
confusionMatrix(pred_train6,y_train)
# Accuracy : 0.697
# Sensitivity : 0.8046          
# Specificity : 0.5614
# Kappa : 0.3734
confusionMatrix(pred_val6,y_test)
# Accuracy : 0.6924
# Sensitivity : 0.8056          
# Specificity : 0.5534  
# Kappa : 0.366  



####### Ignore The below one its still running


#Hyper Parameter search using caret
library(caret)
trctrl <- trainControl(method = "cv", number = 10)

# grid <- expand.grid(sigma=c(1,0.1,0.01,0.001,0.0001),cost= c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
#                     tau=0.1)

svmGrid <- expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5))

set.seed(45)
mod <- train(bankdata_train$target ~ ., data = bankdata_train, 
             method = "svmRadial",
             tuneGrid = svmGrid,
             metric = "Accuracy",
             trControl = trainControl(method = "cv",number = 5))



### Data Visualizations

names(bankdatasmote)
