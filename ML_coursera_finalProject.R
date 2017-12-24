rm(list = ls())

library(caret)
library(randomForest)
library(rpart)
library(e1071)

getwd()
setwd("/Users/prakhar/Desktop")

if(!dir.exists("ML_Project")){
  dir.create("ML_Project")
}

setwd("/Users/prakhar/Desktop/ML_Project")

url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url_train,destfile = paste(getwd(),"train.csv", sep = "/"), method = "curl")
download.file(url_test,destfile = paste(getwd(),"test.csv", sep = "/"), method = "curl")

traindata <- read.csv("train.csv")
testdata <- read.csv("test.csv")

traindata[traindata == ""] <- NA
testdata[testdata == ""] <- NA

#check if traindata has na values
sum(is.na(traindata))

#remove columns that wont be used for analysis
traindata[,1:7] <- NULL
testdata[,1:7] <- NULL

mydataTrain <- traindata

#remove columns with more than 50% NA values for traindata

for(i in 1:length(traindata)){
  if((sum(is.na(traindata[,i])))/nrow(traindata) >= .5){
    for(j in 1:length(traindata)){
      if(length( grep(names(traindata)[i], names(mydataTrain)[j])) == 1)
        mydataTrain[,j] <- NULL
    }
  }
}
traindata <- mydataTrain

mydataTrain <- testdata
#remove columns with more than 50% NA values for testdata
for(i in 1:length(testdata)){
  if((sum(is.na(testdata[,i])))/nrow(testdata) >= .5){
    for(j in 1:length(testdata)){
      if(length( grep(names(testdata)[i], names(mydataTrain)[j])) == 1)
        mydataTrain[,j] <- NULL
    }
  }
}
testdata <- mydataTrain
rm(mydataTrain)
rm(i)
rm(j)
rm(url_test)
rm(url_train)

#check if there are any more NA values
sum(is.na(testdata))
sum(is.na(testdata))

#check structure of file
str(traindata)

#preprocessing data
preprocessData <- preProcess(traindata, method = c("center", "scale"))
newdata <- predict(preprocessData, newdata = traindata)

#feature selection
model1 <- randomForest(newdata$classe~. , data = newdata, ntree = 100)
varImpPlot(model1)
varImp(model1)
rm(model1)
#creating new dataset with top 10 predictors
TrainingData <- data.frame(newdata$roll_belt, newdata$pitch_belt, newdata$yaw_belt, newdata$magnet_dumbbell_y,
                           newdata$magnet_dumbbell_z, newdata$roll_forearm, newdata$pitch_forearm,
                           newdata$magnet_dumbbell_x, newdata$accel_dumbbell_y, newdata$roll_dumbbell,
                           newdata$magnet_belt_y, newdata$gyros_belt_z, newdata$accel_belt_z, newdata$magnet_belt_z,
                           newdata$accel_dumbbell_z, newdata$accel_forearm_x
                           )
TrainingData$classe <- newdata$classe

rm(newdata)
rm(traindata)

#create similar set for testdata
TestData <- data.frame(testdata$roll_belt, testdata$pitch_belt, testdata$yaw_belt, testdata$magnet_dumbbell_y,
                       testdata$magnet_dumbbell_z, testdata$roll_forearm, testdata$pitch_forearm,
                       testdata$magnet_dumbbell_x, testdata$accel_dumbbell_y, testdata$roll_dumbbell,
                       testdata$magnet_belt_y, testdata$gyros_belt_z, testdata$accel_belt_z, testdata$magnet_belt_z,
                       testdata$accel_dumbbell_z, testdata$accel_forearm_x)
TestData$problem_id <- testdata$problem_id
rm(testdata)

#preprocess testdata
preprocessData <- preProcess(TestData[,-17], method = c("center", "scale"))
newdata <- predict(preprocessData, TestData[,-17])
newdata$problem_id <- TestData$problem_id
TestData <- newdata
rm(newdata)

#create validation set
inTrain <- createDataPartition(TrainingData$classe, p = 0.7, list = FALSE)
myTraining <- TrainingData[inTrain,]
myCrossValid <- TrainingData[-inTrain,]

#create different models
model1 <- randomForest(myTraining$classe~. , data = myTraining, ntree = 200)
model2 <- rpart(myTraining$classe~. , data = myTraining)
model3 <- svm(myTraining$classe~.,data = myTraining, type = "nu-classification")

#predicting on train dataset
pred1 <- predict(model1, data = myTraining, type = "class")
pred2 <- predict(model2, data = myTraining, type = "class")
pred3 <- predict(model3, data = myTraining, type = "class")

confusionMatrix(pred1, myTraining$classe)
confusionMatrix(pred2, myTraining$classe)
confusionMatrix(pred3, myTraining$classe)

pred_dataframe <- data.frame(pred1, pred2, pred3)

#creating ensemble model
pred_dataframe$classe <- myTraining$classe
model_rf_en <- randomForest(pred_dataframe$classe~., data = pred_dataframe,
                   n.trees = 100
                   )

#predicting on CrossValidation data
pred5 <- predict(model1, newdata = myCrossValid, type = "class")
pred6 <- predict(model2, newdata = myCrossValid, type = "class")
pred7 <- predict(model3, newdata = myCrossValid, type = "class")

pred_CV_df <- data.frame(pred5, pred6, pred7)
pred_CV_df$classe <- myCrossValid$classe
str(pred_test_df)
colnames(pred_CV_df) <- colnames(pred_dataframe)

pred8 <- predict(model_rf_en, newdata = pred_CV_df, type = "class")
confusionMatrix(pred8, pred_CV_df$classe)

#predicting on test data
colnames(TestData) <- colnames(myTraining)
pred9 <- predict(model1, newdata = TestData, type = "class")
pred10 <- predict(model2, newdata = TestData, type = "class")
pred11 <- predict(model3, newdata = TestData, type = "class")

pred_test_df <- data.frame(pred9, pred10, pred11)
colnames(pred_test_df) <- colnames(pred_dataframe[,1:3])

pred12 <- predict(model_rf_en, newdata= pred_test_df, type = "class")
TestData$classe <- pred12

write.csv(TestData$classe, "prediction.csv", row.names = FALSE)






