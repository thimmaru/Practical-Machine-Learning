#Install required packages
install.packages("caret", dependencies = TRUE)
install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("corrplot")

#Load required packages
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

setwd("C:\\Users\\TRUDR\\OneDrive - Monsanto\\Migrated from My PC\\Desktop\\Data\\Practical Machine Learning\\Course_Project")

# Create directory, downloading the dataset from the source and unzip
if(!file.exists("./data")){dir.create("./data")}

#url for the dataset source for the project:

# Read the Data
trainSet <- read.csv("./data/pml-training.csv")
testSet <- read.csv("./data/pml-testing.csv")
dim(trainSet)
dim(testSet)

# Clean the data
sum(complete.cases(trainSet))
# Remove columns that contain NA missing values
trainSet <- trainSet[, colSums(is.na(trainSet)) == 0] 
testSet <- testSet[, colSums(is.na(testSet)) == 0] 

# Remove variables/ columns that do not contribute to accelerometer measurements
classe <- trainSet$classe
trainRemove <- grepl("^X|timestamp|window", names(trainSet))
trainSet <- trainSet[, !trainRemove]
trainCleaned <- trainSet[, sapply(trainSet, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testSet))
testSet <- testSet[, !testRemove]
testCleaned <- testSet[, sapply(testSet, is.numeric)]


# Slice the data
set.seed(123) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

#Data Modeling
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf

#Then, we estimate the performance of the model on the validation data set
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)

#Model accuracy
accuracy <- postResample(predictRf, testData$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose

# Predicting for Test Data Set
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result

# Correlation & decision tree plots
# 1. Corellation Matrix
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")

# 2. Decision Tree Visualization
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot
