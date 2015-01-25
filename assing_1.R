

# Loading Packages --------------------------------------------------------

library(caret)
library(corrgram)
library(caretEnsemble)


# Downloand and load data -------------------------------------------------

train  <- read.csv("pml-training.csv")
test  <- read.csv("pml-testing.csv")


# Preprocessing data ------------------------------------------------------

str(train)

#removing dates, etc.



drop_col <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
                    "cvtd_timestamp", "new_window", "num_window")

train1 <- train[,!(names(train) %in% drop_col)]
test1 <- test[,!(names(test) %in% drop_col)]

str(train1)


#select variables, wich can be used for test set prediction, no missing....

summary(test)




train1 <- train1[,(colSums(is.na(train1)) == 0)]
test1 <- test1[,(colSums(is.na(test1)) == 0)]

train1b <- train1b[,!(names(train1b) %in% drop_col)]
test1b <- test1b[,!(names(test1b) %in% drop_col)]



keep_col <- c(names(test1), "classe")
train1 <- train1[,names(train1) %in% keep_col]

train1 <- train1[,!(names(train1) %in% drop_col)]
test1 <- test1[,!(names(test1) %in% drop_col)]


#convert factor to numeric
for (i in 1:54){
  train1[,i] <- as.numeric(train1[,i])
  test1[,i] <- as.numeric(test1[,i])
print(i)
}

str(train1)



#identifying variables with small predicitive power

summary(train)

nzv <- nearZeroVar(train1, saveMetrics=T)
nzv


train2 <- train1[,-nzv$nzv]

#Identifying correlated predictors

descrCor <- cor(train2[,1:152])
summary(descrCor[upper.tri(descrCor)])

#cutoff at abs(0.8)

highlyCorDescr <- findCorrelation(descrCor, cutoff = .8)
train3 <- train2[,-highlyCorDescr]


rf_mod  <- train(classe~., data=train1b, method='rf', metric='Accuracy',
                 trControl=trainControl(classProbs=T ))

rf_mod2 <- train(classe~., data=train2, method="rf", metric='Accuracy', 
                 trControl=trainControl(classProbs=T, "repeatedcv", 
                                        number=10, repeats=10),
                 preProc=c("center", "scale"))


train2b <- train2
train2b$predict <- predict(rf_mod2, newdata=train2b)


confusionMatrix(data=RF.testSet$Prediction, RF.testSet$Defect)

#ensemble

my_control1 <- trainControl(method='boot', number=25, savePredictions=TRUE,
                            classProbs=TRUE)

model_list <- caretList(classe~., data=train1b,
  trControl=my_control1,
  methodList=c('knn','rf', 'rpart', 'gbm', 'C5.0', 'gam', 'gamboost', 'gamLoess',
               'gamSpline', 'gcvEarth', 'parRF')
)

c5_mod <- train(classe~., data=train1b, method='C5.0', metric='Accuracy',
                trControl=trainControl(classProbs=T ))

str(train1b)

warnings()



my_control2 <- trainControl(method = "repeatedcv",
                            repeats = 3,
                            savePredictions=TRUE,
                            classProbs=TRUE)


rf_1 <- train(classe~., data=train1b, method='parRF', metric='Accuracy',
              trControl=my_control2,
              preProc = c("center", "scale"))



svmFit <- train(classe ~ ., data = train1b,
                method = "svmRadial",
                trControl = my_control2,
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "Accuracy")



#-------------------------------------------------

set.seed(12345)
inTrain <- createDataPartition(y = train1b$classe,
                                    p = .75,
                                     list = FALSE)

training <- train1[inTrain,]
testing <- train1[-inTrain,]

#First step, train a random forest


rf1 <- randomForest(classe~.,data=training, ntree=250, importance=TRUE)
#variable importance

varImpPlot(rf1)

#performance on the test set:

confusionMatrix(predict(rf1,newdata=testing[,-ncol(testing)]),testing$classe)

#caret train

rf_2  <- train(classe~., data=training, method='rf', metric='Accuracy',
                 trControl=trainControl(method="cv", number=4, classProbs=T),
               preProc = c("center","scale"))


confusionMatrix(predict(rf_2,newdata=testing[,-ncol(testing)]),testing$classe)



svmFit <- train(classe ~ ., data = training,
                method = "svmRadial",
                trControl = trainControl(method="cv", number=4, classProbs=T),
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "Accuracy")

svmFit2 <- train(classe ~ ., data = training,
                method = "svmPoly",
                trControl = trainControl(method="cv", number=4, classProbs=T),
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "Accuracy")



tree1 <- train(classe ~ ., data = training,
               method = "rpart",
               trControl = trainControl(method="cv", number=4, classProbs=T),
               preProc = c("center", "scale"),
               tuneLength=25,
               metric = "Accuracy")

tree2 <- train(classe ~ ., data = training,
               method = "ctree",
               trControl = trainControl(method="cv", number=4, classProbs=T),
               preProc = c("center", "scale"),
               metric = "Accuracy")

gbm_grid <- expand.grid(.interaction.depth = (1:5) * 2,
                        .n.trees = (1:10)*50, .shrinkage = .1)



gbm1 <- train(classe ~ ., data = training,
               method = "gbm",
               trControl = trainControl(method="cv", number=4, classProbs=T),
               preProc = c("center", "scale"),
               metric = "Accuracy",
              tuneGrid=gbm_grid)

log_boost1 <- train(classe ~ ., data = training,
              method = "LogitBoost",
              trControl = trainControl(method="cv", number=4, classProbs=T),
              preProc = c("center", "scale"),
              metric = "Accuracy")

knn1 <- train(classe ~ ., data = training,
              method = "knn",
              trControl = trainControl(method="cv", number=4, classProbs=T),
              preProc = c("center", "scale"),
              metric = "Accuracy")

confusionMatrix(predict(knn1,newdata=testing[,-ncol(testing)]),testing$classe)

