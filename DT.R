rm(list = ls(all.names = TRUE))
df<- read.csv("sgemm_product.csv")
df$RunF<- (df$Run1..ms.+df$Run2..ms.+df$Run3..ms.+df$Run4..ms.)/4
mean(df$RunF)
df$RunF_binary <- 0
df$RunF_binary[df$RunF >= 217.572] <- 1
df <- df[ -c(15:19) ]
df$RunF_binary<-as.factor(df$RunF_binary)
#str(df$RunF_binary)
#Creating the orignal full length tree for the GPU data
#install.packages("tree")
library(tree)
set.seed(5000)
alpha     <- 0.8 # percentage of training set
inTrain   <- sample(1:nrow(df), alpha * nrow(df))
train1 <- df[inTrain,]
test1  <- df[-inTrain,]
#head(train1[,0:14])
library(tree)
tree.model1 <- tree(RunF_binary ~ ., data=train1,     split = c("deviance", "gini"))
summary(tree.model1)

#for the orignal model
pred_tr <- predict(tree.model1, train1[0:14], type= "class")
pred_ts <- predict(tree.model1, test1[0:14], type= "class")
table(pred_tr, train1$RunF_binary)
table(pred_ts, test1$RunF_binary)
sum(pred_ts==test1$RunF_binary)/nrow(test1)
sum(pred_tr==train1$RunF_binary)/nrow(train1)


########################FUNCTIONAL METHOD#################
pruning <- function(tree_model, best_array, testX, trainX,testY, trainY ){
  acc_ts<- list()
  acc_tr<- list()
  for (i in best_array) {
    pruned_trees=prune.tree(tree_model, best=i)
    pred_pr_tr <- predict(pruned_trees, trainX, type= "class")
    pred_pr_ts <- predict(pruned_trees, testX, type= "class")
    acc_ts[i]<-sum(pred_pr_ts==testY)/nrow(testX)
    acc_tr[i]<-sum(pred_pr_tr==trainY)/nrow(trainX)
    newList <- list("Train Acc" = acc_tr, "Test Acc" = acc_ts)
  }
  return(newList)
}




train1_x<-train1[,0:14]
train1_y<-train1[,15]
test1_x<-test1[,0:14]
test1_y<-test1[,15]
summary(tree.model1)
best_array1=c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

acclist1<-pruning(tree_model=tree.model1, best_array=best_array1, testX=test1_x, trainX=train1_x,testY=test1_y, trainY=train1_y )

##second dataset of voice recognition
df2<- read.csv("voice.csv")
set.seed(100)
inTrain2   <- sample(1:nrow(df2), alpha * nrow(df2))
train2 <- df2[inTrain2,]
test2  <- df2[-inTrain2,]
tree.model2 <- tree(label ~ ., data=train2,     split = c("deviance", "gini"))
summary(tree.model2)

train2_x<-train2[,0:20]
train2_y<-train2[,21]
test2_x<-test2[,0:20]
test2_y<-test2[,21]
best_array2=c(2,3,4,5,6,7,8)
acclist2<-pruning(tree_model=tree.model2, best_array=best_array2, testX=test2_x, trainX=train2_x,testY=test2_y, trainY=train2_y )
tr_acc2<-unlist(acclist2$`Train Acc`)
ts_acc2<-unlist(acclist2$`Test Acc`)

bestplt1<- 2:20
bestplt2<- 2:8
tr_acc1<-unlist(acclist1$`Train Acc`)
ts_acc1<-unlist(acclist1$`Test Acc`)

#install.packages("ggplot2")
library(ggplot2)
dfplt1<-data.frame(tr_acc1,ts_acc1,bestplt1)
#head(dfplt1)
ggplot(dfplt1, aes(bestplt1)) + 
  geom_line(aes(y = tr_acc1, colour = "Train")) + 
  geom_line(aes(y = ts_acc1, colour = "Test"))+
  xlab("Nodes")+ylab("Accuracy")+ggtitle("Tree Depth-Run Time Data")+
  labs(color = 'Data')


dfplt2<-data.frame(tr_acc2,ts_acc2,bestplt2)
ggplot(dfplt2, aes(bestplt2)) + 
  geom_line(aes(y = tr_acc2, colour = "Train")) + 
  geom_line(aes(y = ts_acc2, colour = "Test"))+
  xlab("Nodes")+ylab("Accuracy")+ggtitle("Tree Depth-Voice Recogntion")+
  labs(color = 'Data')

##Cross  validation plot for voice data
cv.model2 <- cv.tree(tree.model2,K = 10 )
plot(cv.model2)
#cross validation for GPU data
cv.model1 <- cv.tree(tree.model1,K = 10 )
plot(cv.model1)

# boosting the gpu with 4 node decision trees#
#install.packages("gbm")
library('gbm')
str(train1$RunF_binary)
gpu.boost<- gbm(RunF_binary ~ . ,data =train1,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4)
# relative importance of features
gpu.boost
summary(gpu.boost)



n.trees = seq(from=100 ,to=10000, by=100) 
str(test1$RunF_binary)
#Generating a Prediction matrix for each Tree
gpu_pred<-predict.gbm(gpu.boost,train1,n.trees = n.trees, type = "response")
head(gpu_pred) #dimentions of the Prediction Matrix
max(gpu_pred)
###################################################
# boosting the GPU DT #
#install.packages("caret")

library("caret")
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

objModel1 <- train(train1_x, train1_y, 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))
summary(objModel)
print(objModel)
test2_predb <- predict(object=objModel, test2_x, type='prob')
(t2_cl<- apply(test2_predb,1,which.max))
table(t2_cl,test2_y)
1-(26/(471+454))



# boosting the voice recognition with #

#install.packages("caret")
library("caret")
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

objModel <- train(train2_x, train2_y, 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))
summary(objModel)
print(objModel)
test2_predb <- predict(object=objModel, test2_x, type='prob')
(t2_cl<- apply(test2_predb,1,which.max))
table(t2_cl,test2_y)
1-(26/(471+454))
