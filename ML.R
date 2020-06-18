require(glmnet)
require(e1071)
require(nnet)
require(caret)
require(pROC)
# fetching and cleaning of data

data1 <- read.csv(paste0("D:/PersonalStuff/Master-DS/ML/data/breast-cancer-wisconsin.data") ,  header = FALSE, stringsAsFactors = F)

names(data1) <- c("ID", "thickness" , "cell_size", "cell_shape", "adhesion", "epithelial_size" , "bare_nuclei" , "bland_cromatin" , "normal_nucleoli","mitoses","class")

data = as.data.frame(data1)
data$bare_nuclei = replace(data$bare_nuclei , data$bare_nuclei == '?', NA)
data = na.omit(data)
data$class = (data$class/2) - 1

# separating data into training and testing 

set.seed(080817)
index = 1:nrow(data)
testindex = sample(index,trunc(length(index)/3))
testset = data[testindex,]
trainset = data[-testindex,]

x_train <- data.matrix(trainset[,2:10])
y_train <- data.matrix(trainset[,11])
x_test <- data.matrix(testset[,2:10])

y_test <- data.matrix(testset[,11])



# Applying GLM - LASSO 

glm_model = cv.glmnet(x_train , y_train , alpha=1,nfolds=10)
lambda.min = glm_model$lambda.min
glm_coef = round(coef(glm_model,s=lambda.min),2)

plot(lambda.min)


plot(glmnet(x_train,y_train,family="gaussian" , alpha = 1) , "lambda" , label=T , main="" )
abline(v=log(lambda.min), lty=3)


# applying svm 
dat = data.frame(x_train, y = as.factor(y_train))
svm_model = svm(x_train , y_train , cost=1 , gamma = c(1/(ncol(x_train)-1)) , kernel="radial" , cross=10)

#applying NN
nnet_model = nnet(x_train , y_train , size = 5)


#prediction
glm_pred = round(predict(glm_model, x_test , type="response"),0)
svm_pred = round(predict(svm_model,x_test,type="response") , 0)
nnet_pred = round(predict(nnet_model , x_test , type="raw"),0)

# confusion matrix

confusionMatrix(as.factor(glm_pred) , as.factor(y_test))
confusionMatrix(as.factor(svm_pred) , as.factor(y_test))
confusionMatrix(as.factor(nnet_pred) , as.factor(y_test))
#confusionMatrix(as.factor(glm_pred) , as.factor(y_test))


roc_glm = roc(as.vector(y_test) , as.vector(glm_pred))
roc_svm = roc(as.vector(y_test) , as.vector(svm_pred))
roc_nnet = roc(as.vector(y_test) , as.vector(nnet_pred))

plot.roc(roc_glm , ylim = c(0,1) , xlim = c(1,0) , main = "ROC Curves")

lines(roc_glm , col = "blue")
lines(roc_nnet , col= "green")
lines(roc_svm , col = "red")

legend("bottomright" , legend = c("GLM" , "SVM" , "NN") , col = c("blue","red","green") , lwd = 2)


auc(roc_glm)
auc(roc_svm)
auc(roc_nnet)


### testing of alogorithm 

thickness = 8 
cell_size = 7
cell_shape = 8
adhesion = 5
epithelial_size = 5
bare_nuclei = 7
bland_cromatin = 9
normal_nucleoli = 8
mitosis = 10 

new_data = c(thickness , cell_size , cell_shape , adhesion , epithelial_size , bare_nuclei , bland_cromatin , normal_nucleoli , mitosis)

new_pred_glm = predict(glm_model, data.matrix(t(new_data)) , type = "response")

new_pred_svm = predict(svm_model, data.matrix(t(new_data)) , type = "response")

new_pred_nnet = predict(nnet_model, data.matrix(t(new_data)) , type = "raw")


print(new_pred_glm)
print(new_pred_svm)
print(new_pred_nnet)


# ensemble voting

predictions = data.frame(glm_pred , svm_pred , nnet_pred)
names(predictions) = c("glm" , "svm" , "nnet")

predictions$sum = rowSums(predictions)


algorithm_n = 3 

predictions$ensemble_votes = round(predictions$sum / algorithm_n)

print(predictions$ensemble_votes[1:30])


