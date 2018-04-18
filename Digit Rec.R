library(keras)


# Read Inputs
train <- read.csv(file="C:/Users/DASLAB Hareland4/Desktop/Digit/train.csv", header=TRUE, sep=",")
print(head(train[1:10]))


# Preparing the Data
train <-as.matrix(train)
dimnames(train) <- NULL
train[,2:ncol(train)]<-normalize(train[,2:ncol(train)])
train[,1]<-as.numeric(train[,1])
print(head(train[1:10]))

#creating train and test data
set.seed(5421)
ind<-sample(2,nrow(train),replace = T, prob = c(0.8, 0.2))
x_train <- train[ind==1,]
x_test <- train[ind==2,]
y_train <- x_train[,1]
y_test <- x_test[,1]
x_train <- x_train[,2:ncol(x_train)]
x_test <- x_test[,2:ncol(x_test)]


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Defining the Model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 784, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 500, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# Compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Training and Evaluation
history <- model %>% fit(
  x_train, y_train, 
  epochs = 20, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)



test <- read.csv(file="C:/Users/DASLAB Hareland4/Desktop/Digit/test.csv", header=TRUE, sep=",")
test <- as.matrix(test)
test <- normalize(test)
predict <- model %>% predict_classes(test)

sample <- read.csv(file="C:/Users/DASLAB Hareland4/Desktop/Digit/sample.csv", header=TRUE, sep=",")
kaggle<-data.frame(ImageId=sample[,1],Label=predict)
write.csv(kaggle,"C:/Users/DASLAB Hareland4/Desktop/Digit/Kaggle.csv",row.names = F)


