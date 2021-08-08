##### Installing the necessary packages

install.packages("FSelector")
install.packages("party")
install.packages("rpart.plot")
install.packages("data.tree")
install.packages("ggthemes")

##### Loading the necessary libraries

library(FSelector)
library(rpart)
library(caret)
library(rpart.plot)
library(data.tree)
library(dplyr)
library(caTools)

library(randomForest)
library(psych)
library(pROC)
library(Amelia)

library(ggplot2)
library(plotly)
library(ggthemes)

##### Importing and studying the data

# Importing the data 'water_potability.csv' located in the same folder as the R script
data <- read.csv("./water_potability.csv")

# First observations in the dataset
head(data)

# Last observations in the dataset
tail(data)

# Attribute structure
str(data)

# Basic statistics of the dataset attributes
summary(data)
describe(data)

# Displaying the missing values in each attribute of the dataset
missmap(data)

table(data$Potability)

##### Pre-Processing, cleaning and correcting the dataset

# Generate the list of indexes randomly
shuffle_index <- sample(1:nrow(data))

# We will use those indexes to shuffle the dataset (randomize it)
data <- data[shuffle_index, ]

# Converting the target attribut from numerical form to categorical
data <- mutate(data, Potability = factor(Potability, levels = c(0, 1), labels = c('No', 'Yes')))

# Removing all observations with missing values in them
data <- na.omit(data)
glimpse(data)

##### Divide the dataset into training data and testing data

# The data will be divided into 80% training data, and 20% testing data
set.seed(123)
sample = sample.split(data$Potability, SplitRatio = .80)

# Training data
train_data = subset(data, sample==TRUE)

# Testing data
test_data = subset(data, sample==FALSE)

# Water potability percentage in each of training data and testing data
prop.table(table(train_data$Potability))
prop.table(table(test_data$Potability))

##### Studying the dataset attributes (variables)

# Variables inmportance
# We used random forest just ot see the variables importance since we cant do that using decision tree only

rf_tmp <- randomForest(Potability ~ .,
                       data=train_data, ntree=1000,
                       keep.forest=FALSE,
                       importance=TRUE)


varImpPlot(rf_tmp, main = "Importance des variables")
importance(rf_tmp)

# GGplot Plots
 
feat_imp_df <- importance(rf_tmp) %>% 
    data.frame() %>% 
    mutate(feature = row.names(.)) 

# Feature Importance Graph | MeanDecreaseAccuracy

importanceAccuracyGraph <- ggplot(feat_imp_df, aes(x = reorder(feature, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
    geom_point() +
    coord_flip() +
    theme_classic() +
    labs(
      x     = "Feature",
      y     = "Importance",
      title = "Feature Importance Graph by MeanDecreaseAccuracy",
      color="Feature"
    )

ggplotly(importanceAccuracyGraph)


# Feature Importance Graph | MeanDecreaseGini

importanceGiniGraph <- ggplot(feat_imp_df, aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_point() +
    coord_flip() +
    theme_classic() +
    labs(
      x     = "Feature",
      y     = "Importance",
      title = "Feature Importance Graph by MeanDecreaseGini",
      color="Feature"
    )

ggplotly(importanceGiniGraph)

##### Create decision tree classifier and train it with training data

# Creating the classifier
tree <- rpart(Potability ~.,
              data = train_data, 
              method="class")

##### Decision tree prediction on the testing data

# Prediction on testing data
tree.Potability.predicted <- predict(tree, test_data, type='class')

# Calculate the error rate of the prediction on the testing data 
tab <- table(tree.Potability.predicted, test_data$Potability)
paste("Erreur sur le test_data :", round(1 - sum(diag(tab)) / sum(tab), digits = 2), "%")
cat("\n")

# Generating the 'ROC' curve for the prediction
roc(test_data$Potability,
    as.numeric(tree.Potability.predicted), 
    plot=TRUE, legacy.axes=TRUE, percent=TRUE, print.auc=TRUE)

# Evaluating decision tree classifier performance with confusion matrix
confusionMatrix(tree.Potability.predicted,test_data$Potability)

##### Visualizing the decision tree classifier

# Visualizing decision tree
prp(tree)
fancyRpartPlot(tree ,yesno=2,split.col="black",nn.col="black", 
               caption="",palette="Set3",branch.col="black")

#print(tree2)
#plot(tree2)

# Creating the confusion matrix manually to compare the error rate with succes rate
prediction <- predict(tree, test_data, type = 'class')

table_mat <- table(prediction, reference = test_data$Potability)
table_mat
  
# Calculating the accuracy of the decision tree model
accuracy_test_data <- sum(diag(table_mat)) / sum(table_mat)
cat("\n")
print(paste('Accuracy for test_data', round(accuracy_test_data * 100, digits = 2), "%"))



