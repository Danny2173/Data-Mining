# Read Data set
framinghamdata <- read.csv("framingham.csv")

# Install if necessary

# install.packages("tidyverse")
# install.packages("summarytools")
# install.packages("corrplot")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("cluster")
# install.packages("factoextra")
# install.packages("rpart")
# install.packages("ROSE")
# install.packages("class")
# install.packages("rpart.plot")
# install.packages("randomForest")
# install.packages("caret")

# Libraries
library(tidyverse)
library(summarytools)
library(corrplot)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(rpart)
library(ROSE)
library(class)
library(rpart.plot)
library(randomForest)
library(caret)

######### Data Exploration #########

# 1. Shape of the data set (4240x16)
dim(framinghamdata)

# 2. Examining Data Structure
str(framinghamdata)
glimpse(framinghamdata)

# 3. Variable Analysis
summary(framinghamdata)

# 4. Subset view
head(framinghamdata)

# 5. Target variable Distribution
table(framinghamdata$TenYearCHD)

######### Demographic Analysis ############

# 1. Age Distribution
qplot(age, data = framinghamdata, bins = 20, col = I("black"), xlab = "Age", ylab = "Count")

# 2. Gender + Age Distribution

# Initial analysis (42.9% Male)
sum(framinghamdata$male)/count(framinghamdata)

# Finding the mid-point (50% ideal gender line)
binned_data <- framinghamdata %>%
  group_by(age) %>%
  summarise(total_count = n(), midpoint = total_count / 2)

# Plot stacked bar chart 
ggplot(framinghamdata, aes(x = age, fill = factor(male, labels = c("Female", "Male")))) +
  geom_bar() +
  geom_errorbar(data = binned_data, aes(x = age, ymin = midpoint, ymax = midpoint),
                color = "black", width = 0.8, size = 0.8, inherit.aes = FALSE) +
  labs(x = "Age", y = "Count", fill = "Gender")

# 3. Education Level

# Removing Missing Values for clean chart

filtered_data <- framinghamdata %>%
  filter(!is.na(education))

# Grouping education levels and finding percentages

education_summary <- filtered_data %>%
  group_by(education) %>%
  summarise(count = n()) %>%
  mutate(percentage = round((count / sum(count)) * 100, 1))

# Plotting a bar chart of the education level distribution
ggplot(education_summary, aes(x = factor(education), y = count, fill = factor(education))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(percentage, "%")), vjust = 1.5, color = "black", size = 5) +
  scale_x_discrete(labels = c("1" = "Some High School", "2" = "High School Graduate", 
                              "3" = "Some College/Vocational", "4" = "College Graduate")) +
  labs(x = NULL, y = "Count", fill = "Education Level") +
  theme(axis.text.x = element_blank())

############ Correlation Analysis #############

# Identifying continuous variables
continuous_vars <- framinghamdata[, c("totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose")]

# Pair plot
pairs(continuous_vars, pch = 21, bg = "lightblue", lower.panel = panel.smooth, upper.panel = NULL)

############ Handling Missing Values and Duplicates #############

# Counting rows with missing values
sum(rowSums(is.na(framinghamdata)) > 0)

# Counting by variable
missing_count <- colSums(is.na(framinghamdata))

# Ordering variables by missing value count
missing_counts <- sort(missing_count[missing_count > 0], decreasing = TRUE)

# Bar plot showing missing value distribution
barplot(missing_counts,
        col = "lightblue", 
        ylab = "Count of Missing Values", 
        las = 2,
        ylim = c(0,400)) 

# Checking for Duplicates
sum(duplicated(framinghamdata))

# Dealing with missing values - continuous variables (median)
framinghamdata$cigsPerDay[is.na(framinghamdata$cigsPerDay)] <- median(framinghamdata$cigsPerDay, na.rm=TRUE)
framinghamdata$totChol[is.na(framinghamdata$totChol)] <- median(framinghamdata$totChol, na.rm=TRUE)
framinghamdata$BMI[is.na(framinghamdata$BMI)] <- median(framinghamdata$BMI, na.rm=TRUE)
framinghamdata$glucose[is.na(framinghamdata$glucose)] <- median(framinghamdata$glucose, na.rm=TRUE)
framinghamdata$heartRate[is.na(framinghamdata$heartRate)] <- median(framinghamdata$heartRate, na.rm=TRUE)

# Dealing with missing values - categorical variables (mode)
mode <- function(x) {
  uniquex <- unique(x)
  uniquex[which.max(tabulate(match(x, uniquex)))]
}

framinghamdata$education[is.na(framinghamdata$education)] <- mode(framinghamdata$education)
framinghamdata$BPMeds[is.na(framinghamdata$BPMeds)] <- mode(framinghamdata$BPMeds)

# Check to see if it worked
sum(rowSums(is.na(framinghamdata)) > 0)

############## Data preparation (conversion and scaling) #################

# Variable Conversion
framinghamdata$male <- as.factor(framinghamdata$male)
framinghamdata$currentSmoker <- as.factor(framinghamdata$currentSmoker)
framinghamdata$BPMeds <- as.factor(framinghamdata$BPMeds)
framinghamdata$prevalentStroke <- as.factor(framinghamdata$prevalentStroke)
framinghamdata$prevalentHyp <- as.factor(framinghamdata$prevalentHyp)
framinghamdata$diabetes <- as.factor(framinghamdata$diabetes)
framinghamdata$TenYearCHD <- as.factor(framinghamdata$TenYearCHD)

framinghamdata_scale <- framinghamdata

# Scaling for numerical variables
num_cols <- c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose")
framinghamdata_scale[num_cols] <- scale(framinghamdata_scale[num_cols])
summary(framinghamdata_scale[num_cols])

# Check 
str(framinghamdata_scale)

# Principal Component Analysis
pca_model <- prcomp(framinghamdata_scale[num_cols], scale. = TRUE)
pca_model
summary(pca_model)
df_pca <- pca_model$x[,1:7]

print(df_pca)
# ----------

############# Clustering ############


set.seed(42)

fviz_nbclust(df_pca, kmeans, method = "silhouette")

clusters3 <- kmeans(df_pca, centers = 2, nstart = 25)
print(clusters3)

############# Classification ############


n <- nrow(framinghamdata_scale)
set.seed(42)
train <- sort(sample(1:n, floor(0.7 * n)))
framinghamdata_scale.train <- framinghamdata_scale[train, ]
framinghamdata_scale.test <- framinghamdata_scale[-train, ]

dim(framinghamdata_scale)
# Oversampling the minority class

set.seed(42)
balanced_data <- ovun.sample(TenYearCHD ~ ., data = framinghamdata_scale.train, method = "over", N = 2 * table(framinghamdata_scale.train$TenYearCHD)[1])$data

table(balanced_data$TenYearCHD)

# 1. Logistic Regression
logit.model <- glm(TenYearCHD ~ .,data = balanced_data, family = "binomial")

# 1.1 Predicting on test 
logit.prob <- predict(logit.model, framinghamdata_scale.test, type = "response")
logit.pred <- ifelse(logit.prob > 0.5, 1, 0)

# 1.2 Accuracy
accuracy_logit = mean(logit.pred == framinghamdata_scale.test$TenYearCHD)
print(accuracy_logit)

# 1.3 Confusion matrix

confusion_matrix_logit <- table(Predicted = logit.pred, Actual = framinghamdata_scale.test$TenYearCHD)
print(confusion_matrix_logit)

# 2. KNN


# 2.1 Selecting numerical features
train.knn <- balanced_data[, c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose")]
test.knn <- framinghamdata_scale.test[, c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose")]

# 2.2 Target variable
cl <- balanced_data$TenYearCHD

# 2.3 Looping for k
accuracies <- numeric(40)
for (k in 1:40) {
  knn.pred <- knn(train = train.knn, test = test.knn, cl = cl, k = k)
  accuracies[k] <- mean(knn.pred == framinghamdata_scale.test$TenYearCHD)
}

# 2.4 Plotting
plot(1:40, accuracies, type = "b", pch = 19, col = "blue",
     xlab = "Number of Neighbors", ylab = "Accuracy")

# 2.5 Choosing k = 1
knn.pred.1 <- knn(train = train.knn, test = test.knn, cl = cl, k = 1)

# 2.6 Confusion Matrix
confusion_matrix_k1 <- table(Predicted = knn.pred.1, Actual = framinghamdata_scale.test$TenYearCHD)
print(confusion_matrix_k1)

# 2.7 Accuracy
accuracy_k1 <- mean(knn.pred.1 == framinghamdata_scale.test$TenYearCHD)
print(accuracy_k1)


# 3. Decision tress

# 3.1 Using pre-scaled dataset

n <- nrow(framinghamdata)
set.seed(42)
train <- sort(sample(1:n, floor(0.7 * n)))
framinghamdata.train <- framinghamdata[train, ]
framinghamdata.test <- framinghamdata[-train, ]

dim(framinghamdata)

# Oversampling the minority class
set.seed(42)
balanced_data_no_scale <- ovun.sample(TenYearCHD ~ ., data = framinghamdata.train, method = "over", N = 2 * table(framinghamdata.train$TenYearCHD)[1])$data

# 3.1 Build the model
tree.model <- rpart(TenYearCHD ~ ., data = balanced_data_no_scale, 
  method = "class", 
  parms = list(split = "information"), 
  control = rpart.control(cp = 0.01, minsplit = 10, maxdepth = 4)
)

summary(tree.model)

# 3.2 Plot the decision tree
rpart.plot(tree.model, type = 3, extra = 104, fallen.leaves = TRUE)

# 3.3 Evaluate on test data
tree.pred <- predict(tree.model, framinghamdata.test, type = "class")

# 3.4 Confusion Matrix
confusion_matrix_tree <- table(Predicted = tree.pred, Actual = framinghamdata.test$TenYearCHD)
print(confusion_matrix_tree)

# 3.5 Accuracy 
accuracy_tree <- mean(tree.pred == framinghamdata.test$TenYearCHD)
print(accuracy_tree)

# 4. Random Forest

set.seed(42)
rf.model <- randomForest(TenYearCHD ~ ., data = balanced_data_no_scale, 
  ntree = 500, mtry = 4, importance = TRUE)

# 4.1 Evaluate on the test set
rf.pred <- predict(rf.model, newdata = framinghamdata.test)

# 4.2 Confusion matrix
confusion_matrix_rf <- table(Predicted = rf.pred, Actual = framinghamdata.test$TenYearCHD, positive = "1")
print(confusion_matrix_rf)

# 4.3 Accuracy 
accuracy_rf <- mean(rf.pred == framinghamdata.test$TenYearCHD)
print(accuracy_rf)

# 4.4 Checking importance of variables
importance(rf.model)
varImpPlot(rf.model)


# 5. Final Evaluation

# Compare accuracies
comparison_table <- data.frame(
  Model = c("Logistic Regression", "KNN", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_logit, accuracy_k1, accuracy_tree, accuracy_rf)
)

print(comparison_table)

# 1. Logistic Regression Confusion Matrix
confusion_matrix_logit <- confusionMatrix(as.factor(logit.pred), framinghamdata_scale.test$TenYearCHD, positive = "1")
print(confusion_matrix_logit)

# 2. KNN Confusion Matrix
confusion_matrix_k1 <- confusionMatrix(as.factor(knn.pred.1), framinghamdata_scale.test$TenYearCHD, positive = "1")
print(confusion_matrix_k1)

# 3. Decision Tree Confusion Matrix
confusion_matrix_tree <- confusionMatrix(as.factor(tree.pred), framinghamdata.test$TenYearCHD, positive = "1")
print(confusion_matrix_tree)

# 4. Random Forest Confusion Matrix
confusion_matrix_rf <- confusionMatrix(as.factor(rf.pred), framinghamdata.test$TenYearCHD, positive = "1")
print(confusion_matrix_rf)

# 5. Evaluation
comparison_results <- data.frame(
  Model = c("Logistic Regression", "KNN", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_logit, accuracy_k1, accuracy_tree, accuracy_rf),
  Precision = c(confusion_matrix_logit$byClass["Precision"], confusion_matrix_k1$byClass["Precision"], confusion_matrix_tree$byClass["Precision"], confusion_matrix_rf$byClass["Precision"]),
  Recall = c(confusion_matrix_logit$byClass["Recall"], confusion_matrix_k1$byClass["Recall"], confusion_matrix_tree$byClass["Recall"], confusion_matrix_rf$byClass["Recall"]),
  F1_Score = c(confusion_matrix_logit$byClass["F1"], confusion_matrix_k1$byClass["F1"], confusion_matrix_tree$byClass["F1"], confusion_matrix_rf$byClass["F1"])
)

print(comparison_results)