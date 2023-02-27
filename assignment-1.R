install.packages('car')
library(car)
library(dplyr)

# load the data
data <- read.csv("https://raw.githubusercontent.com/Akshayy99/Big_Data_Analytics_IA1/main/arrest_data.csv")

for (i in 11:ncol(data)) {
  most_common_value <- data %>% 
    select(i) %>% 
    drop_na() %>% 
    group_by(!!sym(names(data)[i])) %>% 
    tally() %>% 
    arrange(desc(n)) %>% 
    slice(1) %>% 
    pull(!!sym(names(data)[i]))
  
  # find the second most common value
  second_most_common_value <- data %>% 
    select(i) %>% 
    drop_na() %>% 
    filter(!!sym(names(data)[i]) != most_common_value) %>% 
    group_by(!!sym(names(data)[i])) %>% 
    tally() %>% 
    arrange(desc(n)) %>% 
    slice(1) %>% 
    pull(!!sym(names(data)[i]))
  
  # replace NA with the second most common value
  data[, i] <- ifelse(is.na(data[, i]), second_most_common_value, data[, i])
}

# create a subset of the data with relevant variables
subset_data <- data %>%
  select(arrest, fin, age, race, wexp, mar, paro, prio, educ, emp1:emp52)

# check for multicollinearity
vif_results <- vif(lm(arrest ~ ., data = subset_data))
vif_results


# create a data frame with only the employment status variables
emp_vars <- data[, grepl("^emp", names(data))]

# perform principal component analysis
pca_emp <- prcomp(emp_vars, scale = TRUE)

# create composite variable as the first principal component
data$emp_comp <- pca_emp$x[,1]

# check VIF of new composite variable
vif(lm(arrest ~ emp_comp + fin + race + wexp + mar + paro + prio + educ, data = data))

set.seed(111)

# Split data into training and validation sets
train_index <- sample(nrow(data), nrow(data)*0.6)
train_data <- data[train_index, ]
val_data <- data[-train_index, ]

# Perform logistic regression
logit_model <- glm(arrest ~ race + wexp + mar + paro + prio + educ + fin + emp_comp, data = train_data, family = binomial(link = "logit"))
summary(logit_model)
# Generate predictions on validation and fitted data
val_preds <- predict(logit_model, val_data, type = "response")
train_preds <- predict(logit_model, train_data, type = "response")

fitprob_logit <- cbind(val_data, val_preds)

ggplot(fitprob_logit, 
       aes(x = factor(arrest), 
           y = val_preds)) +
  geom_point(stat='identity')

fitprob_logit['binary_prob']=factor(ifelse(fitprob_logit$val_preds>0.3,1,0))


cm=confusionMatrix(data= fitprob_logit$binary_prob,reference=factor(val_data$arrest),positive="0")
cm

library(ROCR)
library(pROC)
fitted <- as.numeric(fitprob_logit$binary_prob)
actual <- valid.df$arrest

predobj=prediction(fitted, actual)
rocobj=performance(predobj,measure="tpr",x.measure="fpr")

plot(rocobj, col = "red", main = "ROC Curves", xlab = "False Positive Rate", 
     ylab = "True Positive Rate")

aucobj=performance(predobj,measure="auc")
auc1<- round(as.numeric(aucobj@y.values),3)

# Generate confusion matrices for validation and fitted data
library(caret)
val_cm <- confusionMatrix(factor(ifelse(val_preds > 0.25, 1, 0)), factor(val_data$arrest), positive='0')
train_cm <- confusionMatrix(factor(ifelse(train_preds > 0.25, 1, 0)), factor(train_data$arrest), positive='0')

# View confusion matrices
val_cm
train_cm


