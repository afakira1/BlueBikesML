# BlueBikesML

# Rider Demand Prediction (Machine Learning Project)

## Executive Summary
  Bluebikes experiences highly variable rider demand based on seasonality, day of week, and time of day, which directly affects staffing, bicycle distribution, inventory, and fleet maintenance planning. This project developed a machine learning forecasting model to accurately predict total rider count and support operational decision-making. After cleaning and engineering temporal features from the raw dataset, multiple models were developed and evaluated, including Linear Regression, Regression Tree, Random Forest (Bagging), and a Stacking Ensemble.

  This means the model reduces forecasting mistakes by more than 92% compared to not using a model at all, giving us a much more accurate and reliable understanding of how many riders to expect at any given time. We also reduced the average size of our prediction mistakes (RMSE) from about 182 riders off to about 68 riders off — a 62% improvement. Instead of being nearly 200 bikes wrong on a typical forecast, we are now usually within about 70, enabling far more precise operational planning. These improvements support better balancing of bike availability across stations, optimization of rebalancing logistics, reduced operational costs, and improved customer experience by minimizing stockouts and excess inventory. This model can be used to support daily demand forecasting, seasonal planning, and staffing allocation.


## R Libraries and Utilities

- `lubridate` — date/time parsing and feature extraction
- `rpart` — regression tree modeling
- `rpart.plot` — tree visualization
- `randomForest` — bagging ensemble model
- `BabsonAnalytics.R` — helper script for `easyPrune()` tree pruning
- `stats` — built-in linear regression modeling

## Project Walk-through:
Preliminary
Step 1. Load & Clean Data
# Load data
df <- read.csv("data/train.csv", stringsAsFactors = FALSE)

# Source helper script (for pruning later)
source("scripts/BabsonAnalytics.R")

# Load required libraries
library(lubridate)
library(rpart)
library(rpart.plot)
library(randomForest)

# Convert field types / remove leakage
df$season     <- as.factor(df$season)
df$holiday    <- as.logical(df$holiday)
df$workingday <- as.logical(df$workingday)
df$weather    <- as.factor(df$weather)

df$registered <- NULL     # avoid target leakage
df$casual     <- NULL     # avoid target leakage

# Datetime parsing and feature engineering
df$datetime <- ymd_hms(df$datetime)
df$hour     <- hour(df$datetime)
df$day      <- wday(df$datetime)
df$month    <- month(df$datetime, label = TRUE)

# Convert engineered time columns to categorical
df[, c("hour","day","month")] <- lapply(df[, c("hour","day","month")], factor)

# Remove original datetime field
df$datetime <- NULL

# Preview structure
str(df)
names(df)

Step 2. Partition Data
set.seed(1234)
training_cases <- sample(nrow(df), round(nrow(df) * 0.60))

train <- df[training_cases, ]
test  <- df[-training_cases, ]

Step 3. Build Baseline Linear Regression Model
model_lr <- lm(count ~ ., data = train)
model_lr <- step(model_lr)      # Stepwise selection
summary(model_lr)

predictions_lr <- predict(model_lr, test)
observations   <- test$count

errors_lr <- observations - predictions_lr
rmse_lr   <- sqrt(mean(errors_lr^2))
mape_lr   <- mean(abs(errors_lr / observations))

Step 4. Regression Tree
model_rt <- rpart(count ~ ., data = train)
predictions_rt <- predict(model_rt, test)
errors_rt <- observations - predictions_rt
rmse_rt   <- sqrt(mean(errors_rt^2))
mape_rt   <- mean(abs(errors_rt / observations))

# Grow large tree and prune
stopping_rules <- rpart.control(minbucket = 200, minsplit = 100, cp = 0)
model_rt_big   <- rpart(count ~ ., data = train, control = stopping_rules)

model_prune_rt <- easyPrune(model_rt_big)
rpart.plot(model_prune_rt)

prediction_prune_rt <- predict(model_prune_rt, test)
error_prune_rt <- observations - prediction_prune_rt
rmse_prune_rt  <- sqrt(mean(error_prune_rt^2))
mape_prune_rt  <- mean(abs(error_prune_rt / observations))

Step 5. Bagging / Random Forest
rf <- randomForest(count ~ ., data = train, ntree = 500)

pred_rf <- predict(rf, test)
errors_rf <- observations - pred_rf

rmse_rf  <- sqrt(mean(errors_rf^2))
mape_rf  <- mean(abs(errors_rf / observations))

Step 6. Stacking Ensemble Model
# Predictions of all models for stacked learning
pred_lr_full <- predict(model_lr, df)
pred_rt_full <- predict(model_rt, df)
pred_rf_full <- predict(rf, df)

df_stacked <- cbind(df, pred_lr_full, pred_rt_full, pred_rf_full)

training_stacked <- df_stacked[training_cases, ]
test_stacked     <- df_stacked[-training_cases, ]

stacked <- randomForest(count ~ ., data = training_stacked, ntree = 500)
pred_stacked <- predict(stacked, test_stacked)

errors_stacked <- observations - pred_stacked
rmse_stacked   <- sqrt(mean(errors_stacked^2))
mape_stacked   <- mean(abs(errors_stacked / observations))
