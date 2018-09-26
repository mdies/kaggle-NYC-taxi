# From https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious

library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('xgboost') # modelling
library('caret') # modelling
library('magrittr') # function "%>%"
library('plotly')

setwd("/Users/marta/Dropbox/DataScience/JobHunting/AIA/Kaggle_NYCTaxiTripDuration")

trainfinal <- read.csv("/Users/marta/Dropbox/DataScience/JobHunting/AIA/Kaggle_NYCTaxiTripDuration/trainfinal3.csv")
trainfinal <- trainfinal[-1]
# Note that for trainfinal, 'trip_duration' is already in the format log(trip_duration +1) to
# comply with the evaluation metrics
testfinal <- read.csv("/Users/marta/Dropbox/DataScience/JobHunting/AIA/Kaggle_NYCTaxiTripDuration/testfinal.csv")
testfinal <- testfinal[-1]

# Specific definitions:
#---------------------------------
# predictor features

train_cols <- c("direct_dist", "rush_hour", "vendor_id", "EWR_airport",
                "JFK_airport", "LG_airport", "dropoff_longitude",
                "dropoff_latitude", "pickup_longitude", "pickup_latitude",
                "blizzard","avg_temp","precipitation")

y_col <- c("trip_duration")

cols <- c(train_cols, y_col)

#---------------------------------

# Cross validation step
set.seed(134)
trainIndex <- createDataPartition(trainfinal$trip_duration, p = 0.8, list = FALSE, times = 1)
train <- trainfinal[trainIndex,]
valid <- trainfinal[-trainIndex,]

# Reformatting data to feed XGBoost algorithm
foo <- train %>% select(-trip_duration)
bar <- valid %>% select(-trip_duration)

dtrain <- xgb.DMatrix(as.matrix(foo),label = train$trip_duration)
dvalid <- xgb.DMatrix(as.matrix(bar),label = valid$trip_duration)
dtest <- xgb.DMatrix(as.matrix(testfinal))

# Defining XGBoost hyper-parameters
xgb_params <- list(colsample_bytree = 0.7, #variables per tree 
                   subsample = 0.7, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 5, #tree levels
                   eta = 0.3, #shrinkage
                   eval_metric = "rmse", 
                   objective = "reg:linear",
                   seed = 4321
                   )

watchlist <- list(train=dtrain, valid=dvalid)

# Training the classifier
set.seed(134)
gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   print_every_n = 5,
                   watchlist = watchlist,
                   nrounds = 60)

# Cross validation
# Due to running time constrains, setting nrounds=15 (should be ~100!!!)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 5, nrounds=15)

# Feature importance
imp_matrix <- as.tibble(xgb.importance(feature_names = colnames(train %>% select(-trip_duration)), model = gb_dt))

imp_matrix %>%
  ggplot(aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")

# Predicting trip durations for testfinal data set
test_preds <- predict(gb_dt,dtest)
# Physical time, in seconds
pred_trip_duration <- exp(test_preds) - 1
phys_train_trip_duration <- exp(trainfinal$trip_duration) -1

# Plot histograms and compare them
# Overlaying histograms - Histogram Colored (blue and red)
# https://www.r-bloggers.com/overlapping-histogram-in-r/
hist(phys_train_trip_duration/60, col=rgb(1,0,0,0.5),
     xlim=c(0,60), ylim=c(0,0.08),
     main="Histogram for NYC taxi trip duration", xlab="Trip duration (min)",
     breaks=1000, prob = TRUE) 
hist(pred_trip_duration/60, col=rgb(0,0,1,0.5), breaks=100, prob = TRUE, add=T)  
legend("topright", c("train", "test prediction"), fill=c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)))
box()

# https://datascienceplus.com/standard-deviation-vs-standard-error/
# Standard Error of the Mean (SEM)
mean(phys_train_trip_duration/60)
sd(phys_train_trip_duration/60)/sqrt(length(phys_train_trip_duration/60))

mean(pred_trip_duration/60)
sd(pred_trip_duration/60)/sqrt(length(pred_trip_duration/60))

