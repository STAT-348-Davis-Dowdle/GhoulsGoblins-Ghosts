library(vroom)
library(tidymodels)

# set working directory
setwd("C:/Users/davis/OneDrive - Brigham Young University/Documents/skool/new/stat 348/GhoulsGoblins&Ghosts/GhoulsGoblinsGhosts")

# read in training and test data
train <- vroom("train.csv")
test <- vroom("test.csv")


# data imputation
imp_train <- vroom("trainWithMissingValues.csv")

imp_recipe <- recipe(type ~ ., data = imp_train) %>%
  update_role(id, new_role = "sample ID") %>%
  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(hair_length, has_soul, color), trees = 500) %>%
  step_impute_bag(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul, color), trees = 500)

imp_dataset <- bake(prep(imp_recipe), new_data = imp_train)  

rmse_vec(train[is.na(imp_train)], imp_dataset[is.na(imp_train)])
