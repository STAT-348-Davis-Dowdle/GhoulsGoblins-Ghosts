library(vroom)
library(tidymodels)
library(embed)
library(kernlab)
library(keras)
library(bonsai)
library(lightgbm)
library(dbarts)

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


# multinomial svm
recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "sample ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

model <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(model)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

folds <- vfold_cv(train, v = 5)

results <- workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid)

best <- results %>%
  select_best("roc_auc")

final_workflow <- workflow %>%
  finalize_workflow(best) %>%
  fit(train)

predictions <- final_workflow %>%
  predict(new_data = test, type = "class")

submission <- data.frame(id = test$id, type = predictions$.pred_class)

write.csv(submission, "svm_submission.csv", row.names = F)


# Neural Network
nnrecipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nnmodel <- mlp(hidden_units = tune(),
               epochs = 50) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nnwf <- workflow() %>%
  add_recipe(nnrecipe) %>%
  add_model(nnmodel)

nntuning_grid <- grid_regular(hidden_units(range = c(1, 80)),
                              levels = 30)

nnfolds <- vfold_cv(train, v = 5)

nntuned <- nnwf %>%
  tune_grid(resamples = nnfolds, 
            grid = nntuning_grid)

nntuned %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x= hidden_units, y = mean)) + geom_line()

#ggsave("nnplot.pdf")

nnbest <- nntuned %>%
  select_best("accuracy")

nnfinalwf <- nnwf %>%
  finalize_workflow(nnbest) %>%
  fit(train)

nnpredictions <- nnfinalwf %>%
  predict(new_data = test, type = "class")

nnsubmission <- data.frame(id = test$id, type = nnpredictions$.pred_class)

write.csv(nnsubmission, "nnsubmission.csv", row.names = F)


# Boosting
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boostwf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(boost_model)

boost_grid <- grid_regular(tree_depth(),
                           trees(),
                           learn_rate(), 
                           levels = 3)

boost_folds <- vfold_cv(train, v = 5, repeats = 1)

boost_results <- boostwf %>%
  tune_grid(resamples = boost_folds,
            grid = boost_grid)

bestboost <- boost_results %>%
  select_best("accuracy")

boost_wf <- boostwf %>%
  finalize_workflow(bestboost) %>%
  fit(train)

boost_predictions <- boost_wf %>%
  predict(new_data = test, type = "class")

boost_submission <- data.frame(id = test$id, type = boost_predictions$.pred_class)

write.csv(boost_submission, "boost_submission.csv", row.names = F)
