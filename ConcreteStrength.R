
# title: "Concrete Strength Prediction with R: A Machine Learning Workflow for
# Compressive Strength Regression"

# subtitle: "End-to-End Modeling with workflow_set and Visual Insights via patchwork"

# author: "Damiano Pincolini"

# date: 2024-11-23


# 1. PREFACE

# 1.1. Project goal

# The overall goal of this project is to find out an efficient regression model trained
# and tested on a dataset available on Kaggle.com
# (https://www.kaggle.com/datasets/prathamtripathi/regression-with-neural-networking).
# Beside this, there are a couple of more specific aims to be set:
# 1.  evaluate the impact of applying preprocessing steps either to every variable
# or to a limited group of them,
# 2.  optimizing data visualization in order to increase the overall readability of
# the document.

# As far as point 1 is concerned, the main focus is on usage of the
# workflow_set package, while for second point patchwork and kableExtra packages
# are involved.

# To keep the whole process fast and fluent, I will not use a wide set of engines since
# the main focus is on how to manage preprocessing and to improved data visualization.

# 1.2. Loading packages

pacman::p_load(tidyverse,
               SmartEDA,
               DescTools,
               DataExplorer,
               factoextra,
               ggcorrplot,
               GGally,
               ggforce,
               corrplot,
               cowplot,
               tidymodels,
               randomForest,
               kknn,
               nnet,
               kernlab,
               broom,
               themis,
               rpart.plot,
               vip,
               shapviz,
               knitr,
               kableExtra,
               patchwork)



# 1.3. Data loading and content analysis

# After downloading the .csv file from kaggle.com (see link above) and storing,
# I have saved it into R environment as "DataOrigin" dataset.

DataOrigin <- read_csv2("C:/Users/casa/Documents/Damiano/R/Mentoring/19. concreteStrenght/rawData/Concrete_Data.csv")


# 1.4. Dataset analysis

# 1.4.1. Dataset structure, datatype and quality analysis

options(scipen = 999)


DataOriginQuality1 <- ExpData(DataOrigin, type = 1)

DataOriginQuality1|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)


DataOriginQuality2 <- ExpData(DataOrigin, type = 2)

DataOriginQuality2|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "scale_down"),
                position = "left",
                full_width = FALSE)


DataOrigin|> ExpNumStat(by = "A",
                       Qnt = c(0.25, 0.75),
                       MesofShape = 2,
                       Outlier = TRUE,
                       round = 2)|>
  select(variable = Vname,
         min,
         max,
         mean,
         median,
         quartile1 = "25%",
         quartile3 = "75%",
         SD,
         IQR)|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "scale_down"),
                position = "center",
                full_width = FALSE)


# There's a combination of two concret's "ingredients" that is specifically useful:
# Water/Cement Ratio. This is the most critical parameter for compressive strength.
# A lower ratio (less water) increases compressive strength because it reduces the
# internal porosity of the concrete. However, if the ratio is too low, it can lead
# to compaction difficulties. So, compressive strength is inversely proportional to
# the w/c ratio. 

DataOrigin|>
  group_by(Days)|>
  summarize(count = n())|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)


# 1.4.2. Data featuring

# For further steps, I want to create the following variables:
# 1.  dayRange (categorical) that represents the evolution of strenght based on the
#     numbers of day of concrete preparation.
# 2.  waterCementRatio (numerical).
# 3.  StrengthCategory (categorical).

Data <- DataOrigin|>
  mutate(dayRange = as_factor(case_when(Days == 1 ~ "1 day",
                                        Days == 3 ~ "3 days",
                                        Days == 7 ~ "7 days",
                                        Days == 14 ~ "14 days",
                                        Days == 28 ~ "28 days",
                                        Days == 56 ~ "56 days",
                                        Days == 90 | Days == 91 ~ "90 days",
                                        Days == 100 | Days == 120 ~ "100 days",
                                        Days == 180 ~ "180 days",
                                        Days == 270 ~ "270 days",
                                        Days == 360 | Days == 365 ~ "365 days")),
         dayRange = fct_relevel(dayRange, c("1 day",  "3 days", "7 days", "14 days",
                                            "28 days", "56 days", "90 days",
                                            "100 days", "180 days",
                                            "270 days", "365 days")),
         waterCementRatio = round(Water/Cement, 2),
         StrengthCategory = as_factor(case_when(CompressiveStrength < 10 ~ "Poor",
                                                CompressiveStrength >= 10 & CompressiveStrength < 20 ~ "Fair",
                                                CompressiveStrength >= 20 & CompressiveStrength < 35 ~ "Good",
                                                CompressiveStrength >= 35 & CompressiveStrength < 50 ~ "Very good",
                                                CompressiveStrength >= 50 ~ "Excellent")),
         StrengthCategory = fct_relevel(StrengthCategory, c("Poor",  "Fair", "Good", "Very good", "Excellent")))


# 1.5. Data partitioning

set.seed(789, sample.kind = "Rounding")

DataSplit <- Data|>
  initial_split(prop = 0.80)

DataTrain <- training(DataSplit)

DataTest <- testing(DataSplit)


# 2. EXPLORATIVE DATA ANALYSIS

# 2.1 Feature analysis

# 2.1.1. Target analysis

EdaNum <- DataTrain|>
  ExpNumStat(by = "GA",
             gp = "StrengthCategory",
             Qnt = c(0.25, 0.75),
             MesofShape = 2,
             Outlier = TRUE,
             round = 2)

EdaDistributionShape <- EdaNum|>
  select(Vname, Group, Kurtosis, Skewness)|>
  filter(Group == "StrengthCategory:All")|>
  arrange(desc(Skewness))

p1 <- DataTrain|>
  ggplot(aes(CompressiveStrength))+
  geom_density()+
  labs(title = "Target variable density")

p2 <- EdaDistributionShape|>
  ggplot(aes(Vname, Skewness))+
  geom_col(fill = "blue", alpha = 0.7)+
  geom_text(aes(label = round(Skewness, 1)), 
            hjust = -0.2, 
            color = "darkblue", 
            size = 2) +  # Aggiunge i valori accanto alle barre
  coord_flip()+
  labs(x = "Skewness",
       y = "Predictors",
       title ="Skewness")+
  theme(plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7))               

p3 <- EdaDistributionShape|>
  ggplot(aes(Vname, Kurtosis))+
  geom_col(fill = "red", alpha = 0.7)+
  geom_text(aes(label = round(Kurtosis, 1)), 
            hjust = -0.2, 
            color = "darkred", 
            size = 3) +  # Aggiunge i valori accanto alle barre
  geom_hline(aes(yintercept = 3),
             color = "darkred",
             linewidth = 0.3)+
  coord_flip()+
  labs(x = "Kurtosis",
       y = "Predictors",
       title = "Kurtosis")+
  theme(plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7))


p1 + (p2 / p3)+
  plot_layout(ncol = 2,
              nrow = 1,
              widths = c(2,1),
              heights = NULL)


DataTrainLong <- DataTrain|>
  select(-c(CompressiveStrength, dayRange))|>
  pivot_longer(cols = -c(StrengthCategory),
               names_to = "variable",
               values_to = "value")

DataTrainLong|>
  ggplot(aes(x = value,
             fill = StrengthCategory))+
  geom_boxplot()+
  facet_wrap(~ variable, scales = "free")+
  labs(x = "Value",
       title = "Predictors box-plot") +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_blank(),
        strip.text = element_text(size = 10))


# 2.1.2. Univariate predictors analysis

DataTrainLong|>
  ggplot(aes(x = value,
             fill = StrengthCategory))+
  geom_density(alpha = 0.3)+
  facet_wrap(~ variable,
             ncol = 3,
             scales = "free")+
  labs(x = "Value",
       y = "Density",
       title = "Predictors distribution")+
  theme(legend.position = "bottom",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        strip.text = element_text(size = 10))


DataTrainLong2 <- DataTrain|>
  select(-c(StrengthCategory, dayRange))|>
  pivot_longer(cols = -c(CompressiveStrength),
               names_to = "variable",
               values_to = "value")


DataTrainLong2|>
  ggplot(aes(x = value, y = CompressiveStrength))+
  geom_hex(alpha = 0.7)+
  facet_wrap(~ variable,
             ncol = 3,
             scales = "free")+
  labs(x = "Predictors' values",
       y = "Compressive Strength",
       title = "Correlation between each predictor and target variable")+
  theme(legend.position = "bottom",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7),
        strip.text = element_text(size = 10))


### 2.1.3. Multivariate predictors analysis

EdaCorMatr <- round(cor(DataTrain[,-c(9, 10, 12)], use = "complete.obs"), 1)

ggcorrplot(EdaCorMatr,
                 hc.order = TRUE,
                 type = "lower",
                 lab = TRUE,
                 lab_size = 3,
                 tl.cex = 7,
                 digits = 1,
                 title = "Predictors' correlation",)


# 2.2. Conclusions

# a.  Insight about skewness:
# -   high skewness for one predictor (Days) (Fisher's Gamma Index \> 3),
# -   five predictors (waterCementRation, Superplaticize, Blast, Cement, FlyAsh)
#     with a moderate skewness (values between 0.5 and 1),
# -   the remaining three features show a low-level of skewness.

# A transformation to handle this shape distribution is to be evaluated.
# Specifically, it will be interesting to try a selective transformation only on
# extremely skewed variables.

# b.  Insight about kurtosis:

# -   Days feature is strongly leptokurtic (around 12),
# -   all remaining features have a kurtosis index below 2 (from 1.4 to -1.4) which
#     testify a significant platykurtic distribution.

# c.  Outliers do exist, but it can be assumed that cases with very high or low values
#     are explainable by the final scope for which that concrete blend has been
#     calculated.

# d.  Only two predictors are correlated between themselves. Cement and (the new
#     created) waterCementRatio.


# 3. TRAINING AND TESTING ML MODELS WITH WORKFLOW_SETS PACKAGE

# Some features have a non-normal distribution visible via skewness and/or kurtosis.
# In order to get a performing ML model, an effective pre-processing path is to be
# selected. Specifically, I'm interested in checking if a general predictors'
# transformation works better or worst than a targeted feature transformation.
# During preprocessing phase, it'll be convenient to handle:
# -   the skewness of some or all predictors,
# -   the kurtosis of some or all predictors,
# -   the skewness and the kurtosis of target variable.

# Above mentioned skewness and kurtosis is supposed to be reduced with a power
# transformation (Yeo-Johnson) that is applied to either all predictors, or a group
# of them; in both cases, target variable will be either transformed or kept with
# its original scale.

# To keep the process simple, I'll pick only a couple of models that both seem
# appropriate to this case and are supposed to be sensitive to any of the above
# mentioned "issues" referred to dataset features:
# -   XGBoost (sensible to outliers),
# -   linear regression (sensible to skewness).

# Regarding to metric, the need for transforming variables (possibly including output
# as well) involves to use only metrics not expressed in the target variable's scale in
# order to keep results comparable. To this purpose, R-squared seems to fit the bill.
# It expresses the proportion of the variance in the response variable returned by a
# model that can be explained by the predictor variables. Its values ranges from 0
# (worst predictive model) to 1 (best predictive model) despite the scale of features
# used.


# 3.1. Training

# 3.1.1. Setting cross validation and metrics selection

set.seed(123, sample.kind = "Rounding")

WfSetCvFolds <- vfold_cv(DataTrain, v = 5)

custom_metrics <- metric_set(rsq)


### 3.1.2. Preprocessing recipes

WfSetRecipe1Plain <- DataTrain|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)

WfSetRecipe2All <- DataTrain|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)|>
  step_YeoJohnson(all_predictors())

WfSetRecipe2Any <- DataTrain|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)|>
  step_YeoJohnson(Days,
                  waterCementRatio,
                  Superplasticizer,
                  BlastFurnaceSlag,
                  Cement,
                  FlyAsh)

WfSetRecipe3All_target <- DataTrain|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)|>
  step_YeoJohnson(all_predictors(),
                  CompressiveStrength)

WfSetRecipe3Any_target <- DataTrain|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)|> 
  step_YeoJohnson(Days,
                  waterCementRatio,
                  Superplasticizer,
                  BlastFurnaceSlag,
                  Cement,
                  FlyAsh,
                  CompressiveStrength)


# 3.1.3. Model Specifications

WfSetModXgb <-
  boost_tree(tree_depth = tune(),
             trees = tune())|>
  set_engine("xgboost")|>
  set_mode("regression")

WfSetModLR <-
  linear_reg()|>
  set_engine("lm")|>
  set_mode("regression")


# 3.1.4. Workflow Sets

WfSetWorkflows <-
  workflow_set(preproc = list(NoPrep = WfSetRecipe1Plain,
                              YjAll = WfSetRecipe2All,
                              YjAny = WfSetRecipe2Any,
                              YjAll_target = WfSetRecipe3All_target,
                              YjAny_target = WfSetRecipe3Any_target),
               models = list(LR = WfSetModLR,
                             Xgb = WfSetModXgb))


# 3.1.5. Tuning and selection best performing workflow and hyperparameters

WfSetGridCtrl<- control_grid(
  save_pred = TRUE,
  parallel_over = "resamples",
  save_workflow = TRUE)

WfSetGridResults <-
  WfSetWorkflows %>%
  workflow_map(
    seed = 1503,
    resamples = WfSetCvFolds,
    grid = 5,
    metrics = custom_metrics,  
    control = WfSetGridCtrl)

WfSetRankResults <- WfSetGridResults|>
  rank_results(rank_metric = "rsq",
               select_best = FALSE)|>
  select(wflow_id, .metric, mean, rank)|>
  group_by(wflow_id, .metric)|>
  summarize(CvAvgScore = mean(mean), .groups = "drop")|>
  arrange(desc(CvAvgScore))|>
  mutate(Rank = dense_rank(desc(CvAvgScore)))|>
  arrange(Rank)

WfSetRankResults|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)

# According to rsq metric we can observe that XgBoost brings definitely better
# performance than linear regression regardless of preprocessing steps.
# Specifically, XGBoost model's perfomances are pretty steady and show a very
# limited range: from 0.9127 to 0.9175, while linear regression rsq varies from
# 0.6136 to 0.8149. This confirms that XGBoost is less sensitive to dataset
# issues like outliers, skewness etc and thus to preprocessing activities.
# On the other hand, skewness and/or kurtosis have stonger impact on linear
# regression model and this requires a significant handling before running models.
# Nevertheless, even after such a preprocessing session, the "plain" XGBoost model
# still performs better than the most intensively preprocessed workflow with LR model.

# Let's pick the best hyperparameters that have been used during model training
# (5 combinations have been set in the workflow mapping).

WfSetBestResult <- 
  WfSetGridResults |> 
  extract_workflow_set_result("YjAny_target_Xgb") |>
  select_best(metric = "rsq")

WfSetBestResult|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)


# 3.2. Testing

# 3.2.1. Performance on the test set

WfSetTestResults <- 
  WfSetGridResults %>% 
  extract_workflow("YjAny_target_Xgb") %>% 
  finalize_workflow(WfSetBestResult) %>% 
  last_fit(split = DataSplit,
           metrics = custom_metrics) 

collect_metrics(WfSetTestResults)|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)

# An rsq equal to 0.96 states that the chosen model is able to expalin the 96% of
# variance of the test dataset.
# Quite suriprisingly, this result outperform the rsq of training phase where 0.91 rsq
# was reached.

# As previuosly said, the model selection has been based on rsq due to the
# tranformations that in some cases have been applied to the target variable.

# It may be interesting measuring other metrics (notably RMSE and MAE) on the test
# dataset transformed according to the preprocess recipe that has been choosen
# (Yeo-Jonhson applied to both some predictors and target feature) so to have a more
# complete view of the model's prections ability.

# To to that, we need to prep the recipe and apply to test dataset and then extract
# RMSE and MAE which we are interested to investigate.

# rmse on test set

WfSetTestResults_rmse <- 
  WfSetGridResults %>% 
  extract_workflow("YjAny_target_Xgb") %>% 
  finalize_workflow(WfSetBestResult) %>% 
  last_fit(split = DataSplit,
           metrics = metric_set(rmse)) # fisso la metrica rmse come oggetto metric_set

collect_metrics(WfSetTestResults_rmse)|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)

# mae on test set

WfSetTestResults_mae <- 
  WfSetGridResults %>% 
  extract_workflow("YjAny_target_Xgb") %>% 
  finalize_workflow(WfSetBestResult) %>% 
  last_fit(split = DataSplit,
           metrics = metric_set(mae))

collect_metrics(WfSetTestResults_mae)|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)


# To evaluate this metrics, it is necessary to know the content of the test dataset
# preprocessed with the Yeo-Johnson transformation applied to a group of predictors
# (Days, waterCementRatio, Superplasticizer, BlastFurnaceSlag, Cement and FlyAsh) and
# target.

# prep recipe on test set and extract transformed test dataset

WfSetRecipeYjAny_target <- DataTest|>
  recipe(CompressiveStrength ~ .)|>
  step_rm(dayRange, StrengthCategory)|> # step "base" per pulire il dataset da dati categoriali
  step_YeoJohnson(Days,
                  waterCementRatio,
                  Superplasticizer,
                  BlastFurnaceSlag,
                  Cement,
                  FlyAsh,
                  CompressiveStrength)|>
  prep()

DataTestPrepYjAny_target <- WfSetRecipeYjAny_target|>
  juice()


# After, transforming test set, the values of the target value (CompressiveStrength)
# changes significantly.

targetPrepMean <-
  round(mean(DataTestPrepYjAny_target$CompressiveStrength),2)

targetOrigMean <-
  round(mean(DataTest$CompressiveStrength), 2)

DataTestTargetMean <-
  tibble(Test_dataset = c("Original", "Preprocessed"),
         Target_variable_mean = c(targetOrigMean, targetPrepMean))

DataTestTargetMean|>
  kable(format = "latex",
        booktabs = TRUE)|>
  kable_styling(latex_options = c("striped", "hold_position"),
                position = "left",
                full_width = FALSE)


# 3.2.2. Residuals analysis

WfSetResults <- WfSetTestResults|>
  collect_predictions()|>
  mutate(residuals = .pred - CompressiveStrength)

p7_1 <- WfSetResults|>
ggplot(aes(x = .pred, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red") +
  scale_y_continuous(limits = c(-10, 10))+
  labs(title = "Residuals vs Predicted values",
       x = "Predicted values",
       y = "Residuals") +
  theme_minimal()


# Residuals distribution

p7_2 <- WfSetResults|>
  ggplot(aes(x = residuals))+
  geom_histogram(bins = 30, fill = "lightblue", color = "black") +
  labs(title = "Residuals distribution",
       x = "Residuals",
       y = "Count") +
  theme_minimal()


# Q-Q plot 

p7_3 <- WfSetResults|>
ggplot(aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(title = "Residuals vs Theoretical",
       subtitle = "Q-Q Plot",
       x = "Theoretical",
       y = "Residuals") +
  theme_minimal()


p7_1 / (p7_2 + p7_3)
