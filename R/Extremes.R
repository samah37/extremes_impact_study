url <-
  "https://cran.r-project.org/src/contrib/randomForest_4.7-1.tar.gz"
install.packages(url, repos = NULL, type = "source")
install.packages("doMC", repos = "http://R-Forge.R-project.org")
library(randomForest)
library(pROC)
library(ROCR)
library(data.table)
library(rms)
library(caret)
library(mccr)
library(foreach)
library(doMC)
library(randomForest)
library(arules)
library(Ckmeans.1d.dp)
library(effsize)
library(imputeTS)
library(ScottKnottESD)
library(rpart)
library(DescTools)
library(caret)
library(mlbench)
library(vegan)
library(DiscNoise)

# Correlation_analysis function performs spearman correlation analysis on a given dataset
#
# Arguments:
#   - data: The input dataset for correlation analysis
#
# Returns:
#   - A character vector containing the names of the variables that are highly correlated
#
Correlation_analysis <- function(data) {
  data[sapply(data, is.factor)] <-
    data.matrix(data[sapply(data, is.factor)])
  dmy <- dummyVars(" ~ .", data = data, fullRank = T)
  data_transformed <- data.frame(predict(dmy, newdata = data))
  correlation_matrix <- cor(data_transformed, method = "spearman")
  highly_correlated <-
    findCorrelation(correlation_matrix, cutoff = 0.7)
  return (names(data_transformed)[highly_correlated])
}

# Redundancy_analysis function performs R2 redundancy analysis on a given dataset
#
# Arguments:
#   - data: The input dataset for redundancy analysis
#
# Returns:
#   - Result of the redundancy analysis

Redundancy_analysis <- function(dataset) {
  column_names <- colnames(dataset)
  formula_str <- paste(column_names, collapse = " + ")
  formula <- formula(paste("~", formula_str))
  result <- redun(formula, data = dataset, r2 = 0.9, nk = length(column_names))
  return(result)
}



# create_bootstrap function creates out-of-sample bootstrap samples for validation
#
# Arguments:
#   - data: The input dataset for creating out-of-sample bootstrap samples
#   - boot_size: The number of bootstrap samples to create
#   - seed: Controls the randomness of the sampling process (optional, default = TRUE)
#     - If a numeric value is provided, it sets the seed for reproducibility
#     - If TRUE, it sets the seed to a fixed value (42) for reproducibility
#     - If FALSE, it does not set a seed
#
# Returns:
#   - A list containing two elements:
#     - train_indices: A list of indices representing the training samples for each bootstrap sample
#     - test_indices: A list of indices representing the testing samples for each bootstrap sample


create_bootstrap <- function(data, boot_size, seed = TRUE) {
  train_indices <- vector("list", length = boot_size)
  test_indices <- vector("list", length = boot_size)
  
  if (is.numeric(seed)) {
    set.seed(seed)
  } else if (seed == TRUE) {
    set.seed(42)
  } else {
    seed <- FALSE
  }
  
  for (count in seq_len(boot_size)) {
    boot_data <- data[sample(nrow(data), replace = TRUE), ]
    train_index <- sample(nrow(boot_data), size = nrow(boot_data), replace = TRUE)
    test_index <- setdiff(seq_len(nrow(boot_data)), train_index)
    
    train_indices[[count]] <- train_index
    test_indices[[count]] <- test_index
  }
  
  return(list(train_indices = train_indices, test_indices = test_indices))
}

# remove_noise function removes noisy data points from a variable in the dataset
#
# Arguments:
#   - train: The training dataset as a data frame
#   - dep_var: The name of the dependent variable column in the dataset
#   - cutpoint: The cutpoint value used to classify data points into two classes
#   - target: The target value used to define the range of noisy data points around the cutpoint
#
# Returns:
#   - The training dataset with noisy data points removed from the dependent variable
#

remove_noise <- function(train, var, cutpoint, target) {
  var_values <- train[, var]
  filtered_rows <- var_values < cutpoint - target | var_values > cutpoint + target
  return(train[filtered_rows, ])
}

# remove_extremes function removes extreme values from a training dataset
#
# Arguments:
#   - train: The training dataset as a data frame
#   - dep_var: The name of the dependent variable column in the training dataset
#   - percentage : The percentage of extreme values to remove (between 0 and 1)
#
# Returns:
#   - The filtered training dataset without extreme values
#
remove_extremes <- function(data, var, percentage) {
  lower_limit <- quantile(data[[var]], percentage)
  upper_limit <- quantile(data[[var]], 1 - percentage)
  filtered_data <-
    data[data[[var]] >= lower_limit & data[[var]] <= upper_limit,]
  return(filtered_data)
}




# Model_creator function creates and evaluates a machine learning model
#
# Arguments:
#   - classifier: The type of classifier to use for model building
#   - data: The input dataset for model training and evaluation
#   - dep_var: The name of the dependent variable in the dataset
#   - cutpoint: The cutpoint value for binarizing the dependent variable (optional)
#   - target: The target class for binary classification (optional)
#   - ...: Additional arguments to be passed to the build_model and predict_generic functions
#
# Returns:
#   - The median Area Under the ROC Curve (AUC) value for binary classifiers

Model_creator <-
  function(classifier,
           data,
           dep_var,
           cutpoint,
           target,
           ...) {
    indices <- create_bootstrap(data, 100)
    train_indices <- indices[[1]]
    test_indices <- indices[[2]]
    train <- data[train_indices[[1]],]
    train <- remove_noise(train, dep_var, cutpoint, target)
    test <- data[test_indices[[1]],]
    train[, dep_var] <- NULL
    test[, dep_var] <- NULL

    # Building the model
    model <- build_model(classifier, train, ...)
    actuals <- test$response
    test$response <- NULL
    predicted_response <- predict(model, newdata = test, type = "raw")
    predicted_probablity <-
      predict(model, newdata = test, type = "raw")
    classes <- unique(predicted_response)

    if (length(classes) == 2) {
      auc <- get_auc(actuals, as.matrix(predicted_probablity))
      return(median(auc))
    } else {
      stop(
        'Only binary classifiers are supported.'
      )
    }
  }

# best_model function finds the best target value for binary classification based on the Area Under the ROC Curve (AUC)
#
# Arguments:
#   - data: The input dataset for model training and evaluation
#   - dep_var: The name of the dependent variable in the dataset
#   - classifier: The type of classifier to use for model building
#   - limit: The upper limit of the target value range to explore
#   - step_size: The step size to increment the target value
#   - cutpoint: The cutpoint value for binarizing the dependent variable (optional)
#
# Returns:
#   - The best target value that maximizes the Area Under the ROC Curve (AUC)
#

best_model <-
  function(data,
           dep_var,
           classifier,
           limit,
           step_size,
           cutpoint = NULL) {
    sequence <- seq(0, limit, by = step_size)
    response <-
      ifelse(data[, dep_var] <= cutpoint, 'class1', 'class2')
    data <- cbind(data, response)
    best_target <- 0
    best_auc <- 0

    for (desc_percentage in sequence) {
      target <- cutpoint * (desc_percentage / 100)
      results <-
        Model_creator(classifier, data, dep_var, cutpoint, target)

      if (results > best_auc) {
        best_target <- target
        best_auc <- results
      }
    }

    return(best_target)
  }
# topk_impact function calculates the impact on top k most important features compared to a baseline model using topk_overlap
#
# Arguments:
#   - importance_results: A list of importance results for different models
#
# Returns:
#   - A list containing the impacts for top 5, top 3, and top 1 features
#
topk_impact <- function(importance_results) {
  #baseline model ranking
  sk <- as.data.frame(sk_esd(importance_results[1])$groups)
  baseline_rank <- as.list(rownames(sk))
  top1 <- list()
  top3 <- list()
  top5 <- list()
  overall <- list()
  for (i in 2:length(importance_results)) {
    sk1 <- as.data.frame(sk_esd(importance_results[i])$groups)
    model_rank <- as.list(rownames(sk1))
    baseline_feature_ID <- gsub(".*\\.", "", baseline_rank)
    model_feature_ID <- gsub(".*\\.", "", model_rank)
    top5 <-
      rbind(intersect(baseline_feature_ID[1:5], model_feature_ID[1:5]) / unique(c(
        baseline_feature_ID[1:5], model_feature_ID[1:5]
      )),
      top5)
    top3 <-
      rbind(intersect(baseline_feature_ID[1:3], model_feature_ID[1:3]) / unique(c(
        baseline_feature_ID[1:3], model_feature_ID[1:3]
      )),
      top3)
    top1 <-
      rbind(intersect(baseline_feature_ID[1:1], model_feature_ID[1:1]) / unique(c(
        baseline_feature_ID[1:1], model_feature_ID[1:1]
      )),
      top1)
  }
  return(top5, top3, top1)
}
# overall_impact function calculates the Kendall's tau correlation coefficients between pairs of importance rankings
#
# Arguments:
#   - importance_results: A list of importance results for different models
#
# Returns:
#   - A list containing the Kendall's tau correlation coefficients between pairs of importance rankings
#
overall_impact <- function(importance_results) {
  listRanks <- list()
  j <- 0
  list_global <- list()

  listRanks <- list()
  j <- 0
  for (k in 2:4) {
    for (i in c(1, k)) {
      rownames(importance_results[[i]]) <-
        paste0("X.", 0:(nrow(importance_results[[i]]) - 1))
      colnames(importance_results[[i]]) <-
        paste0("X.", 0:(ncol(importance_results[[i]]) - 1))
      j <- j + 1
      # Extract the groups from the importance result and convert to a data frame
      groups <- sk_esd(importance_results[i])$groups
      groups_df <- as.data.frame(groups)

      # Extract the row names of the data frame
      row_names <- rownames(groups_df)
      print(row_names)

      # Extract the last two characters from each row name
      last_two_chars <-
        substring(row_names, nchar(row_names) - 1, nchar(row_names))

      # Convert the character vector to numeric
      last_two_chars_numeric <- as.numeric(last_two_chars)

      # Add the numeric vector to the list
      listRanks[[j]] <- last_two_chars_numeric

    }

    combined_ranks <- do.call(cbind, listRanks)
    # Calculate Kendall's tau correlation coefficients between all pairs of vectors
    cor_matrix <-
      cor(as.numeric(unlist(listRanks[1])), as.numeric(unlist(listRanks[2])), method = "kendall")
    list_global <- rbind(list_global, as.list(cor_matrix))

  }
  return (list_global)

}

# compute_impact function calculates the impact of noise or extremes on model performance and importance rankings
#
# Arguments:
#   - data: The input data as a data frame
#   - dep_var: The name of the dependent variable column in the data
#   - classifier: The type of classifier to use ('rf', 'lrm', 'cart', 'knn')
#   - limit: The upper limit of the impact analysis (percentage)
#   - step_size: The step size for the impact analysis (percentage)
#   - parallel: Boolean indicating whether to run the analysis in parallel (default: FALSE)
#   - n_cores: Number of CPU cores to use for parallel execution (default: 1)
#   - boot_size: The number of bootstrap samples for model evaluation (default: 100)
#   - cutpoint: The cutpoint for splitting the dependent variable into classes (default: NULL)
#   - save_interim_results: Boolean indicating whether to save interim results (default: FALSE)
#   - dest_path: The destination path to save the interim results (required if save_interim_results is TRUE)
#   - option: The option for analysis: 'noise', 'extremes', or 'both'
#
# Returns:
#   - A list containing the impact analysis results: performance impact, top k impact, and overall impact
#
Evaluate_impact <-
  function(data,
           dep_var,
           classifier,
           limit,
           step_size,
           parallel = FALSE,
           n_cores = 1,
           boot_size = 100,
           cutpoint = NULL,
           save_interim_results = FALSE,
           dest_path = NULL,
           option = NULL) {
    stopifnot(!missing(dep_var),!missing(limit),!missing(step_size),
              !missing(option))

    stopifnot(is.data.frame(data))

    if (parallel && missing(n_cores))
      stop("Please ensure that you specify the number of cores for parallel execution.")

    if (!classifier %in% c('rf', 'lrm', 'cart', 'knn'))
      stop(
        "Please note that the program exclusively supports Random Forest, Logistic Regression, CART, and KNN classifiers."
      )

    if (!option %in% c('noise', 'extremes', 'both'))
      stop("Please be aware that the program solely supports the options: noise, extremes, or both")

    if (save_interim_results && missing(dest_path))
      stop("Please provide the destination path to save the results")

    boot_size <- ifelse(missing(boot_size), 100, boot_size)

    cutpoint <-
      ifelse(is.null(cutpoint), median(data[[dep_var]]), cutpoint)

    sequence <-
      seq(0,
          ifelse(option == "noise", limit, 0.15),
          by = ifelse(option == "noise", step_size, 0.05))

    performance_results <- list()
    importance_results <- list()
    response <-
      ifelse(data[[dep_var]] <= cutpoint, 'class1', 'class2')
    data <- cbind(data, response)

    for (desc_percentage in sequence) {
      if (option == "noise") {
        target <- cutpoint * (desc_percentage / 100)
        data <- data
      } else {
        target <-
          ifelse(
            option == "both",
            best_model(data, dep_var, classifier, limit, step_size, cutpoint),
            0
          )
        data <- remove_extremes(data, dep_var, desc_percentage)
      }

      results <- RWKH_framework(
          classifier,
          as.data.frame(data),
          parallel,
          n_cores,
          boot_size,
          dep_var,
          cutpoint,
          target,
          outliers_percentage
        )

      metrics <- do.call('rbind', lapply(results, function(x)
        x[[1]]))
      colnames(metrics) <-
        c('accuracy',
          'precision',
          'recall',
          'brier_score',
          'auc',
          'f_measure',
          'mcc')
      performance_results[[as.character(desc_percentage)]] <-
        metrics
      imp <- do.call('rbind', lapply(results, function(x)
        x[[2]]))
      importance_results[[as.character(desc_percentage)]] <- imp
    }

    if (save_interim_results == TRUE) {
      saveRDS(performance_results,
              paste(dest_path, '/', 'performance.rds', sep = ''))
      saveRDS(importance_results,
              paste(dest_path, '/', 'importance.rds', sep = ''))
    }

    h <- head(sequence, 1)
    t <- tail(sequence, 1)
    stub1 <- performance_results[[as.character(h)]]
    stub2 <- performance_results[[as.character(t)]]
    titles <-
      c('accuracy',
        'precision',
        'recall',
        'brier_score',
        'auc',
        'f_measure',
        'mcc')
    Performance_impact <- matrix(nrow = 7, ncol = 4)

    for (k in 1:7) {
      Performance_impact[k, 1] <- titles[k]
      Performance_impact[k, 2] <-
        round(max(median(stub1[, k])) / median(stub2[1, k]))
      Performance_impact[k, 3] <-
        ifelse(
          wilcox.test(stub1[, k], stub2[, k], paired = FALSE)$p.value < 0.05,
          'Significant',
          'Not-Significant'
        )
      Performance_impact[k, 4] <-
        paste(as.character(cohen.d(stub1[, k], stub2[, k])$estimate),
              as.character(cohen.d(stub1[, k], stub2[, k])$magnitude),
              sep = ' ')
    }



    return(list(
      Performance_impact,
      topk_impact(importance_results),
      overall_impact(importance_results)
    ))

  }

