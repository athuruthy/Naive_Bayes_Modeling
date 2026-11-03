# ==============================================================================
# 1. LOAD LIBRARIES
# ==============================================================================
# e1071: Implementation of the Naive Bayes algorithm
# caret: Model evaluation and confusion matrix computation
# ggplot2 & cowplot: Visualization and plot styling
# tidyr & dplyr: Data reshaping, pivoting, and aggregation
# pander: Markdown-compliant table formatting
library(e1071)
library(caret)
library(ggplot2)
library(cowplot)
library(tidyr)
library(dplyr)
library(pander)


# ==============================================================================
# 2. DATA IMPORT & CLEANING (RAW VALUES)
# ==============================================================================
# Read the raw plant-level EPA eGRID 2007 data file
orig_data <- read.csv("eGRID2007_plant.csv")

# Clean metadata header rows: remove first 3 description rows and promote row 1 to column names
data <- orig_data[-c(1:3), ] 
colnames(data) <- data[1, ]
data <- data[-1, ]

# Filter out observations missing a primary fuel designation and format as factor
data <- data[which(data$PLFUELCT != ""), ]
data$PLFUELCT <- as.factor(data$PLFUELCT)

# Strip comma thousands-separators and cast emission variables to numeric
data$PLSO2AN <- gsub(",", "", data$PLSO2AN)
data$PLSO2AN <- as.numeric(data$PLSO2AN)

data$PLNOXAN <- gsub(",", "", data$PLNOXAN)
data$PLNOXAN <- as.numeric(data$PLNOXAN)

data$PLCO2AN <- gsub(",", "", data$PLCO2AN)
data$PLCO2AN <- as.numeric(data$PLCO2AN)

data$PLCH4AN <- gsub(",", "", data$PLCH4AN)
data$PLCH4AN <- as.numeric(data$PLCH4AN)


# ==============================================================================
# 3. SUMMARY STATISTICS (RAW UNITS)
# ==============================================================================
# Calculate mean annual emissions by primary fuel type to inspect physical scales
summary_stats <- data %>%
  group_by(PLFUELCT) %>%
  summarize(
    "Avg CO2 (Tons)" = mean(PLCO2AN),
    "Avg SO2 (Tons)" = mean(PLSO2AN),
    "Avg NOx (Tons)" = mean(PLNOXAN),
    "Avg CH4 (Tons)" = mean(PLCH4AN),
  )

# Output summary table
summary_stats %>%  
  pander(caption = "Plant means by Fuel Category")


# ==============================================================================
# 4. LOG TRANSFORMATION FOR MODELING
# ==============================================================================
# Clean data to apply log1p transformation (log(1 + x))
# Normalizes heavily right-skewed emission distributions for Gaussian Naive Bayes
data <- orig_data[-c(1:3), ] 
colnames(data) <- data[1, ]
data <- data[-1, ]

data <- data[which(data$PLFUELCT != ""), ]
data$PLFUELCT <- as.factor(data$PLFUELCT)

data$PLSO2AN <- gsub(",", "", data$PLSO2AN)
data$PLSO2AN <- log1p(as.numeric(data$PLSO2AN))

data$PLNOXAN <- gsub(",", "", data$PLNOXAN)
data$PLNOXAN <- log1p(as.numeric(data$PLNOXAN))

data$PLCO2AN <- gsub(",", "", data$PLCO2AN)
data$PLCO2AN <- log1p(as.numeric(data$PLCO2AN))

data$PLCH4AN <- gsub(",", "", data$PLCH4AN)
data$PLCH4AN <- log1p(as.numeric(data$PLCH4AN))


# ==============================================================================
# 5. EXPLORATORY DATA VISUALIZATION
# ==============================================================================
# Pivot emission variables into long format for multi-panel ggplot faceting
plot_data <- data %>%
  select(PLFUELCT, PLNOXAN, PLSO2AN, PLCO2AN, PLCH4AN) %>%
  pivot_longer(
    cols = c(PLNOXAN, PLSO2AN, PLCO2AN, PLCH4AN), 
    names_to = "Emission_Type", 
    values_to = "Amount"
  )

# Generate violin plots to inspect distribution separation and overlap across fuel categories
ggplot(plot_data, aes(x = PLFUELCT, y = Amount, fill = PLFUELCT)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  facet_wrap(~ Emission_Type, scales = "free_y") + 
  labs(
    title = "Emission Distributions by Fuel Type",
    x = "Fuel Category",
    y = "Emissions (Tons - Log Scale)",
    fill = "Fuel"
  ) +
  theme(legend.position = "bottom")


# ==============================================================================
# 6. DATA PARTITIONING (70/30 TRAIN/TEST SPLIT)
# ==============================================================================
# Set random seed to guarantee reproducibility
set.seed(123)

# Randomly sample 70% of rows for training; hold out remaining 30% for testing
index <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
train_data <- data[index, ]
test_data  <- data[-index, ]


# ==============================================================================
# 7. MODEL TRAINING & PREDICTION
# ==============================================================================
# Fit Gaussian Naive Bayes classifier on training data using the four transformed emissions
model <- naiveBayes(as.factor(PLFUELCT) ~ PLNOXAN + PLCO2AN + PLSO2AN + PLCH4AN, data = train_data)

# Extract test features and generate class predictions
predict_data <- test_data[, c("PLFUELCT", "PLNOXAN", "PLCO2AN", "PLSO2AN", "PLCH4AN")]
predict <- predict(model, predict_data)


# ==============================================================================
# 8. PERFORMANCE EVALUATION & CONFUSION MATRIX HEATMAP
# ==============================================================================
# Compute confusion matrix and overall classification metrics
c_mat <- confusionMatrix(predict, as.factor(test_data$PLFUELCT))

# Convert frequency table to data frame for ggplot rendering
cm_df <- as.data.frame(c_mat$table)

# Plot confusion matrix heatmap showing predicted vs. actual reference labels
ggplot(data = cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +                                      # Tile boundaries
  scale_fill_gradient(low = "white", high = "red") +                # Frequency heat gradient
  geom_text(aes(label = Freq), size = 6) +                          # Frequency count labels
  labs(
    title = "4x4 Confusion Matrix: Fuel Category Classification",
    subtitle = paste0("Model Accuracy: ", round(c_mat$overall["Accuracy"] * 100, 2), "%"),
    x = "Actual Fuel Category (Reference)",
    y = "Predicted Fuel Category",
    fill = "Count"
  ) +
  theme(
    axis.text = element_text(size = 12, face = "bold"),
    title = element_text(size = 14),
    plot.subtitle = element_text(margin = margin(b = 10))
  )