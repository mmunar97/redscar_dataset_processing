library(rjson)
library(caret)
library(ROCR)
library(e1071)
library(rpart)

################################ RAW DATA READ ################################
train_redness_raw <- fromJSON(file = "train_redness_evaluation.json")
test_redness_raw <- fromJSON(file = "test_redness_evaluation.json")

############################### DATA PROCESSING ###############################
col_names = c("IMAGE_NAME","INFECTION", "NUMBER_COMPONENTS",
              "MAX_LIU_PROP", "MAX_SHAMIR_PROP", "MAX_CHAMORRO_PROP", "MAX_AMANTE_PROP",
              "MAX_LIU_CHR_PROP", "MAX_SHAMIR_CHR_PROP", "MAX_CHAMORRO_CHR_PROP", "MAX_AMANTE_CHR_PROP") 

train_redness = data.frame(matrix(nrow = 0, ncol = length(col_names))) 
colnames(train_redness) = col_names

test_redness = train_redness

processRawData <- function(rawData, dataframeToFill){
  for (i in 1:length(rawData)) {
    image_sample = rawData[[i]]
    
    if (image_sample$NUMBER_CONNECTED_COMPONENTS == 0) {
      dataframeToFill[nrow(dataframeToFill) + 1,] = c(image_sample$IMAGE_NAME, 
                                                      0,
                                                      image_sample$NUMBER_CONNECTED_COMPONENTS,
                                                      0, 0, 0, 0, 0, 0, 0, 0)
    } else {
      red_proportions_liu = c()
      red_proportions_shamir = c()
      red_proportions_chamorro = c()
      red_proportions_amante = c()
      red_proportions_liu_chr = c()
      red_proportions_shamir_chr = c()
      red_proportions_chamorro_chr = c()
      red_proportions_amante_chr = c()
      
      for (j in 1:image_sample$NUMBER_CONNECTED_COMPONENTS) {
        region_liu_prop = image_sample[paste("LIU_COLOR_REDPROP_region=",j, sep="")]
        region_shamir_prop = image_sample[paste("SHAMIR_COLOR_REDPROP_region=",j, sep="")]
        region_chamorro_prop = image_sample[paste("CHAMORRO_COLOR_REDPROP_region=",j, sep="")]
        region_amante_prop = image_sample[paste("AMANTE_COLOR_REDPROP_region=",j, sep="")]
        
        region_liu_chr_prop = image_sample[paste("LIU_COLOR_CHR_REDPROP_region=",j, sep="")]
        region_shamir_chr_prop = image_sample[paste("SHAMIR_COLOR_CHR_REDPROP_region=",j, sep="")]
        region_chamorro_chr_prop = image_sample[paste("CHAMORRO_COLOR_CHR_REDPROP_region=",j, sep="")]
        region_amante_chr_prop = image_sample[paste("AMANTE_COLOR_CHR_REDPROP_region=",j, sep="")]
        
        red_proportions_liu = c(red_proportions_liu, region_liu_prop)
        red_proportions_shamir = c(red_proportions_shamir, region_shamir_prop)
        red_proportions_chamorro = c(red_proportions_chamorro, region_chamorro_prop)
        red_proportions_amante = c(red_proportions_amante, region_amante_prop)
        
        red_proportions_liu_chr = c(red_proportions_liu_chr, region_liu_chr_prop)
        red_proportions_shamir_chr = c(red_proportions_shamir_chr, region_shamir_chr_prop)
        red_proportions_chamorro_chr = c(red_proportions_chamorro_chr, region_chamorro_chr_prop)
        red_proportions_amante_chr = c(red_proportions_amante_chr, region_amante_chr_prop)
      }
      
      dataframeToFill[nrow(dataframeToFill) + 1,] = c(image_sample$IMAGE_NAME, 
                                                      0,
                                                      image_sample$NUMBER_CONNECTED_COMPONENTS,
                                                      max(unlist(red_proportions_liu)),
                                                      max(unlist(red_proportions_shamir)),
                                                      max(unlist(red_proportions_chamorro)),
                                                      max(unlist(red_proportions_amante)),
                                                      max(unlist(red_proportions_liu_chr)),
                                                      max(unlist(red_proportions_shamir_chr)),
                                                      max(unlist(red_proportions_chamorro_chr)),
                                                      max(unlist(red_proportions_amante_chr)))
    }
  }
  dataframeToFill$INFECTION = grepl('infection=1', dataframeToFill$IMAGE_NAME)
  
  dataframeToFill$MAX_LIU_PROP = as.numeric(dataframeToFill$MAX_LIU_PROP)
  dataframeToFill$MAX_SHAMIR_PROP = as.numeric(dataframeToFill$MAX_SHAMIR_PROP)
  dataframeToFill$MAX_CHAMORRO_PROP = as.numeric(dataframeToFill$MAX_CHAMORRO_PROP)
  dataframeToFill$MAX_AMANTE_PROP = as.numeric(dataframeToFill$MAX_AMANTE_PROP)
  
  dataframeToFill$MAX_LIU_CHR_PROP = as.numeric(dataframeToFill$MAX_LIU_CHR_PROP)
  dataframeToFill$MAX_SHAMIR_CHR_PROP = as.numeric(dataframeToFill$MAX_SHAMIR_CHR_PROP)
  dataframeToFill$MAX_CHAMORRO_CHR_PROP = as.numeric(dataframeToFill$MAX_CHAMORRO_CHR_PROP)
  dataframeToFill$MAX_AMANTE_CHR_PROP = as.numeric(dataframeToFill$MAX_AMANTE_CHR_PROP)
  
  return(dataframeToFill)
}

train_redness = processRawData(train_redness_raw, train_redness)
test_redness = processRawData(test_redness_raw, test_redness)

################################## CLASSIFY ###################################

train_redness_subset = train_redness[, c(4,5,6,7)]
train_redness_subset_chr = train_redness[, c(8,9,10,11)]

test_redness_subset = test_redness[, c(4,5,6,7)]
test_redness_subset_chr = test_redness[, c(8,9,10,11)]

# # Only red proportions of methods considering achromatic colors with logistic classification
# 
# classification_model = glm(INFECTION~MAX_LIU_PROP+MAX_SHAMIR_PROP+MAX_CHAMORRO_PROP+MAX_AMANTE_PROP, data=train_redness, family=binomial)
# model_prediction = predict(classification_model,train_redness_subset, type="response")
#   
# roc_pred <- prediction(model_prediction, as.integer(train_redness$INFECTION))
# roc_performance <- performance(roc_pred, "tpr", "fpr")
# plot(roc_performance, colorize=TRUE, print.cutoffs.at=seq(0, 0.1, by=0.005), main="ROC curve - Logistic classification of all achromatic proportions")


# Only red proportions of methods considering chromatic colors with logistic classification

classification_model_chr = glm(INFECTION~MAX_LIU_CHR_PROP+MAX_SHAMIR_CHR_PROP+MAX_CHAMORRO_CHR_PROP+MAX_AMANTE_CHR_PROP, data=train_redness, family=binomial)
model_prediction_chr = predict(classification_model_chr,train_redness_subset_chr, type="response")

roc_pred_chr <- prediction(model_prediction_chr, as.integer(train_redness$INFECTION))
roc_performance_chr <- performance(roc_pred_chr, "tpr", "fpr")
par(cex.axis=1.3)
plot(roc_performance_chr, 
     print.cutoffs.at=seq(0.03, 0.1, by=0.01), 
     points.pch = 20, points.col = "darkblue", points.cex=2,
     text.adj=c(1.2,-0.5),
     main="ROC curve - Logistic classification of chromatic red proportions", 
     cex.main=1.5, cex.lab=1.3)


# # Only red proportions of methods considering achromatic colors with SVM
# 
# svm_model = svm(INFECTION~MAX_LIU_PROP+MAX_SHAMIR_PROP+MAX_CHAMORRO_PROP+MAX_AMANTE_PROP, data=train_redness, type = 'C-classification')
# svm_model_prediction = predict(svm_model,train_redness_subset)
# 
# svm_roc_pred <- prediction(as.numeric(svm_model_prediction), as.integer(train_redness$INFECTION))
# svm_roc_performance <- performance(svm_roc_pred, "tpr", "fpr")
# plot(svm_roc_performance, colorize=TRUE, print.cutoffs.at=seq(0, 0.1, by=0.01), main="ROC curve - SVM classification of all achromatic proportions")
# 
# # Only red proportions of methods considering achromatic colors with decision tree
# 
# dt_model = rpart(INFECTION~MAX_LIU_PROP+MAX_SHAMIR_PROP+MAX_CHAMORRO_PROP+MAX_AMANTE_PROP, data=train_redness, method = 'class')
# dt_model_prediction = predict(dt_model, train_redness_subset, type='class')
# 
# dt_roc_pred <- prediction(as.numeric(dt_model_prediction), as.integer(train_redness$INFECTION))
# dt_roc_performance <- performance(dt_roc_pred, "tpr", "fpr")
# plot(dt_roc_performance, colorize=TRUE, print.cutoffs.at=seq(0, 0.1, by=0.01), main="ROC curve - Decision Tree classification of all achromatic proportions")



# It seems that the logistic classification eliminating the achromatic colours, and therefore considering only 
# the chromatic ones, is the one that offers the best classification. According to the ROC curve, it is the 0.08 
# threshold that offers the best classification. 
logistic_classification_train = factor(as.integer(model_prediction_chr >= 0.08), levels=c(0,1))
gt_train = factor(as.integer(train_redness$INFECTION), levels=c(0,1))
logistic_confusion_matrix_train <- confusionMatrix(data=logistic_classification_train, reference = gt_train)

true_negatives_train  = logistic_confusion_matrix_train$table[1,1]
false_negatives_train = logistic_confusion_matrix_train$table[1,2]
false_positives_train = logistic_confusion_matrix_train$table[2,1]
true_positives_train  = logistic_confusion_matrix_train$table[2,2]

tpr_train = true_positives_train/(true_positives_train+false_negatives_train)
fpr_train = false_positives_train/(false_positives_train+true_negatives_train)
tnr_train = true_negatives_train/(true_negatives_train+false_positives_train)
balanced_accuracy_train = (tpr_train+tnr_train)/2


model_prediction_chr_test = predict(classification_model_chr, test_redness_subset_chr, type="response")
logistic_classification_test = factor(as.integer(model_prediction_chr_test >= 0.08), levels=c(0,1))
gt_test = factor(as.integer(test_redness$INFECTION), levels=c(0,1))
logistic_confusion_matrix_test <- confusionMatrix(data=logistic_classification_test, reference = gt_test)

true_negatives_test  = logistic_confusion_matrix_test$table[1,1]
false_negatives_test = logistic_confusion_matrix_test$table[1,2]
false_positives_test = logistic_confusion_matrix_test$table[2,1]
true_positives_test  = logistic_confusion_matrix_test$table[2,2]

tpr_test = true_positives_test/(true_positives_test+false_negatives_test)
fpr_test = false_positives_test/(false_positives_test+true_negatives_test)
tnr_test = true_negatives_test/(true_negatives_test+false_positives_test)
balanced_accuracy_test = (tpr_test+tnr_test)/2


# Determine the images that have infection but have not been detected, both in the train and test set.
false_negatives_indices_train = (logistic_classification_train==0) & (gt_train==1)
false_negatives_samples_train = train_redness[false_negatives_indices_train,]
