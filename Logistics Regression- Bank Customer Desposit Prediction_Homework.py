# Databricks notebook source
# MAGIC %md 
# MAGIC ## Nick Curci
# MAGIC ### Homework
# MAGIC 
# MAGIC #### Due Date: Thursday 4/16 at the start of the class.
# MAGIC 
# MAGIC - Please try to improve areaUnderROC value for this deposit prediction model. Please document what changes you have made, what is the best model/algorithms you found and new areaUnderROC value.
# MAGIC 
# MAGIC - please export the completed notebook as HTML, zip it and submit it on Blackboard by due date.

# COMMAND ----------

# MAGIC %md ##Predict whether a bank customer will subscribe to a term deposit
# MAGIC The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe (Yes/No) to a term deposit. The dataset can be downloaded from [Kaggle](http://www.kaggle.com/rouseguy/bankbalanced).
# MAGIC 
# MAGIC [Attribute Information](http://archive.ics.uci.edu/ml/datasets/bank+marketing)
# MAGIC 
# MAGIC Input variables:
# MAGIC ##### bank client data:
# MAGIC 1. - age (numeric)
# MAGIC 2. - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# MAGIC 3. - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# MAGIC 4. - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# MAGIC 5. - default: has credit in default? (categorical: 'no','yes','unknown')
# MAGIC 6. - housing: has housing loan? (categorical: 'no','yes','unknown')
# MAGIC 7. - loan: has personal loan? (categorical: 'no','yes','unknown')
# MAGIC #### related with the last contact of the current campaign:
# MAGIC 8. - contact: contact communication type (categorical: 'cellular','telephone')
# MAGIC 9. - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# MAGIC 10. - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# MAGIC 11. - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# MAGIC #### other attributes:
# MAGIC 12. - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# MAGIC 13. - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# MAGIC 14. - previous: number of contacts performed before this campaign and for this client (numeric)
# MAGIC 15. - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

# COMMAND ----------

# MAGIC %fs ls /mnt/classdata/bank

# COMMAND ----------

#read bank data
bank = spark.read.csv('/mnt/classdata/bank/bank.csv', header = True, inferSchema = True)
bank.printSchema()

# COMMAND ----------

display(bank)

# COMMAND ----------

bank.describe('age', 'balance','duration', 'pdays').show()

# COMMAND ----------

# MAGIC %md ###Feature Engineering with Transformers
# MAGIC 
# MAGIC [RFormula](https://spark.apache.org/docs/latest/ml-features#rformula)

# COMMAND ----------

from pyspark.ml.feature import RFormula

bank_rf = RFormula(formula="deposit ~ .")

# COMMAND ----------

# MAGIC %md ####Prepare (Train and transform) our data frame

# COMMAND ----------

fittedRF = bank_rf.fit(bank)

preparedDF = fittedRF.transform(bank)

preparedDF.select('deposit', 'label', 'features').show(10, False)


# COMMAND ----------

# MAGIC %md split our data into train and test set

# COMMAND ----------

# MAGIC %md
# MAGIC # FIRST CHANGE, CHANGED THE SPLITS TO 90-10

# COMMAND ----------

(train, test) = preparedDF.randomSplit([0.9, 0.1], seed=100)


# COMMAND ----------

train.count()
test.count()

# COMMAND ----------

# MAGIC %md ####Instantiate an instance of LogisticsRegression, set the label columns and the feature columns

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
#lr = LogisticRegression(labelCol="label",featuresCol="features")
lr= LogisticRegression()

# COMMAND ----------

# MAGIC %md ### Train the logistics regression model based on training dataset

# COMMAND ----------

lrModel = lr.fit(train)

# COMMAND ----------

# MAGIC %md ###Using the model making prediction

# COMMAND ----------

lrPrediction=lrModel.transform(test)
lrPrediction.select("label", "prediction").show(10, False)

# COMMAND ----------

# MAGIC %md calcualte accuracy

# COMMAND ----------

print("prediction accuracy is: ", lrPrediction.where("prediction==label").count()/lrPrediction.count())

tp=lrPrediction.where("label=1 and prediction=1").count()
fp=lrPrediction.where("label=0 and prediction=1").count()
tn=lrPrediction.where("label=0 and prediction=0").count()
fn=lrPrediction.where("label=1 and prediction=0").count()

print("true positive is: ", tp)

print("false positive is: ", fp)

print("true negative is: ", tn)

print("false negative is ", fn)

print("precision is ", tp/(tp+fp)) 

print("recall is ", tp/(tp+fn))


# COMMAND ----------

featureIndex=preparedDF.schema["features"].metadata["ml_attr"]["attrs"]
x=0
#print numberic feature
for x in range(len(lrModel.coefficients)-1):
  try:
    print("feature", featureIndex["numeric"][x]['idx'], " ", featureIndex["numeric"][x]['name'], ': ', lrModel.coefficients[x])
  except:
    continue

# print binary feature   
for x in range(len(lrModel.coefficients)-1):
  try:
    print("feature", featureIndex["binary"][x]['idx'], " ", featureIndex["binary"][x]['name'], ': ', lrModel.coefficients[x])
  except:
    continue

# COMMAND ----------

# MAGIC %md Evalaute model

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()\
  .setMetricName("areaUnderROC")\
  .setRawPredictionCol("prediction")\
  .setLabelCol("label")

# COMMAND ----------

# MAGIC %md
# MAGIC # NEW PREDICTION VALUE

# COMMAND ----------

evaluator.evaluate(lrPrediction)

# COMMAND ----------

# ROC for test data
display(lrModel, test, "ROC")

# COMMAND ----------

display(lrModel.summary.roc)

# COMMAND ----------

display(lrModel.summary.pr)

# COMMAND ----------

# MAGIC %md ## Decision Trees
# MAGIC 
# MAGIC You can read more about [Decision Trees](http://spark.apache.org/docs/latest/mllib-decision-tree.html) in the Spark MLLib Programming Guide.
# MAGIC The Decision Trees algorithm is popular because it handles categorical
# MAGIC data and works out of the box with multiclass classification tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC # SECOND CHANGE, USED MAX DEPTH OF 6 FOR THE DECISION TREE

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=6)

# Train model with Training Data
dtModel = dt.fit(train)

# COMMAND ----------

display(dtModel)

# COMMAND ----------

featureIndex=preparedDF.schema["features"].metadata["ml_attr"]["attrs"]

print(featureIndex)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
dtPrediction = dtModel.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(dtPrediction)

# COMMAND ----------

# MAGIC %md ## Random Forest
# MAGIC 
# MAGIC Random Forests uses an ensemble of trees to improve model accuracy.
# MAGIC You can read more about [Random Forest] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC 
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Random Forest]: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forests

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(train)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
rfPrediction = rfModel.transform(test)

# COMMAND ----------

# MAGIC %md We will evaluate our Random Forest model with BinaryClassificationEvaluator.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(rfPrediction)

# COMMAND ----------

print("prediction accuracy is: ", rfPrediction.where("prediction==label").count()/rfPrediction.count())

tp=lrPrediction.where("label=1 and prediction=1").count()
fp=lrPrediction.where("label=0 and prediction=1").count()
tn=lrPrediction.where("label=0 and prediction=0").count()
fn=lrPrediction.where("label=1 and prediction=0").count()

print("true positive is: ", tp)

print("false positive is: ", fp)

print("true negative is: ", tn)

print("false negative is ", fn)

print("precision is ", tp/(tp+fp)) 

print("recall is ", tp/(tp+fn))


# COMMAND ----------

# MAGIC %md ### Gradient-Boosted Trees

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
gbtClassifier=GBTClassifier()

gbtModel=gbtClassifier.fit(train)

# Make predictions on test data using the Transformer.transform() method.
gbtPrediction = gbtModel.transform(test)

# Evaluate Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(gbtPrediction)

# COMMAND ----------

print("prediction accuracy is: ", gbtPrediction.where("prediction==label").count()/gbtPrediction.count())

tp=lrPrediction.where("label=1 and prediction=1").count()
fp=lrPrediction.where("label=0 and prediction=1").count()
tn=lrPrediction.where("label=0 and prediction=0").count()
fn=lrPrediction.where("label=1 and prediction=0").count()

print("true positive is: ", tp)

print("false positive is: ", fp)

print("true negative is: ", tn)

print("false negative is ", fn)

print("precision is ", tp/(tp+fp))

print("recall is ", tp/(tp+fn))

# COMMAND ----------

# MAGIC %md ###Make Predictions
# MAGIC 
# MAGIC As Gradient-boosted Trees model gives us the best areaUnderROC value, we will use the bestModel obtained from Gradient-Boosted Tree model for deployment, and use it to generate predictions on new data. In this example, we will simulate this by generating predictions on the entire dataset.

# COMMAND ----------

# Generate predictions for entire dataset
finalPredictions = gbtModel.transform(preparedDF)

# COMMAND ----------

# Evaluate best model
evaluator.evaluate(finalPredictions)

# COMMAND ----------

# create a SQL view
finalPredictions.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %md In an operational environment, analysts may use a similar machine learning pipeline to obtain predictions on new data, organize it into a table and use it for analysis or lead targeting.

# COMMAND ----------

# MAGIC %sql
# MAGIC select case when label=1 and prediction=1 then "true positive"
# MAGIC             when label=1 and prediction=0 then "false negative"
# MAGIC             when label=0 and prediction=0 then "true negative"
# MAGIC             when label=0 and prediction=1 then "false positive"
# MAGIC             else "N/A"
# MAGIC             End as status, count(*) as NumberofRecords
# MAGIC from finalPredictions 
# MAGIC group by status

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # END
# MAGIC ## The training and testing datasets were modified to a 9:1 ratio, the ROC value increased by over .1, the decision tree was changed to allow a max depth of 6 rather than the original 3 and this increase gave us a better model. I think that the decision tree classifier is the best of the classifiers because it allows for direct manipulation of the depth of search
