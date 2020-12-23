#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql.types import BooleanType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from helpers.helper_functions import translate_to_file_string
from pyspark.sql import DataFrameReader
from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, Normalizer, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from helpers.helper_functions import translate_to_file_string
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[2]:


inputFile = "hdfs:///data/heart_val.csv"


# In[3]:


spark = (SparkSession
       .builder
       .appName("HeartDiseaseAnalRf")
       .master("yarn")
       .getOrCreate())


# In[4]:


# load data file.
# create a DataFrame using an ifered Schema 
df = spark.read.option("header", "true")        .option("inferSchema", "true")        .option("delimiter", ";")        .csv(inputFile)


# In[5]:


#remove the outliner
df_filtered=df.filter(df.age > 30)


# In[6]:


#transform labels
labelIndexer = StringIndexer().setInputCol("target").setOutputCol("label").fit(df)
sexIndexer = StringIndexer().setInputCol("sex").setOutputCol("sex_num").fit(df)


# In[7]:


#feature columns
featureCols = df.columns.copy()
featureCols.remove("target")
featureCols.remove("sex")
featureCols = featureCols + ["sex_num"]


# In[8]:


#vector assembler of all features
assembler =  VectorAssembler(outputCol="features", inputCols=featureCols)


# In[9]:


#Build feauture Indexer 
featureIndexer = VectorIndexer(inputCol="features",outputCol="indexedFeatures", maxCategories=6)


# In[10]:


#Convert Indexed labels back to original labels
predConverter = IndexToString(inputCol="prediction",outputCol="predictedLabel",labels=labelIndexer.labels)


# In[11]:


#create the Random Forest Classification
rf = RandomForestClassifier(labelCol="label", featuresCol="features",impurity="gini",                  minInstancesPerNode=1, featureSubsetStrategy='sqrt', subsamplingRate=0.95, seed= 12345)


# In[12]:


# build a network para grip
paramGrid = (ParamGridBuilder()
             #.addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
               .addGrid(rf.maxDepth, [2, 5, 10])
             #.addGrid(rf.maxBins, [10, 20, 40, 80, 100])
               .addGrid(rf.maxBins, [5, 10, 20, 30])
             #.addGrid(rf.numTrees, [5, 20, 50, 100, 500])
               .addGrid(rf.numTrees, [5, 20, 50])
             .build())


# In[13]:


#split data for testing

splits = df.randomSplit([0.6, 0.4 ], 5756)
train = splits[0]
test = splits[1]


# In[14]:


#Pipelining of all steps 
pipeline = Pipeline(stages= [labelIndexer,sexIndexer,  assembler, featureIndexer, rf , predConverter])


# In[16]:


#build evaluator 
evaluator =  BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction", metricName="areaUnderROC")
#evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") #using different evaluators


# In[17]:


#Cross validator
cvRf = CrossValidator(estimator=pipeline, evaluator=evaluator,estimatorParamMaps=paramGrid,numFolds=5, parallelism=4)


# In[18]:


#train model
rfModel = cvRf.fit(train)


# In[19]:


#Find out the best model
rfBestModel = rfModel.bestModel.stages[4] # the stage at index 1 in the pipeline is the SVMModel
print("Best Params: \n", rfBestModel.explainParams())
print("Param Map: \n", rfBestModel.extractParamMap())
#print(cvSVMModel.getEstimatorParamMaps()[np.argmax(cvSVMModel.avgMetrics)])


# In[20]:


#test model
predictions = rfModel.transform(test)
predictions.show()


# In[21]:


accuracy = evaluator.evaluate(predictions)
print("Test Error = " ,(1.0 - accuracy))


# In[22]:


spark.stop()

