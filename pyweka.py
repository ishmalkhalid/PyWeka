import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter

#to start the JVM for weka
jvm.start(max_heap_size="1024m", packages=True)

# to load a dataset
loader = Loader(classname="weka.core.converters.CSVLoader")
data = loader.load_file("iris.csv")
data.class_is_last()

filtered = data
filtered.class_is_last()

#to identify and load a ML algorithm for classification, here random forests
classifier1 = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I", "100", "-K", "0", "-S", "1", "-num-slots", "1"])
print(classifier1.options)
evaluation = Evaluation(filtered)

#use 10 fold cross validation
evaluation.crossvalidate_model(classifier1, filtered, 10, Random(42))


classifier2 = Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")
print(classifier2.options)
evaluation2 = Evaluation(filtered)
evaluation2.crossvalidate_model(classifier2, filtered, 10, Random(42))


classifier3 = Classifier(classname="weka.classifiers.trees.J48")
print(classifier3.options)
evaluation3 = Evaluation(filtered)
evaluation3.crossvalidate_model(classifier3, filtered, 10, Random(42))

classifier4 = Classifier(classname="weka.classifiers.functions.SMO")
print(classifier4.options)
evaluation4 = Evaluation(filtered)
evaluation4.crossvalidate_model(classifier4, filtered, 10, Random(42))

classifier5 = Classifier(classname="weka.classifiers.meta.Bagging")
print(classifier5.options)
evaluation5 = Evaluation(filtered)
evaluation5.crossvalidate_model(classifier5, filtered, 10, Random(42))

classifier6 = Classifier(classname="weka.classifiers.rules.OneR")
print(classifier6.options)
evaluation6 = Evaluation(filtered)
evaluation6.crossvalidate_model(classifier6, filtered, 10, Random(42))

filtered2 = Filter("weka.filters.supervised.instance.Resample")
filtered2.inputformat(data)
# filtered = data
# filtered2.class_is_last()
filtered = filtered2.filter(data)


classifier7 = Classifier(classname="weka.classifiers.rules.JRip")
print(classifier7.options)
evaluation7 = Evaluation(filtered)
evaluation7.crossvalidate_model(classifier7, filtered, 10, Random(42))


# # #to retrieve the results and organize the representation
print("------------------------Random Forest: ------------")
print(evaluation.summary())
# print(evaluation.class_details())

print("------------------------Naive Bayes: ------------")
print(evaluation2.summary())
# print(evaluation2.class_details())

print("------------------------Trees J48: ------------")
print(evaluation3.summary())
# print(evaluation3.class_details())

print("------------------------Functions SMO: ------------")
print(evaluation4.summary())
# print(evaluation4.class_details())

print("------------------------Meta Bagging: ------------")
print(evaluation5.summary())
# print(evaluation4.class_details())

print("------------------------Rules OneR: ------------")
print(evaluation6.summary())
# print(evaluation4.class_details())

print("------------------------Rules JRip: ------------")
print(evaluation7.summary())
# print(evaluation4.class_details())

# # #to save the generated model
serialization.write("model", classifier)