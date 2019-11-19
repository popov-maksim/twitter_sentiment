package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, Tokenizer}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.DataFrame

class Tree(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred

  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  private val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setMinDF(30)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

  // create the trainer and set its parameters
  val gbt = new DecisionTreeClassifier()
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, cv, gbt))



  // train the model
  val model = pipeline.fit(trainingData)
  println("done")

  private val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(labelColumn)
    .setPredictionCol(predictionColumn)
    .setMetricName("accuracy")


  println(s"Test set accuracy = ${evaluator.evaluate(model
    .transform(trainingData)
    .select(labelColumn, predictionColumn))}")

}