package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.DataFrame

class GBTree(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred

  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  private val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setNumFeatures(30000)
    .setOutputCol("features")

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // create the trainer and set its parameters
  val gbt = new DecisionTreeClassifier()
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, gbt))



  // train the model
  val model = pipeline.fit(trainingData)
  println("done")

  private val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(labelColumn)
    .setPredictionCol(predictionColumn)
    .setMetricName("accuracy")


  println(s"Test set accuracy = ${evaluator.evaluate(model
    .transform(testData)
    .select(labelColumn, predictionColumn))}")

}

