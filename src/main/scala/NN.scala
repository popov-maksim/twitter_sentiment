package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame

class NN(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred

  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  private val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
    .setNumFeatures(1000)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
  val layers = Array[Int](hashingTF.getNumFeatures, 100, 50, 10)

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)
    .setBlockSize(16)
    .setMaxIter(10)

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, trainer))



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

