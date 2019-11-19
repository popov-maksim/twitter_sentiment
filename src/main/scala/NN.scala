package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame

class NN(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred

  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  private val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setVocabSize(3000)


  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
  val layers = Array[Int](3000, 200, 40, 2)

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)
    .setBlockSize(16)
    .setMaxIter(10)

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, cv, trainer))



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



