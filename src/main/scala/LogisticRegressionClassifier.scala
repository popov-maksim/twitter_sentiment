package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.DataFrame


class LogisticRegressionClassifier(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred
  val pathToModel: String = "trained_model"

  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

//  private val hashingTF = new HashingTF()
//    .setInputCol(tokenizer.getOutputCol)
//    .setOutputCol("features")

//  private val ngram = new NGram()
//    .setN(2).setInputCol("filtered").
//    setOutputCol("ngrams")

//  val remover = new StopWordsRemover()
//    .setInputCol("words")
//    .setOutputCol("filtered")

  private val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setMinDF(10)



  private val lr = new LogisticRegression()
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)
    .setElasticNetParam(0.8)
    .setRegParam(0.001)
    .setMaxIter(10000)

  private val stages = Array(
    tokenizer,
//    hashingTF,
//    remover,
//    ngram,
    cv,
    lr
  )

  // defining an evaluator for cross validation
  private val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(labelColumn)
    .setPredictionCol(predictionColumn)
    .setMetricName("accuracy")

  // creating a pipeline
  private val pipeline = new Pipeline().setStages(stages)

  val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))
  val model = pipeline.fit(trainingData)

  println(s"Test set accuracy = ${evaluator.evaluate(model
    .transform(testData)
    .select(labelColumn, predictionColumn))}")

  model.write.overwrite().save(pathToModel)


}
