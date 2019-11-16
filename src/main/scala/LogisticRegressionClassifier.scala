package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame


class LogisticRegressionClassifier(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred
  val pathToModel: String = "trained_model"

  // describing preprocessing steps
  private val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

  private val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")

  private val lr = new LogisticRegression()
    .setLabelCol(labelColumn)
    .setFeaturesCol("features")
    .setPredictionCol(predictionColumn)
    .setElasticNetParam(0.8)
    .setRegParam(0.001)
    .setMaxIter(10000)

  private val stages = Array(
    tokenizer,
    hashingTF,
    lr
  )

  // defining an evaluator for cross validation
  private val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(labelColumn)
    .setPredictionCol(predictionColumn)
    .setMetricName("accuracy")

  // creating a pipeline
  private val pipeline = new Pipeline().setStages(stages)

  // get model with most suitable params
  val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))
  val model = pipeline.fit(trainingData)

  println(s"Test set accuracy = ${evaluator.evaluate(model
    .transform(testData)
    .select(labelColumn, predictionColumn))}")

  model.write.overwrite().save(pathToModel)
//  def crossValidate(df: DataFrame): CrossValidatorModel = {
//
//    // cross validation
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(hashingTF.numFeatures, Array(10, 100, 100000))
//      .addGrid(lr.regParam, Array(0.1, 0.01))
//      .build()
//
//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(evaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(3)
//      .setParallelism(2)
//
//    // fitting the model
//    val model: CrossValidatorModel = cv.fit(df)
//
//    model
//
//  }

}
