package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class LogisticRegressionClassifier(target: String, pred: String) {

  val labelColumn: String = target
  val predictionColumn: String = pred

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
    .setMaxIter(10)

  private val stages = Array(
    tokenizer,
    hashingTF,
    lr
  )

  // defining an evaluator for cross validation
  private val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(labelColumn)
    .setPredictionCol(predictionColumn)
    .setMetricName("f1")

  // creating a pipeline
  private val pipeline = new Pipeline().setStages(stages)

  // get model with most suitable params
  def crossValidate(df: DataFrame): CrossValidatorModel = {

    // cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(2)

    // fitting the model
    val model: CrossValidatorModel = cv.fit(df)

    model

  }

}
