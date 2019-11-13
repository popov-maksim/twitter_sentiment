import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame

class LogisticRegressionClassifier(train: DataFrame, labelColumn: String, predictionColumn: String) {

  // initialization
  val trainDF: DataFrame = train
  val target: String = labelColumn
  val pred: String = predictionColumn
  val modelName: String = "LogisticRegression"

  def train(saveModel: Boolean = true): PipelineModel = {
    // describing preprocessing steps
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setPredictionCol(predictionColumn)
      .setMaxIter(10)

    val stages = Array(
      tokenizer,
      hashingTF,
      lr
    )

    // creating a pipeline
    val pipeline = new Pipeline().setStages(stages)

    // fitting the model
    val model: PipelineModel = pipeline.fit(trainDF)

    if (saveModel) {
      model.write.overwrite().save("trained_model")
    }

    model
  }

}
