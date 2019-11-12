import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object MainClass {

  // specify here dataset used for training
  val dataset: String = "train.csv"

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("KusalMaxDanMishaTwitter").setMaster("local")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val ssc = new StreamingContext(conf, Seconds(1))

    val isTrain: Boolean = whichMode(args)

    if (isTrain) {
      println("Train mode")

      val df: DataFrame = loadData(spark)

      // split data on train and test
      val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

      val labelColumn: String = "label"
      val model: PipelineModel = train(trainingData, labelColumn)
      val f1: Double = validate(testData, model, labelColumn)

      println("F1 score on test: " + f1)

    } else {
      println("Ready to listen to Twitter")
    }

  }

  // getting mode from given args
  def whichMode(args: Array[String]): Boolean = {
    if (args.length < 1) {
      false
    } else {
      val mode = args(0).toLowerCase()
      if (mode == "train") true else false
    }
  }

  def loadData(spark: SparkSession): DataFrame ={
    // defining a schema for the DataFrame
    val schemaStruct = StructType(
        StructField("id", IntegerType) ::
        StructField("label", DoubleType) ::
        StructField("original text", StringType) :: Nil
    )

    // reading data
    spark.read
      .option("header", value = true)
      .schema(schemaStruct)
      .csv(dataset)
      .withColumn("text", trim(col("original text")))
      .drop("id")
      .drop("original text")
      .na.drop()
  }

  def train(trainDF: DataFrame, labelColumn: String, saveModel: Boolean = true): PipelineModel = {
    // describing preprocessing steps
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val gbt = new GBTClassifier()
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelColumn)
      .setMaxIter(50)

    val stages = Array(
      tokenizer,
      hashingTF,
      gbt
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

  def validate(testDF: DataFrame, model: PipelineModel, labelColumn: String): Double = {
    //We'll make predictions using the model and the test data
    val predictions = model.transform(testDF)

    //This will evaluate the error/deviation of the regression using the Root Mean Squared deviation
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setRawPredictionCol("Predicted " + labelColumn)
      .setMetricName("fMeasure")

    evaluator.evaluate(predictions)

  }

  def predict(): Unit = {

  }

}
