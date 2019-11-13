package src.main.scala

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}


object MainClass {

  // specify here dataset used for training
  val dataset: String = "train.csv" // change for HDFS
  val ipAddress: String = "10.91.66.168"
  val port: Int = 8989
  val pathToModel: String = "trained_model" // change for HDFS

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("KusalMaxDanMishaTwitter").setMaster("local") // change for running on Cluster
    val spark = SparkSession.builder().config(conf).getOrCreate()
    //val ssc = new StreamingContext(conf, Seconds(1))

    val isTrain: Boolean = whichMode(args)

    if (isTrain) {
      println("Train mode")

      val df: DataFrame = loadData(spark)

      val labelColumn: String = "label"
      val predictionColumn: String = "predicted column"

      val clf1 = new LogisticRegressionClassifier(labelColumn, predictionColumn)

      val model: CrossValidatorModel = clf1.crossValidate(df)

      val f1: Double = model.getEvaluator
        .evaluate(model
          .transform(df)
          .select(labelColumn, predictionColumn))

      // TODO choose the best one based on the metric

      println("F1 score on train: " + f1)

      model.write.overwrite().save(pathToModel)

    } else {
      println("Ready to listen to Twitter. To train a model first use param [train].")
      val model = CrossValidatorModel.load(pathToModel)
      // TODO transformations on twits and run through model
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

  // as a dataset for training there is used Twitter Sentiment (https://www.kaggle.com/c/twitter-sentiment-analysis2/data)
  def loadData(spark: SparkSession): DataFrame = {

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

}
