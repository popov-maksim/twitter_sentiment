package src.main.scala

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SaveMode
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext, Time}


object MainClass {
  // specify here dataset used for training
  val dataset: String = "train.csv"

  // for streaming
  val ipAddress: String = "10.90.138.32"
  val port: Int = 8989

  // basic paths
  val pathToModel: String = "trained_model"
  val checkpointDirectory: String = "checkpoint"

  // some constants
  val trainMode = true
  val testMode = false

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("KusalMaxDanMishaTwitter")//.setMaster("local") // change for running on Cluster
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext

    val (mode, predOut) = parseArgs(args)

    val labelColumn: String = "label"
    val predictionColumn: String = "predicted column"

    if (mode == trainMode) {
      println("Train mode")

      val df: DataFrame = loadData(spark)

      val clf1 = new LogisticRegressionClassifier(labelColumn, predictionColumn, df)
    } else {
      println("Ready to listen to Twitter. To train a model first use param [train].")

      // Function to create and setup a new StreamingContext
      val ssc = new StreamingContext(sc, Seconds(1))
      ssc.checkpoint(checkpointDirectory)

      val model = PipelineModel.load(pathToModel)

      val twits = ssc.socketTextStream(ipAddress, port)
      twits.print()

      // counting words
      twits.flatMap(_.split(" "))
          .map(word => (word.toLowerCase(), 1))
          .updateStateByKey(updateFunction _)
          .transform(rdd => rdd.map(x => (x._2, x._1))
            .sortByKey(ascending = false)
            .map(x => (x._2, x._1)))
          .print()

      // inferencing sentiment of a twit
      twits.foreachRDD { (rdd: RDD[String], time: Time) =>
        if (!rdd.isEmpty())
        {
          import spark.implicits._

          val input: String = rdd.first()

          val data = Seq((1.0, input))
          val df = data.toDF(labelColumn, "text")
          val prediction = model.transform(df)
            .select(predictionColumn)
            .first().toString()

          // write result to file
          Seq((time.toString(), input, prediction))
            .toDF("Time", "Text", "Sentiment")
            .coalesce(1)
            .write
            .mode(SaveMode.Append)
            .csv(predOut)
        }
      }

      ssc.start()
      ssc.awaitTermination()
    }
  }

  def parseArgs(args: Array[String]): (Boolean, String) = {
    if (args.length < 1) {
      println("set up params <mode> <filename for predictions>")

      (true, "")
    }
    else {
      val mode = getMode(args(0))

      val defaultPredOut = "pred_out.csv"

      if (args.length < 2) {
        (mode, defaultPredOut)
      }
      else {
        (mode, args(1))
      }
    }
  }

  // getting mode from given args
  def getMode(modeArg: String): Boolean = modeArg.toLowerCase match {
    case "train" => true
    case _ => false
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

  def updateFunction(newValues: Seq[Int], runningCount: Option[Int]): Option[Int] = {
    val newCount: Int = newValues.sum + runningCount.getOrElse(0)
    Some(newCount)
  }
}