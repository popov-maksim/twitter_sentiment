package src.main.scala

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext, Time}


object MainClass {

  // specify here dataset used for training
  val dataset: String = "train.csv"

  val ipAddress: String = "10.91.66.168"
  val port: Int = 8989

  val pathToModel: String = "trained_model"

  val trainMode = true
  val testMode = false

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("KusalMaxDanMishaTwitter").setMaster("local") // change for running on Cluster
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext
    val ssc = new StreamingContext(sc, Seconds(1))

    val (mode, wordOut, predOut) = parseArgs(args)

    val labelColumn: String = "label"
    val predictionColumn: String = "predicted column"

    if (mode == trainMode) {
      println("Train mode")

      val df: DataFrame = loadData(spark)

      val clf1 = new LogisticRegressionClassifier(labelColumn, predictionColumn, df)



    } else {
      println("Ready to listen to Twitter. To train a model first use param [train].")

      val model = CrossValidatorModel.load(pathToModel)

      val twits = ssc.socketTextStream(ipAddress, port)

      // counting words for
      twits.flatMap(_.split(" "))
          .map(word => (word, 1))
          .reduceByKey(_ + _)
          .map(x => (x._2, x._1))
          .foreachRDD{ (rdd: RDD[(Int, String)], _) =>
            rdd.sortByKey()
              .map(x => (x._1, x._2))
              .saveAsTextFile(wordOut) // write result to file
          }

      // inferencing sentiment of a twit
      twits.foreachRDD { (rdd: RDD[String], time: Time) =>
        import spark.implicits._

        val input = rdd.toString()

        println("Twit: " + input)

        val data = Seq((1.0, input))
        val df = sc.parallelize(data).toDF(labelColumn, "text")
        val prediction = model.transform(df)
          .select(predictionColumn)
          .first()

        // write result to file
        sc.parallelize(Seq((time, input, prediction)))
          .saveAsTextFile(predOut)

      }

      // don't know whether it is needed
      ssc.start()
      ssc.awaitTermination()

    }

  }

  def parseArgs(args: Array[String]): (Boolean, String, String) = {

    if (args.length < 1) {
      println("set up params <mode> <filename for word counts> <filename for predictions>")

      (true, "", "")

    }
    else {

      val mode = getMode(args(0))

      val defaultWordOut = "word_count"
      val defaultPredOut = "pred_out.csv"

      if (args.length < 2) {
        (mode, defaultWordOut, defaultPredOut)
      }
      else if (args.length < 3) {
        (mode, args(1), defaultPredOut)
      }else {
        (mode, args(1), args(2))
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

}