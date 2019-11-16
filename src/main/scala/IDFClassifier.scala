package src.main.scala
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.DataFrame

class IDFClassifier(target: String, pred: String, data: DataFrame) {

  val labelColumn: String = target
  val predictionColumn: String = pred


  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
  val wordsData = tokenizer.transform(data)

  val hashingTF = new HashingTF()
    .setInputCol("words").setOutputCol("rawFeatures")

  val featurizedData = hashingTF.transform(wordsData)
  // alternatively, CountVectorizer can also be used to get term frequency vectors

  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)

  val rescaledData = idfModel.transform(featurizedData)
  featurizedData.select("label", "rawFeatures").show()

}
