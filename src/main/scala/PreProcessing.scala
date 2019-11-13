package src.main.scala

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.util.CoreMap

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StopWordsRemover

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer


object PreProcessing {
  def lowerCase(dataset: Dataset[Row], column: String): Dataset[Row] = {
    dataset.withColumn(column, regexp_replace(lower(dataset(column)), "[^a-zA-Z0-9 ]", ""));
  }

  def isLetter(str: String) = str.forall(char => Character.isLetter(char));

  def lemmatize(text: String, pipeline: StanfordCoreNLP): Seq[String] ={

    val annotation = new Annotation(text);
    pipeline.annotate(annotation);
    val lemma = new ArrayBuffer[String]();
    val sentences = annotation.get(classOf[SentencesAnnotation]);
    for (sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])){
      val lem = token.get(classOf[LemmaAnnotation]);
      if (lemma.length > 2 && isLetter(lem)){
        lemma += lem.toLowerCase
      }
    }
    lemma
  }


  def stopWord(dataset:Dataset[Row], inColumn: String, outColumn:String): Dataset[Row] = {
    val stopWordsRemover = new StopWordsRemover()
                                .setInputCol(inColumn)
                                .setOutputCol(outColumn);
  stopWordsRemover.transform(dataset);
  }
}
