name := "sentiment_analysis"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.4.4" ,
  "org.apache.spark" % "spark-streaming_2.11" % "2.4.4" % "provided",
  "org.apache.spark" % "spark-sql_2.11" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)
