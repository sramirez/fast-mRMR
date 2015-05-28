package examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import java.util.ArrayList
import org.apache.spark.mllib.feature._

object FStest {

  def main(args: Array[String]): Unit = {
  	val initStartTime = System.nanoTime()

	val conf = new SparkConf().setAppName("FS test")
	val sc = new SparkContext(conf)
	val raw = MLUtils.loadLibSVMFile(sc, "a2a.libsvm")
  val data = raw.map({case LabeledPoint (label, values) => LabeledPoint(label, Vectors.dense(values.toArray))})
	val selector = MrmrSelector.train(data)    
	val redData = data.map { lp => 
      		LabeledPoint(lp.label, selector.transform(lp.features)) 
    	} 
    
    	println(redData.first().toString())
  }

}
