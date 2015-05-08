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
	val rawdata = MLUtils.loadLibSVMFile(sc, "a2a.libsvm")
	val data = rawdata.map{ lp => 
		val newclass = if(lp.label == -1.0) 0 else 1
		new LabeledPoint(newclass, lp.features)
	}
    
    	val criterion = new MrmrSelector()
    	val selector = InfoThSelector.train(criterion, data)
    
    	val redData = data.map { lp => 
      		LabeledPoint(lp.label, selector.transform(lp.features)) 
    	} 
    
    	println(redData.first().toString())
  }

}
