/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import breeze.linalg._
import breeze.numerics._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.SparkException

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {
  
  private var classCol: Array[Byte] = null
  private var marginalProb: RDD[(Int, BDV[Float])] = null
  private var jointProb: RDD[(Int, BDM[Float])] = null
  
  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   * @param n Number of elements
   * 
   */
  private[feature] def entropy(freqs: Seq[Long], n: Long) = {
    freqs.aggregate(0.0)({ case (h, q) =>
      h + (if (q == 0) 0  else (q.toDouble / n) * (math.log(q.toDouble / n) / math.log(2)))
    }, { case (h1, h2) => h1 + h2 }) * -1
  }

  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   */
  private[feature] def entropy(freqs: Seq[Long]): Double = {
    entropy(freqs, freqs.reduce(_ + _))
  }  
  
  /**
   * Method that calculates mutual information (MI) and conditional mutual information (CMI) 
   * simultaneously for several variables. Indexes must be disjoint.
   *
   * @param data RDD of data (first element is the class attribute)
   * @param varX Indexes of primary variables (must be disjoint with Y and Z)
   * @param varY Indexes of secondary variable (must be disjoint with X and Z)
   * @param nInstances    Number of instances
   * @param nFeatures Number of features (including output ones)
   * @return  RDD of (primary var, (MI, CMI))
   * 
   */
  def computeMI(
      rawData: RDD[(Long, Byte)],
      varX: Seq[Int],
      varY: Int,
      nInstances: Long,      
      nFeatures: Int,
      counter: Map[Int, Int]) = {
    
    // Pre-requisites
    require(varX.size > 0)

    // Broadcast variables
    val sc = rawData.context
    val label = nFeatures - 1
    // A boolean vector that indicates the variables involved on this computation
    val fselected = Array.ofDim[Boolean](nFeatures)
    fselected(varY) = true // output feature
    varX.map(fselected(_) = true)
    val bFeatSelected = sc.broadcast(fselected)
    val getFeat = (k: Long) => (k % nFeatures).toInt
    // Filter data by these variables
    val data = rawData.filter({ case (k, _) => bFeatSelected.value(getFeat(k))})
     
    // Broadcast Y vector
    val yCol: Array[Byte] = if(varY == label){
	    // classCol corresponds with output attribute, which is re-used in the iteration 
      classCol = data.filter({ case (k, _) => getFeat(k) == varY}).values.collect()
      classCol
    }  else {
      data.filter({ case (k, _) => getFeat(k) == varY}).values.collect()
    }    
    
    val histograms = computeHistograms(data, (varY, yCol), nFeatures, counter)
    val jointTable = histograms.mapValues(_.map(_.toFloat / nInstances))
    val marginalTable = jointTable.mapValues(h => sum(h(*, ::)).toDenseVector)
      
    // If y corresponds with output feature, we save for CMI computation
    if(varY == label) {
      marginalProb = marginalTable.cache()
      jointProb = jointTable.cache()
    }
    
    val yProb = marginalTable.lookup(varY)(0)
    // Remove output feature from the computations
    val fdata = histograms.filter{case (k, _) => k != label}
    computeMutualInfo(fdata, yProb, nInstances)
  }
  
  private def computeHistograms(
      data: RDD[(Long, Byte)],
      yCol: (Int, Array[Byte]),
      nFeatures: Long,
      counter: Map[Int, Int]) = {
    
    val maxSize = 256 
    val byCol = data.context.broadcast(yCol._2)    
    val bCounter = data.context.broadcast(counter) 
    val ys = counter.getOrElse(yCol._1, maxSize).toInt
      
    data.mapPartitions({ it =>
      var result = Map.empty[Int, BDM[Long]]
      for((k, x) <- it) {
        val feat = (k % nFeatures).toInt; val inst = (k / nFeatures).toInt
        val xs = bCounter.value.getOrElse(feat, maxSize).toInt
        val m = result.getOrElse(feat, BDM.zeros[Long](xs, ys))        
        m(x, byCol.value(inst)) += 1
        result += feat -> m
      }
      result.toIterator
    }).reduceByKey(_ + _)
  }
  
  private def computeMutualInfo(
      data: RDD[(Int, BDM[Long])],
      yProb: BDV[Float],
      n: Long) = {    
    
    val byProb = data.context.broadcast(yProb)       
    data.mapValues({ m =>
      var mi = 0.0d
      // Aggregate by row (x)
      val xProb = sum(m(*, ::)).map(_.toFloat / n)
      for(i <- 0 until m.rows){
        for(j <- 0 until m.cols){
          val pxy = m(i, j).toFloat / n
          val py = byProb.value(j); val px = xProb(i)
          if(pxy != 0 && px != 0 && py != 0) // To avoid NaNs
            mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
        }
      } 
      mi        
    })  
  }  
}
