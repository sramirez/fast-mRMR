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

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{InfoTheory => IT}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{Vector, DenseVector, SparseVector}
import org.apache.spark.annotation.Experimental
import org.apache.spark.SparkException

/**
 * Train a info-theory feature selection model according to a criterion. 
 */
class MrmrSelector private[feature] extends Serializable {

  // Pool of criterions
  private type Pool = RDD[(Int, MrmrCriterion)]
  // Case class for criterions by feature
  protected case class F(feat: Int, crit: Double) 

  /**
   * Perform a info-theory selection process.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nToSelect Number of features to select.
   * @param nFeatures Number of total features in the dataset.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[feature] def selectFeatures(
      data: RDD[(Long, Byte)], 
      nToSelect: Int,
      nFeatures: Int) = {
    
    val label = nFeatures - 1 
    val nInstances = data.count() / nFeatures
    val counterByKey = data.map({ case (k, v) => (k % nFeatures).toInt -> v})
          .distinct().groupByKey().mapValues(_.max + 1).collectAsMap().toMap
    
    // calculate relevance
    val MiAndCmi = IT.computeMI(
        data, 0 until label, label, nInstances, nFeatures, counterByKey)
    var pool = MiAndCmi.map{case (x, mi) => (x, new MrmrCriterion(mi))}
      .collectAsMap()  
    // Print most relevant features
    // Print most relevant features
    val strRels = MiAndCmi.collect().sortBy(-_._2)
      .take(nToSelect)
      .map({case (f, mi) => (f + 1) + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)  
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val newMiAndCmi = IT.computeMI(data, pool.keys.toSeq, 
          selected.head.feat, nInstances, nFeatures, counterByKey) 
          .map({ case (x, crit) => (x, crit) })
          .collectAsMap()
        
      pool.foreach({ case (k, crit) =>
        newMiAndCmi.get(k) match {
          case Some(_) => crit.update(_)
          case None => 
        }
      })

      // get maximum and save it
      val max = pool.maxBy(_._2.score)
      // select the best feature and remove from the whole set of features
      selected = F(max._1, max._2.score) +: selected
      pool = pool - max._1
    }    
    selected.reverse
  }
  
  private[feature] def run(
      data: RDD[LabeledPoint], 
      nToSelect: Int, 
      numPartitions: Int) = {
    
    val nPart = if(numPartitions == 0) data.context.getConf.getInt(
        "spark.default.parallelism", 500) else numPartitions
      
    val requireByteValues = (l: Double, v: Vector) => {        
      val values = v match {
        case SparseVector(size, indices, values) =>
          values
        case DenseVector(values) =>
          values
      }
      val condition = (value: Double) => value <= Byte.MaxValue && 
        value >= Byte.MinValue && value % 1 == 0.0
      if (!values.forall(condition(_)) || !condition(l)) {
        throw new SparkException(s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
        
    val nAllFeatures = data.first.features.size + 1        
    val columnarData: RDD[(Long, Byte)] = data.zipWithIndex().flatMap ({
      case (LabeledPoint(label, values: SparseVector), r) => 
        requireByteValues(label, values)
        // Not implemented yet!
        throw new NotImplementedError()           
      case (LabeledPoint(label, values: DenseVector), r) => 
        requireByteValues(label, values)
        val rindex = r * nAllFeatures
        val inputs = for(i <- 0 until values.size) yield (rindex + i, values(i).toByte)
        val output = Array((rindex + values.size, label.toByte))
        inputs ++ output    
    }).sortByKey(numPartitions = nPart) // put numPartitions parameter        
    columnarData.persist(StorageLevel.MEMORY_AND_DISK_SER)  
        
    require(nToSelect < nAllFeatures)        
    val selected = selectFeatures(columnarData, nToSelect, nAllFeatures)
          
    columnarData.unpersist()
  
    // Print best features according to the mRMR measure
    val out = selected.map{case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel)}.mkString("\n")
    println("\n*** mRMR features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    new SelectorModel(selected.map{case F(feat, rel) => feat}.sorted.toArray)
    }
}

object MrmrSelector {

  /**
   * Train a mRMR selection model according to a given criterion
   * and return a subset of data.
   *
   * @param   data RDD of LabeledPoint (discrete data as integers in range [0, 255]).
   * @param   nToSelect maximum number of features to select
   * @param   numPartitions number of partitions to structure the data.
   * @return  A mRMR selector that selects a subset of features from the original dataset.
   * 
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum of 256 distinct values. In this manner, data can be transformed
   * to byte class directly, making the selection process much more efficient. 
   * 
   */
  def train( 
      data: RDD[LabeledPoint],
      nToSelect: Int = 25,
      numPartitions: Int = 0) = {
    new MrmrSelector().run(data, nToSelect, numPartitions)
  }
}

