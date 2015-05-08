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


import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

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
 * 
 * @param criterionFactory Factory to create info-theory measurements for each feature.
 * @param data RDD of LabeledPoint (discrete data).
 * 
 */
class InfoThSelector private[feature] (val selector: MrmrSelector) extends Serializable {

  // Pool of criterions
  private type Pool = RDD[(Int, InfoThCriterion)]
  // Case class for criterions by feature
  protected case class F(feat: Int, crit: Double) 

  /**
   * Perform a info-theory selection process without pool optimization.
   * 
   * @param data Data points (first element is the class attribute).
   * @param nToSelect Number of features to select.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[feature] def selectFeatures(data: RDD[BV[Byte]], nToSelect: Int) = {
    
    val nElements = data.count()
    val nFeatures = data.first.size - 1
    val label = 0
    
    // calculate relevance
    val mutualInfo = IT.computeMutualInfo(data, 1 to nFeatures, Seq(label), nElements, nFeatures)
    var pool = mutualInfo.map{case ((x, y), mi) => (x, selector.init(mi))}
      .collectAsMap()  
    // Print most relevant features
    val strRels = mutualInfo.collect().sortBy(-_._2)
      .take(100)
      .map({case ((f, _), mi) => f + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)  
    // get maximum and select it
    val firstMax = pool.maxBy(_._2.score)
    var selected = Seq(F(firstMax._1, firstMax._2.score))
    pool = pool - firstMax._1

    while (selected.size < nToSelect) {
      // update pool
      val mutualInfo = IT.computeMutualInfo(data, pool.keys.toSeq, Seq(selected.head.feat), 
        nElements, nFeatures) 
        .map({ case ((x, _), crit) => (x, crit) })
        .collectAsMap()
      pool.foreach({ case (k, crit) =>
        mutualInfo.get(k) match {
          case Some(_) => crit.update(_)
          case None => 
        }
      })
      
      val out = pool.map{case (k, crit) => k + "\t" + "%.4f".format(crit.score)}.mkString("\n")
      println("\n*** pool features ***\nFeature\tScore\n" + out)

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
      poolSize: Int = 30) = {
        
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
            throw new SparkException(s"mRMR selector requires integer values in range [0, 255]")
          }
        }
        
        val byteData: RDD[BV[Byte]] = data.map {
          case LabeledPoint(label, values: SparseVector) => 
            requireByteValues(label, values)
            new BSV[Byte](0 +: values.indices.map(_ + 1), 
              label.toByte +: values.values.toArray.map(_.toByte), values.size + 1)
          case LabeledPoint(label, values: DenseVector) => 
            requireByteValues(label, values)
            new BDV[Byte](label.toByte +: values.toArray.map(_.toByte))
        }

        byteData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        val nFeatures = byteData.first.size - 1
        require(nToSelect < nFeatures)
        
        val selected = selectFeatures(byteData, nToSelect)
      
        byteData.unpersist()
      
        // Print best features according to the mRMR measure
        val out = selected.map{case F(feat, rel) => feat + "\t" + "%.4f".format(rel)}.mkString("\n")
        println("\n*** mRMR features ***\nFeature\tScore\n" + out)
        // Features must be sorted
        new SelectorModel(selected.map{case F(feat, rel) => feat - 1}.sorted.toArray)
    }
}

object InfoThSelector {

  /**
   * Train a feature selection model according to a given criterion
   * and return a subset of data.
   *
   * @param   criterionFactory Initialized criterion to use in this selector
   * @param   data RDD of LabeledPoint (discrete data as integers in range [0, 255]).
   * @param   nToSelect maximum number of features to select
   * @param   poolSize number of features to be used in pool optimization.
   * @return  A feature selection model which contains a subset of selected features
   * 
   * Note: LabeledPoint data must be integer values in double representation 
   * with a maximum of 256 distinct values. In this manner, data can be transformed
   * to byte class directly, making the selection process much more efficient. 
   * 
   */
  def train(
      selector: MrmrSelector, 
      data: RDD[LabeledPoint],
      nToSelect: Int = 25) = {
    new InfoThSelector(selector).run(data, nToSelect)
  }
}

