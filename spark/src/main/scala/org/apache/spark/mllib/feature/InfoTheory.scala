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
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast

/**
 * Information Theory function and distributed primitives.
 */
object InfoTheory {

  private val log2 = { x: Double => math.log(x) / math.log(2) }
  
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
  
  
  /* Pair generator for dense data */
  private def DenseGenerator(
      dv: BDV[Byte], 
      varX: Broadcast[Seq[Int]],
      varY: Broadcast[Seq[Int]]) = {
    
     var pairs = Seq.empty[((Any, Byte, Byte), Long)]
     val multY = varY.value.length > 1
     
     for(xind <- varX.value){
       for(yind <- varY.value) {
         val indexes = if(multY) (xind, yind) else xind
         pairs = ((indexes, dv(xind), dv(yind)), 1L) +: pairs
       }
     }     
     pairs
  } 
  
  /* Pair generator for dense data */
  private def SparseGenerator(
      dv: BSV[Byte], 
      varX: Broadcast[Seq[Int]],
      varY: Broadcast[Seq[Int]]) = {
        
     var dpairs = Seq.empty[((Any, Byte, Byte), Long)]
     val multY = varY.value.length > 1
     
     for(xi <- varX.value; if dv(xi) != 0){
       val xa = dv(xi)
       for(yi <- varY.value; if dv(yi) != 0) {
         val ya = dv(yi)
         val indexes = if(multY) (xi, yi) else xi
         dpairs = ((indexes, xa, ya), 1L) +: dpairs
       }
     }     
     dpairs
  }
  
  private def zerosGenerator(
      data: RDD[BSV[Byte]],
      bvarX: Broadcast[Seq[Int]],
      bvarY: Broadcast[Seq[Int]]) = {
    
    val accZeros = data.mapPartitions({ it => 
      var part = Map.empty[(Int, Byte), Array[Int]]
      if(it.hasNext) {  
        for (sv <- it) {
          val yelems = sv(bvarY.value).pairs.iterator.toArray // it is repeatedly accessed
          for (i <- 0 until bvarX.value.size if sv(bvarX.value(i)) == 0) {
            for((yi, yval) <- yelems) {
              // sum up 1 to the global vector
              val v = part.getOrElse((yi, yval), new Array[Int](bvarX.value.size))
              v(i) = v(i) + 1
              part += ((yi, yval) -> v)
            }
          }
        }
      }
      part.toIterator          
    }).reduceByKey((_ , _).zipped.map(_ + _))
     
    accZeros.flatMap{ case ((yi, ya), acc) =>
      var dpairs = Seq.empty[((Any, Byte, Byte), Long)]
      for(i <- 0 until bvarX.value.size if acc(i) != 0) {        
        val xi = bvarX.value(i)
        val idx = if(bvarY.value.length > 1) (xi, yi) else xi
        dpairs = ((idx, 0: Byte, ya), acc(i).toLong) +: dpairs
      }
      dpairs
    }
  }
    
  
  /**
   * Method that calculates mutual information (MI)
   * simultaneously for several variables. Indexes must be disjoint.
   *
   * @param data RDD of data (first element is the class attribute)
   * @param varX Indexes of primary variables (must be disjoint with Y)
   * @param varY Indexes of secondary variable (must be disjoint with X)
   * @param n    Number of instances
   * @return     RDD of (primary var, MI)
   * 
   */
  def computeMutualInfo(
      data: RDD[BV[Byte]],
      varX: Seq[Int],
      varY: Seq[Int],
      n: Long,      
      nFeatures: Int, 
      inverseX: Boolean = false) = {
    
    // Pre-requisites
    require(varX.size > 0 && varY.size > 0)

    // Broadcast variables
    val sc = data.context
    val bvarX = sc.broadcast(varX)
    val bvarY = sc.broadcast(varY)
    
    // Common function to generate pairs, it chooses between sparse and dense processing 
    val pairs = data.first match {
      case _: BDV[Byte] =>
        val denseData = data.map(_.asInstanceOf[BDV[Byte]])
        val generator = DenseGenerator(_: BDV[Byte], bvarX, bvarY)
        denseData.flatMap(generator).reduceByKey(_ + _)
      case _: BSV[Byte] =>     
        val sparseData = data.map(_.asInstanceOf[BSV[Byte]])
        val generator = SparseGenerator(_: BSV[Byte], bvarX, bvarY)
        val densePairs = sparseData.flatMap(generator).reduceByKey(_ + _)
        val zeroPairs = zerosGenerator(sparseData, bvarX, bvarY)
        densePairs.union(zeroPairs)        
    }
    computeMI(pairs, varY(0), n)
  }
  
  /**
   * Submethod that calculates MIfor dense data.
   *
   * @param data RDD of data (first element is the class attribute)
   * @param pairsGenerator Function that generates the combinations between variables
   * @param firstY First Y index
   * @param n    Number of instances
   * @return     RDD of (primary var, MI)
   * 
   */
  private[mllib] def computeMI(
      pairs: RDD[((Any, Byte, Byte), Long)],
      firstY: Int,
      n: Long) = {
    
  val combinations = pairs
    // Split each combination keeping instance keys
      .flatMap {
      case ((k, x, y), q) =>            
        Seq(((k, 1:Byte /* "xy" */ , (x, y)),    (Set.empty,  q)),
            ((k, 2:Byte /* "x" */  , x),         (Set(y), q)),
            ((k, 3:Byte /* "y" */  , y),         (Set(x), q)))
    }
    
    val createCombiner: ((Byte, Long)) => (Long, Long, Long) = {
      case (1, q) => (q, 0, 0)
      case (2, q) => (0, q, 0)
      case (3, q) => (0, 0, q)
    }

    val mergeValues: ((Long, Long, Long), (Byte, Long)) => 
        (Long, Long, Long) = {
      case ((qxy, qx, qy), (ref, q)) =>
        ref match {
          case 1 => (qxy + q, qx, qy)
          case 2 => (qxy, qx + q, qy)
          case 3 => (qxy, qx, qy + q)
        }
    }

    val mergeCombiners: (
        (Long, Long, Long), 
        (Long, Long, Long)) => 
        (Long, Long, Long) = {
      case ((qxy1, qx1, qy1), (qxy2, qx2, qy2)) =>
        (qxy1 + qxy2, qx1 + qx2, qy1 + qy2)
    }
  
    // Count frequencies for each combination
    val grouped_frequencies = combinations.reduceByKey({
      case ((keys1, q1), (keys2, q2)) => (keys1 ++ keys2, q1 + q2)
    })      
    // Separate by origin of combinations
    .flatMap({
      case ((k, ref, value), (keys, q))  => 
          val none: Option[Byte] = None
          ref match {
            case 1 =>
                val (x, y) = value.asInstanceOf[(Byte, Byte)]
                Seq(((k, x, y), (1:Byte, q)))
            case 2 =>
              val x = value.asInstanceOf[Byte]
              for (y <- keys) yield ((k, x, y.asInstanceOf[Byte]), (2:Byte, q))
            case 3 =>
              val y = value.asInstanceOf[Byte]
              for (x <- keys) yield ((k, x.asInstanceOf[Byte], y), (3:Byte, q))
          }        
    })
    // Group by origin
    .combineByKey(createCombiner, mergeValues, mergeCombiners)

    // Calculate MI by instance
    grouped_frequencies.map({case ((k, _, _), (qxy, qx, qy)) =>
    // Select id
      val finalKey = k match {
        case (kx: Int, ky: Int) => (kx, ky)
        case kx: Int => (kx, firstY)
      }           
      
      val pxy = qxy.toDouble / n
      val px = qx.toDouble / n
      val py = qy.toDouble / n
      (finalKey, pxy * log2(pxy / (px * py)))
    })
    // Compute the final result by attribute
    .reduceByKey(_ + _)
  }
}
