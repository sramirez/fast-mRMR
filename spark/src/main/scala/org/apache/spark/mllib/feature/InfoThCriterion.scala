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

/**
 * Trait which declares needed methods to define a criterion for Feature Selection
 */
trait InfoThCriterion extends Serializable with Ordered[InfoThCriterion] {

  var relevance: Double = 0.0

  /**
   * Protected method to set the relevance.
   * The default value is 0.0.
   */
  protected def setRelevance(relevance: Double): InfoThCriterion = {
    this.relevance = relevance
    this
  }

  /** 
   *  Compares the score of two criterions
   */
  override def compare(that: InfoThCriterion) = {
    this.score.compare(that.score)
  }

  /** 
   * Initialize a criterion with a given relevance value
   */
  def init(relevance: Double): InfoThCriterion

  /**
   * 
   * Updates the criterion score with new mutual information and conditional mutual information.
   * @param mi Mutual information between the criterion and another variable.
   * 
   */
  def update(mi: Double): InfoThCriterion

  /**
   * Returns the value of the criterion for in a precise moment.
   */
  def score: Double

}
