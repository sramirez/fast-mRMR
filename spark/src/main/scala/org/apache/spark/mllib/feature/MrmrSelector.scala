package org.apache.spark.mllib.feature


/**
 * Minimum-Redundancy Maximum-Relevance criterion (mRMR)
 */
class MrmrSelector extends InfoThCriterion {

  var redundance: Double = 0.0
  var selectedSize: Int = 0

  override def bound = relevance
  override def score = {
    if (selectedSize != 0) {
      relevance - redundance / selectedSize
    } else {
      relevance
    }
  }
  override def init(relevance: Double): InfoThCriterion = {
    this.setRelevance(relevance)
  }
  override def update(mi: Double): InfoThCriterion = {
    redundance += mi
    selectedSize += 1
    this
  }
  override def toString: String = "MRMR"
}