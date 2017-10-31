package example.optim.aggregator

import org.apache.spark.ml.linalg.{DenseVector, Vector}

trait DifferentiableLossAggregator[
  Datum,
  Agg <: DifferentiableLossAggregator[Datum, Agg]] extends Serializable {

  self: Agg =>

  protected var weightSum = 0.0
  protected var lossSum = 0.0

  protected val dim: Int
  protected lazy val gradientSumArray = Array.ofDim[Double](dim)

  def add(datum: Datum): Agg

  def merge(other: Agg): Agg = {
    if (other.weightSum != 0) {
      weightSum += other.weightSum
      lossSum += other.lossSum
    }
    other.gradientSumArray.zipWithIndex.foreach { case (v, i) =>
      this.gradientSumArray(i) += v
    }
    this
  }

  def loss: Double = lossSum / weightSum

  def gradient: Vector = {
    val result = gradientSumArray.clone().map((1.0 / weightSum) * _)
    new DenseVector(result)
  }
}
