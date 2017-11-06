package example.optim.aggregator

import example.feature.Point
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.BinaryFactorizationMachinesModel
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

class BinaryFactorizationMachinesAggregator(k: Int, n: Int, weights: (Double, Double))(bcCoefficients: Broadcast[Vector])
  extends DifferentiableLossAggregator[Point, BinaryFactorizationMachinesAggregator] {

  override protected val dim: Int = bcCoefficients.value.size

  @transient
  private lazy val coefficientsArray = bcCoefficients.value.toArray
  @transient
  private lazy val intercept = coefficientsArray.last
  @transient
  private lazy val (coefficients, v) = {
    val (coefficients, v) = coefficientsArray.init.splitAt(n)
    (Vectors.dense(coefficients), Matrices.dense(k, n, v))
  }

  private def binaryUpdateInPlace(features: Vector, weight: Double, label: Double): Unit = {
    val localGradientArray = gradientSumArray
    val margin = -BinaryFactorizationMachinesModel.predict(features, coefficients, intercept, v)
    val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - label)

    val precomputed = v.multiply(features) // f * 1
    features.foreachActive { case (i, value) =>
        localGradientArray(i) += multiplier * value
        precomputed.foreachActive { case (f, pre) =>
          localGradientArray(n * (1 + f) + i) += multiplier * value * (pre - v(f, i) * value)
        }
    }
    // Intercept
    localGradientArray(dim - 1) += multiplier

    def log1pExp(x: Double) = if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }

    if (label > 0) {
      lossSum += weight * log1pExp(margin)
    } else {
      lossSum += weight * (log1pExp(margin) - margin)
    }
  }

  override def add(point: Point): BinaryFactorizationMachinesAggregator = point match {
    case Point(label, features) =>
      val weight = if (label > 0) weights._1 else weights._2
      binaryUpdateInPlace(features, weight, label)
      weightSum += weight
      this
  }
}
