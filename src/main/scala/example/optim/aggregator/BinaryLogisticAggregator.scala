package example.optim.aggregator

import example.feature.Point
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector

class BinaryLogisticAggregator(weights: (Double, Double))(bcCoefficients: Broadcast[Vector])
  extends DifferentiableLossAggregator[Point, BinaryLogisticAggregator] {

  override val dim = bcCoefficients.value.size

  @transient
  private lazy val coefficientsArray = bcCoefficients.value.toArray

  private def binaryUpdateInPlace(features: Vector, weight: Double, label: Double): Unit = {
    val localCoefficients = coefficientsArray
    val localGradientArray = gradientSumArray
    val margin = - {
      var sum = 0.0
      features.foreachActive { (index, value) =>
        sum += localCoefficients(index) * value
      }
      // Intercept
      sum += localCoefficients(features.size)
      sum
    }

    val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - label)

    features.foreachActive { (index, value) =>
      localGradientArray(index) += multiplier * value
    }
    // Intercept
    localGradientArray(features.size) += multiplier

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

  override def add(point: Point): BinaryLogisticAggregator = {
    point match {
      case Point(label, features) =>
        val weight = if (label > 0) weights._1 else weights._2
        binaryUpdateInPlace(features, weight, label)
        weightSum += weight
        this
    }
  }
}
