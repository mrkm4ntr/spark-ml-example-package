package example.optim.loss

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.DiffFunction
import example.optim.aggregator.DifferentiableLossAggregator
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import spire.ClassTag

import scala.collection.mutable

class RDDLossFunction[T: ClassTag, Agg <: DifferentiableLossAggregator[T, Agg]: ClassTag](
  data: RDD[T],
  getAggregator: Broadcast[Vector] => Agg,
  regParamL2: Double,
  aggregateDepth: Int
) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val bcCoefficients = data.context.broadcast(Vectors.dense(coefficients.toArray))
    val aggregator = {
      val seqOp = (c: Agg, datum: T) => c.add(datum)
      val combOp = (c1: Agg, c2: Agg) => c1.merge(c2)
      data.treeAggregate(getAggregator(bcCoefficients))(seqOp, combOp, aggregateDepth)
    }

    val (regVal, totalGradients) = if (regParamL2 == 0.0) {
      (0.0, aggregator.gradient.toArray)
    } else {
      var sum = 0.0
      val arrayBuffer = mutable.ArrayBuffer[Double]()
      aggregator.gradient.foreachActive { case (i, v) =>
        val coefficient = coefficients(i)
        arrayBuffer += v + regParamL2 * coefficient
        sum += coefficient * coefficient
      }
      (0.5 * regParamL2 * sum, arrayBuffer.toArray)
    }
    // Cannot call with non-blocking in this package
    bcCoefficients.destroy()
    (aggregator.loss + regVal, new BDV(totalGradients))
  }
}
