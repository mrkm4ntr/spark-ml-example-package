package org.apache.spark.ml.classification
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasAggregationDepth, HasMaxIter, HasTol}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable

trait BinomialLogisticRegressionParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasTol with HasAggregationDepth {

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 10)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)
}

class BinomialLogisticRegression(override val uid: String) extends ProbabilisticClassifier[Vector, BinomialLogisticRegression, BinomialLogisticRegressionModel]
  with BinomialLogisticRegressionParams {
  override def copy(extra: ParamMap): BinomialLogisticRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinomialLogisticRegressionModel = {
    val points = dataset.select($(labelCol), $(featuresCol)).rdd.map {
      case Row(label: Double, features: Vector) => Point(label, features)
    }
    val optimizer = new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    val costFun = new BinomialLogisticCostFun(points, $(aggregationDepth))
    val init = Vectors.zeros(points.first().features.size + 1)
    val states = optimizer.iterations(new CachedDiffFunction(costFun), new BDV(init.toArray))
    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    val allCoefficients = state.x.toArray.clone
    new BinomialLogisticRegressionModel(uid, Vectors.dense(allCoefficients.init), allCoefficients.last)
  }
}

case class Point(label: Double, features: Vector)

class BinomialLogisticCostFun(
  points: RDD[Point],
  aggregationDepth: Int
) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val bcCoefficients = points.context.broadcast(Vectors.fromBreeze(coefficients))
    val logisticAggregator = {
      val seqOp = (c: BinomialLogisticAggregator, point: Point) => c.add(point)
      val combOp = (c1: BinomialLogisticAggregator, c2: BinomialLogisticAggregator) => c1.merge(c2)
      points.treeAggregate(new BinomialLogisticAggregator(bcCoefficients))(seqOp, combOp, aggregationDepth)
    }
    val totalGradientVector = logisticAggregator.gradient
    bcCoefficients.destroy(blocking = false)
    (logisticAggregator.loss, new BDV(totalGradientVector.toArray))
  }
}

class BinomialLogisticAggregator(
  bcCoefficients: Broadcast[Vector]
) extends Serializable {
  private var weightSum = 0.0
  private var lossSum = 0.0

  @transient
  private lazy val coefficientsArray = bcCoefficients.value.toArray

  private lazy val gradientSumArray = new Array[Double](bcCoefficients.value.size)

  private def binaryUpdateInPlace(features: Vector, label: Double): Unit = {
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

    val multiplier = 1.0 / (1.0 + math.exp(margin) - label)

    features.foreachActive { (index, value) =>
      localGradientArray(index) += multiplier * value
    }
    // Intercept
    localGradientArray(features.size) += multiplier

    if (label > 0) {
      lossSum += MLUtils.log1pExp(margin)
    } else {
      lossSum += MLUtils.log1pExp(margin) - margin
    }
  }

  def add(point: Point): this.type = point match {
    case Point(label, features) =>
      binaryUpdateInPlace(features, label)
      weightSum += 1
      this
  }

  def merge(other: BinomialLogisticAggregator): this.type = {
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
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)
    new DenseVector(result.toArray)
  }
}

class BinomialLogisticRegressionModel(
  override val uid: String,
  val coefficients: Vector,
  val intercept: Double
) extends ProbabilisticClassificationModel[Vector, BinomialLogisticRegressionModel]
  with BinomialLogisticRegressionParams {
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    Vectors.dense(rawPrediction.asInstanceOf[DenseVector].values.map(p => 1 / (1 + math.exp(-p))))

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = Vectors.dense(Array(BLAS.dot(features, coefficients) + intercept))

  override def copy(extra: ParamMap): BinomialLogisticRegressionModel = defaultCopy(extra)
}
