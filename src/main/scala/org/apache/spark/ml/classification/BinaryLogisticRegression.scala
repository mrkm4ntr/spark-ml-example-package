package org.apache.spark.ml.classification
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasAggregationDepth, HasMaxIter, HasThreshold, HasTol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable

trait BinaryLogisticRegressionParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasTol with HasAggregationDepth with HasThreshold {

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 10)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  def setThreshold(value: Double): this.type  = set(threshold, value)
  setDefault(threshold -> 0.5)
}

class BinaryLogisticRegression(override val uid: String) extends ProbabilisticClassifier[Vector, BinaryLogisticRegression, BinaryLogisticRegressionModel]
  with BinaryLogisticRegressionParams {

  def this() = this(Identifiable.randomUID("binlogreg"))

  override def copy(extra: ParamMap): BinaryLogisticRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinaryLogisticRegressionModel = {
    val points = dataset.select($(labelCol), $(featuresCol)).rdd.map {
      case Row(label: Double, features: Vector) => Point(label, features)
    }
    val optimizer = new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    val costFun = new BinaryLogisticCostFun(points, $(aggregationDepth))
    val init = Vectors.zeros(points.first().features.size + 1)
    val states = optimizer.iterations(new CachedDiffFunction(costFun), new BDV(init.toArray))
    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    val allCoefficients = state.x.toArray.clone
    new BinaryLogisticRegressionModel(uid, Vectors.dense(allCoefficients.init), allCoefficients.last)
  }
}

case class Point(label: Double, features: Vector)

class BinaryLogisticCostFun(
  points: RDD[Point],
  aggregationDepth: Int
) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val bcCoefficients = points.context.broadcast(Vectors.fromBreeze(coefficients))
    val logisticAggregator = {
      val seqOp = (c: BinaryLogisticAggregator, point: Point) => c.add(point)
      val combOp = (c1: BinaryLogisticAggregator, c2: BinaryLogisticAggregator) => c1.merge(c2)
      points.treeAggregate(new BinaryLogisticAggregator(bcCoefficients))(seqOp, combOp, aggregationDepth)
    }
    val totalGradientVector = logisticAggregator.gradient
    bcCoefficients.destroy(blocking = false)
    (logisticAggregator.loss, new BDV(totalGradientVector.toArray))
  }
}

class BinaryLogisticAggregator(
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

    val multiplier = 1.0 / (1.0 + math.exp(margin)) - label

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

  def merge(other: BinaryLogisticAggregator): this.type = {
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

class BinaryLogisticRegressionModel(
  override val uid: String,
  val coefficients: Vector,
  val intercept: Double
) extends ProbabilisticClassificationModel[Vector, BinaryLogisticRegressionModel]
  with BinaryLogisticRegressionParams {
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    Vectors.dense(rawPrediction.asInstanceOf[DenseVector].values.map(p => 1.0 / (1.0 + math.exp(-p))))

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = Vectors.dense(Array(BLAS.dot(features, coefficients) + intercept))

  override def copy(extra: ParamMap): BinaryLogisticRegressionModel = defaultCopy(extra)

  override protected def probability2prediction(probability: Vector): Double =
    if (probability(0) > $(threshold)) 1 else 0

  protected override def raw2prediction(rawPrediction: Vector): Double =
    probability2prediction(raw2probability(rawPrediction))

  def findSummaryModelAndProbabilityCol():
  (BinaryLogisticRegressionModel, String) = {
    $(probabilityCol) match {
      case "" =>
        val probabilityColName = "probability_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setProbabilityCol(probabilityColName), probabilityColName)
      case p => (this, p)
    }
  }

  def evaluate(dataset: Dataset[_]): BinaryLogisticRegressionSummary2 = {
    val (summaryModel, probabilityColName) = findSummaryModelAndProbabilityCol()
    new BinaryLogisticRegressionSummary2(summaryModel.transform(dataset),
      probabilityColName, $(labelCol), $(featuresCol))
  }
}

trait BinaryClassificationSummary extends Serializable {
  def predictions: DataFrame
  def probabilityCol: String
  def labelCol: String
  def featuresCol: String
}

class BinaryLogisticRegressionSummary2(
  override val predictions: DataFrame,
  override val probabilityCol: String,
  override val labelCol: String,
  override val featuresCol: String
) extends BinaryClassificationSummary {

  @transient private val binaryMetrics = new BinaryClassificationMetrics(
    predictions.select(col(probabilityCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(score: Vector, label: Double) => (score(0), label)
    }, 100
  )

  lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()
}
