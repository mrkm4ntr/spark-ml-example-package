package org.apache.spark.ml.classification
import breeze.optimize.{CachedDiffFunction, LBFGS, OWLQN}
import breeze.linalg.{DenseVector => BDV}
import example.classification.BinaryClassificationSummary
import example.feature.Point
import example.optim.aggregator.BinaryLogisticAggregator
import example.optim.loss.RDDLossFunction
import example.param.HasBalancedWeight
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable

trait BinaryLogisticRegressionParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasTol with HasAggregationDepth with HasThreshold
  with HasRegParam with HasElasticNetParam with HasBalancedWeight {

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 10)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold -> 0.5)

  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  def setBalancedWeight(value: Boolean): this.type  = set(balancedWeight, value)
  setDefault(balancedWeight -> true)
}

class BinaryLogisticRegression(override val uid: String) extends ProbabilisticClassifier[Vector, BinaryLogisticRegression, BinaryLogisticRegressionModel]
  with BinaryLogisticRegressionParams {

  def this() = this(Identifiable.randomUID("binlogreg"))

  override def copy(extra: ParamMap): BinaryLogisticRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinaryLogisticRegressionModel = {
    val points = dataset.select($(labelCol), $(featuresCol)).rdd.map {
      case Row(label: Double, features: Vector) => Point(label, features)
    }

    val numOfCoefficients = points.first().features.size + 1

    val weights = if ($(balancedWeight)) {
      val seqOp = (c: BinaryClassSummarizer, point: Point) => c.add(point.label)
      val combOp = (c1: BinaryClassSummarizer, c2: BinaryClassSummarizer) => c1.merge(c2)
      val summarizer = points.treeAggregate(new BinaryClassSummarizer)(seqOp, combOp)
      summarizer.weights()
    } else (1.0, 1.0)

    val regParamL1 = $(elasticNetParam) * $(regParam)
    val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

    val optimizer = if (regParamL1 == 0.0) {
      new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    } else {
      def regParamL1Fun = (index: Int) => {
        val isIntercept = index == numOfCoefficients - 1
        if (isIntercept) {
          0.0
        } else {
          regParamL1
        }
      }
      new OWLQN[Int, BDV[Double]]($(maxIter), 10, regParamL1Fun, $(tol))
    }

    val costFun = new RDDLossFunction(points, new BinaryLogisticAggregator(weights)(_), regParamL2, $(aggregationDepth))
    val init = Vectors.zeros(numOfCoefficients)
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

class BinaryClassSummarizer extends Serializable {
  private val distinctMap = new mutable.HashMap[Int, Long]

  def add(label: Double): this.type = {
    val counts = distinctMap.getOrElse(label.toInt, 0L)
    distinctMap.put(label.toInt, counts + 1L)
    this
  }

  def merge(other: BinaryClassSummarizer): BinaryClassSummarizer = {
    val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    largeMap.distinctMap ++= smallMap.distinctMap
    largeMap
  }

  def weights() = {
    val numOfPos = distinctMap(1)
    val numOfNeg = distinctMap(0)
    val total = numOfPos + numOfNeg
    (total / (2.0 * numOfPos), total / (2.0 * numOfNeg))
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

  override protected def predictRaw(features: Vector): Vector = {
    val margin = BLAS.dot(features, coefficients) + intercept
    Vectors.dense(Array(-margin, margin))
  }

  override def copy(extra: ParamMap): BinaryLogisticRegressionModel = {
    val newModel = copyValues(new BinaryLogisticRegressionModel(uid, coefficients, intercept), extra)
    newModel.setParent(parent)
  }

  override protected def probability2prediction(probability: Vector): Double =
    if (probability(1) > $(threshold)) 1 else 0

  protected override def raw2prediction(rawPrediction: Vector): Double =
    probability2prediction(raw2probability(rawPrediction))

  def findSummaryModelAndProbabilityCol(): (BinaryLogisticRegressionModel, String) = {
    $(probabilityCol) match {
      case "" =>
        val probabilityColName = "probability_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setProbabilityCol(probabilityColName), probabilityColName)
      case p => (this, p)
    }
  }

  def evaluate(dataset: Dataset[_]): BinaryClassificationSummary = {
    val (summaryModel, probabilityColName) = findSummaryModelAndProbabilityCol()
    new BinaryClassificationSummary(summaryModel.transform(dataset),
      probabilityColName, $(labelCol), $(featuresCol))
  }
}




