package org.apache.spark.ml.classification

import breeze.optimize.{CachedDiffFunction, LBFGS}
import breeze.linalg.{DenseVector => BDV}
import example.classification.BinaryClassificationSummary
import example.feature.Point
import example.optim.aggregator.BinaryFactorizationMachinesAggregator
import example.optim.loss.RDDLossFunction
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamMap, Params}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable


trait HasK extends Params {

  final val k: IntParam = new IntParam(this, "k", "")

  final def getK: Int = $(k)
}

trait BinaryFactorizationMachinesParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasTol with HasAggregationDepth with HasThreshold
  with HasK {

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 10)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold -> 0.5)

  def setK(value: Int): this.type = set(k, value)
  setDefault(k -> 4)
}

class BinaryFactorizationMachines(override val uid: String)
  extends ProbabilisticClassifier[Vector, BinaryFactorizationMachines, BinaryFactorizationMachinesModel]
  with BinaryFactorizationMachinesParams {

  def this() = this(Identifiable.randomUID("binfm"))

  override def copy(extra: ParamMap): BinaryFactorizationMachines = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinaryFactorizationMachinesModel = {
    val points = dataset.select($(labelCol), $(featuresCol)).rdd.map {
      case Row(label: Double, features: Vector) => Point(label, features)
    }

    val numOfFeatures = points.first().features.size
    val numOfCoefficients = numOfFeatures + 1 + $(k) * numOfFeatures

    val optimizer = new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    val costFun = new RDDLossFunction(points, new BinaryFactorizationMachinesAggregator($(k), numOfFeatures)(_), 0.0, $(aggregationDepth))
    val init = Vectors.zeros(numOfCoefficients)
    val states = optimizer.iterations(new CachedDiffFunction(costFun), new BDV(init.toArray))
    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    val allCoefficients = state.x.toArray.clone
    val intercept = allCoefficients.last
    val (coefficients, v) = {
      val (coefficients, v) = allCoefficients.init.splitAt(numOfFeatures)
      (Vectors.dense(coefficients), Matrices.dense($(k), numOfFeatures, v))
    }
    new BinaryFactorizationMachinesModel(uid, coefficients, intercept, v)
  }
}

class BinaryFactorizationMachinesModel(
  override val uid: String,
  val coefficients: Vector,
  val intercept: Double,
  val v: Matrix // f * n
) extends ProbabilisticClassificationModel[Vector, BinaryFactorizationMachinesModel]
  with BinaryFactorizationMachinesParams {

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    Vectors.dense(rawPrediction.asInstanceOf[DenseVector].values.map(p => 1.0 / (1.0 + math.exp(-p))))

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector =
    Vectors.dense(Array(BinaryFactorizationMachinesModel.predict(features, coefficients, intercept, v)))

  override def copy(extra: ParamMap): BinaryFactorizationMachinesModel = defaultCopy(extra)

  override protected def probability2prediction(probability: Vector): Double =
    if (probability(0) > $(threshold)) 1 else 0

  protected override def raw2prediction(rawPrediction: Vector): Double =
    probability2prediction(raw2probability(rawPrediction))

  def findSummaryModelAndProbabilityCol(): (BinaryFactorizationMachinesModel, String) = {
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

object BinaryFactorizationMachinesModel {

  def predict(features: Vector, coefficients: Vector, intercept: Double, v: Matrix) = {
    val interactive = v.rowIter.foldLeft(0.0) { case (acc, v) =>
      var fst = 0.0
      var snd = 0.0
      features.foreachActive { case (index, x) =>
        val vx = v(index) * x
        fst += vx
        snd += math.pow(vx, 2.0)
      }
      acc + math.pow(fst, 2.0) - snd
    }
    BLAS.dot(features, coefficients) + intercept + 0.5 * interactive
  }
}
