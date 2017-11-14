package org.apache.spark.ml.classification

import java.util.Base64

import breeze.optimize.{CachedDiffFunction, LBFGS, OWLQN}
import breeze.linalg.{DenseVector => BDV}
import example.classification.BinaryClassificationSummary
import example.feature.Point
import example.optim.aggregator.BinaryFactorizationMachinesAggregator
import example.optim.loss.RDDLossFunction
import example.param.HasBalancedWeight
import example.stat.BinaryClassSummarizer
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup}
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamMap, Params}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{Identifiable, Instrumentation, MLWritable, MLWriter}
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable


trait HasK extends Params {

  final val k: IntParam = new IntParam(this, "k", "")

  final def getK: Int = $(k)
}

trait BinaryFactorizationMachinesParams extends ProbabilisticClassifierParams
  with HasMaxIter with HasTol with HasAggregationDepth with HasThreshold
  with HasRegParam with HasElasticNetParam with HasBalancedWeight with HasK

class BinaryFactorizationMachines(override val uid: String)
  extends ProbabilisticClassifier[Vector, BinaryFactorizationMachines, BinaryFactorizationMachinesModel]
  with BinaryFactorizationMachinesParams {

  def this() = this(Identifiable.randomUID("binfm"))

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

  def setBalancedWeight(value: Boolean): this.type = set(balancedWeight, value)
  setDefault(balancedWeight -> true)

  def setK(value: Int): this.type = set(k, value)
  setDefault(k -> 4)

  override def copy(extra: ParamMap): BinaryFactorizationMachines = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinaryFactorizationMachinesModel = {
    val points = dataset.select($(labelCol), $(featuresCol)).rdd.map {
      case Row(label: Double, features: Vector) => Point(label, features)
    }

    val instr = Instrumentation.create(this, points)
    instr.logParams(maxIter, tol, aggregationDepth, threshold, regParam, elasticNetParam, balancedWeight, k)

    val numOfFeatures = points.first().features.size
    val numOfCoefficients = numOfFeatures + 1 + $(k) * numOfFeatures

    instr.logNumFeatures(numOfFeatures)

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

    val costFun = new RDDLossFunction(points,
      new BinaryFactorizationMachinesAggregator($(k), numOfFeatures, weights)(_), regParamL2, $(aggregationDepth))
    val init = Vectors.dense(Array.fill(numOfCoefficients)(1E-6))
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
    val group = AttributeGroup.fromStructField(dataset.schema($(featuresCol)))
    val model = copyValues(new BinaryFactorizationMachinesModel(uid, coefficients, intercept, v, group.attributes))
    instr.logSuccess(model)
    model
  }
}

class BinaryFactorizationMachinesModel(
  override val uid: String,
  val coefficients: Vector,
  val intercept: Double,
  val v: Matrix, // f * n
  val attributes: Option[Array[Attribute]],
  var decodeColumnName: Boolean = false
) extends ProbabilisticClassificationModel[Vector, BinaryFactorizationMachinesModel]
  with BinaryFactorizationMachinesParams with MLWritable {

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    Vectors.dense(rawPrediction.asInstanceOf[DenseVector].values.map(p => 1.0 / (1.0 + math.exp(-p))))

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val margin = BinaryFactorizationMachinesModel.predict(features, coefficients, intercept, v)
    Vectors.dense(Array(-margin, margin))
  }

  override def copy(extra: ParamMap): BinaryFactorizationMachinesModel = {
    val newModel = copyValues(new BinaryFactorizationMachinesModel(uid, coefficients, intercept, v, attributes, decodeColumnName), extra)
    newModel.setParent(parent)
  }

  override protected def probability2prediction(probability: Vector): Double =
    if (probability(1) > $(threshold)) 1 else 0

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

  // TODO: handle exception
  override def write: MLWriter =
    new BinaryFactorizationMachinesModel.BinaryFactorizationMachinesModelWriter(this, attributes.get, decodeColumnName)
}

object BinaryFactorizationMachinesModel {

  private case class FMModel(
    negative_sampling_rate: Double,
    factor: Int,
    bias: Double,
    fields: List[FMField]
  )

  private case class FMField(
    field: String,
    weight: Option[Double],
    vector: Option[Seq[Double]],
    values: Option[Seq[FMValue]]
  )

  private case class FMValue(
    index: String,
    weight: Double,
    vector: Seq[Double]
  )

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

  private[BinaryFactorizationMachinesModel] class BinaryFactorizationMachinesModelWriter(
    instance: BinaryFactorizationMachinesModel,
    attributes: Array[Attribute],
    decodeColumnName: Boolean
  ) extends MLWriter {

    lazy val decoder = Base64.getDecoder

    override protected def saveImpl(path: String): Unit = {
      val params = (attributes, instance.coefficients.toArray, instance.v.colIter.toList).zipped.toList
      val (numerics, binaries) = params.partition(_._1.attrType.name == "numeric")
      val fields = numerics.map { case (attr, v, vec) => FMField(attr.name.get, Some(v), Some(vec.toArray), None)} ++
        binaries.map { case (attr, v, vec) =>
          val split = attr.name.get.split("_", 2)
          (if (decodeColumnName) new String(decoder.decode(split(0))) else split(0), split(1), v, vec)
        }.groupBy(_._1).toList.map { case (k, vs) =>
          FMField(k, None, None, Some(vs.map { case (_, index, v, vec) => FMValue(index, v, vec.toArray)}))
        }
      val fmModel = FMModel(1.0, instance.v.numRows, instance.intercept, fields)
      sparkSession.createDataFrame(Seq(fmModel)).repartition(1).write.json(path)
    }
  }
}
