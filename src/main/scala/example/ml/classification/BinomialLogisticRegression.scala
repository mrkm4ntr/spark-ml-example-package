package example.ml.classification

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{DenseVector, Matrix, SparseVector, Vector}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset


trait BinomialLogisticRegressionParams extends Params {
}

class BinomialLogisticRegression(override val uid: String)
  extends ProbabilisticClassifier[Vector, BinomialLogisticRegression, BinomialLogisticRegressionModel]
    with BinomialLogisticRegressionParams with Logging {

  def this() = this(Identifiable.randomUID("logisticregression"))

  override def copy(extra: ParamMap): BinomialLogisticRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BinomialLogisticRegressionModel = ???
}

object BinomialLogisticRegression {
}

class BinomialLogisticRegressionModel(
  override val uid: String,
  val coefficients: Matrix,
  val intercepts: Vector
) extends ProbabilisticClassificationModel[Vector, BinomialLogisticRegressionModel] with BinomialLogisticRegressionParams {
  

  override protected def predictRaw(features: Vector): Vector = Vectors.dense(
    coefficients.multiply(features).toArray.zip(intercepts.toArray).map { case(x, y) => x + y })

  override def copy(extra: ParamMap): BinomialLogisticRegressionModel = defaultCopy(extra)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = rawPrediction match {
    case dv: DenseVector => {
      for (i <- 0 until dv.values.size) {
        dv.values(i) = 1.0 / (1.0 + math.exp(-dv.values(i)))
      }
      dv
    }
    case _: SparseVector =>
      throw new RuntimeException("Unexpected error in BinomialLogisticRegressionModel: raw2probabilitiesInPlace encountered SparseVector")
  }

  override def numClasses: Int = 2
}
