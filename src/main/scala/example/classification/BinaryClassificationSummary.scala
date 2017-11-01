package example.classification

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

class BinaryClassificationSummary(
  val predictions: DataFrame,
  val probabilityCol: String,
  val labelCol: String,
  val featuresCol: String
) extends Serializable {

  @transient private val binaryMetrics = new BinaryClassificationMetrics(
    predictions.select(col(probabilityCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(score: Vector, label: Double) => (score(0), label)
    }, 100
  )

  lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()
}
