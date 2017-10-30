package example.param

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasBalancedWeight extends Params {

  final val balancedWeight: BooleanParam = new BooleanParam(this.uid, "balancedWeight", "")

  final def getBalancedWeight: Boolean = $(balancedWeight)
}
