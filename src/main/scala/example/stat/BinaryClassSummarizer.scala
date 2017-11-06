package example.stat

import scala.collection.mutable

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
