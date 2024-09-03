from pyspark import SparkConf
from pyspark.sql import SparkSession

# 设置 SparkConf 并调整 Spark 配置
conf = SparkConf()
conf.set("spark.executor.memory", "6g")
conf.set("spark.driver.memory", "6g")
conf.set("spark.executor.cores", "4")
conf.set("spark.submit.deployMode", "client")
conf.set("spark.hadoop.fs.defaultFS", "hdfs://1.95.75.42:8020")


def get_or_create(re=False):
    # 初始化 SparkSession
    if re:
        conf.set("spark.sql.shuffle.partitions", "200")

    spark = SparkSession.builder \
        .appName("Rem") \
        .master("yarn") \
        .config(conf=conf) \
        .enableHiveSupport() \
        .getOrCreate()
    return spark
