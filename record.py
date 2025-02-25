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

from sparksession_init import get_or_create
from pyspark.ml.feature import StringIndexer, VectorAssembler

spark = get_or_create()

# 读取数据表
user_behavior_df = spark.sql("SELECT uid,book_id,total_reading_duration,is_favorited,rating,comment_count,share_count,interaction_count,comment_likes FROM businessdata.expand_data")
user_behavior_df.show(5)

# 创建 StringIndexer 实例，将 uid 列从字符串转换为数值类型
uid_indexer = StringIndexer(inputCol="uid", outputCol="numeric_uid")
user_behavior_df_indexed = uid_indexer.fit(user_behavior_df).transform(user_behavior_df)

# 创建 StringIndexer 实例，将 book_id 列从字符串转换为数值类型
book_indexer = StringIndexer(inputCol="book_id", outputCol="numeric_book_id")
user_behavior_df_indexed = book_indexer.fit(user_behavior_df_indexed).transform(user_behavior_df_indexed)
user_behavior_df_indexed.write.mode("overwrite").saveAsTable("user_behavior_indexed")

# 选择特征列并创建特征向量
assembler = VectorAssembler(
    inputCols=["rating", "is_favorited", "total_comments", "share_count", "total_reading_duration"],
    outputCol="features"
)

# 在这里根据 assembler 计算出每一个特征向量
user_behavior_df_features = assembler.transform(user_behavior_df_indexed)
# user_behavior_df_features.show(5)

# 创建 ID 映射表
book_id_mapping = user_behavior_df_indexed.select("book_id", "numeric_book_id").distinct()
book_id_mapping = book_id_mapping.withColumnRenamed("numeric_book_id", "recommend_numeric_book_id")
book_id_mapping.write.mode("overwrite").saveAsTable("book_id_mapping")

# 计算每本书的特征向量
book_features_df = user_behavior_df_features.select("numeric_book_id", "features").distinct()
# book_features_df.show(5)

book_features_df.write.mode("overwrite").saveAsTable("zky_rem.book_features")

spark.stop()

from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
from sparksession_init import get_or_create
from pyspark.sql.functions import col
import os

model_path = "hdfs://1.95.75.42:8020/user/models/minhash_model"

spark = get_or_create(re=True)

book_features_df = spark.sql("SELECT * FROM zky_rem.book_features")
book_features_df.show(5, truncate=False)
book_features_df = book_features_df.repartition(200).cache()

# 使用 MinHash 计算近似相似度
# 如果模型已经存在，则加载，否则训练新模型并保存
if not os.path.exists(model_path):
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)
    mh_model = mh.fit(book_features_df)
    mh_model.write().overwrite().save(model_path)
else:
    mh_model = MinHashLSHModel.load(model_path)

# 使用 MinHash 模型进行相似度计算
similarity_df = mh_model.approxSimilarityJoin(
    book_features_df, book_features_df, 0.8, distCol="similarity"
).select(
    col("datasetA.numeric_book_id").alias("book1_id"),
    col("datasetB.numeric_book_id").alias("book2_id"),
    col("similarity")
).filter(col("book1_id") != col("book2_id"))

# # 保存 MinHash 模型到指定路径
# mh_model.write().overwrite().save(model_path)

# 显示计算后的相似度
# similarity_df.show(5,truncate=False)

similarity_df.write.mode("overwrite").saveAsTable("zky_rem.book_similarity")
similarity_df.show(5, truncate=False)
spark.stop()


from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from sparksession_init import get_or_create
from pyspark.sql.functions import col, collect_list, array, struct
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

spark = get_or_create()

user_behavior_df_indexed = spark.sql("SELECT * FROM zky_rem.user_behavior_indexed")
similarity_df = spark.sql("SELECT * FROM zky_rem.book_similarity")
book_id_mapping = spark.sql("SELECT * FROM zky_rem.book_similarity")

# 获取所有用户ID
user_ids = user_behavior_df_indexed.select("uid").distinct()

# 定义空 DataFrame 的 schema
schema = StructType([
    StructField("uid", StringType(), True),
    StructField("recommended_books", ArrayType(StringType()), True)
])

# 初始化空的 DataFrame，用于存储所有用户的推荐结果
all_user_recommendations = spark.createDataFrame([], schema)

# 打印空的 DataFrame schema 以确认其结构
all_user_recommendations.printSchema()

# 为每个用户生成推荐
for user_id_row in user_ids.collect():
    user_id = user_id_row['uid']

    # 获取特定用户喜欢的书籍
    user_liked_books = user_behavior_df_indexed \
        .filter((col("uid") == user_id) & (col("is_favorited") == 1)) \
        .select("numeric_book_id") \
        .distinct()

    # 调整 join 操作，确保列名匹配，使用 broadcast join 优化性能
    user_liked_books_with_sim = user_liked_books.join(
        similarity_df.hint("broadcast"),
        user_liked_books.numeric_book_id == similarity_df.book1_id
    )

    # 过滤掉用户已经喜欢的书籍
    filtered_df = user_liked_books_with_sim.filter(col("numeric_book_id") != col("book2_id"))

    # 定义窗口规格，按用户喜欢的书籍 ID 分组，并按照相似度降序排列
    window_spec = Window.partitionBy("numeric_book_id").orderBy(col("similarity").desc())

    # 使用窗口函数计算每对用户喜欢的书籍和推荐书籍的最大相似度
    ranked_df = filtered_df.withColumn("rank", row_number().over(window_spec))

    # 选择相似度最高的前 50 本推荐书籍
    user_recommendations = ranked_df.filter(col("rank") <= 50).select(
        col("book2_id").alias("recommended_book_id"),
        col("similarity")
    )

    # 进行 IO 映射，使用 broadcast join 优化性能
    recommendations_with_id = user_recommendations \
        .join(book_id_mapping.hint("broadcast"),
              user_recommendations.recommended_book_id == book_id_mapping.recommend_numeric_book_id,
              how="left") \
        .select(
        col("book_id").alias("recommended_book_id")
    )

    # 将推荐的书籍ID收集为数组
    recommended_books_array = \
        recommendations_with_id.agg(collect_list("recommended_book_id").alias("recommended_books")).collect()[0][
            "recommended_books"]

    from pyspark.sql.types import StructType, StructField, StringType, ArrayType

    # 定义 schema
    schema = StructType([
        StructField("uid", StringType(), True),
        StructField("recommended_books", ArrayType(StringType()), True)
    ])

    # 创建一个包含用户ID和推荐书籍ID数组的DataFrame
    user_recommendations_df = spark.createDataFrame([(user_id, recommended_books_array)], schema)

    # 将当前用户的推荐结果附加到总的结果DataFrame中
    all_user_recommendations = all_user_recommendations.union(user_recommendations_df)

# 显示所有用户的推荐结果
all_user_recommendations.show(truncate=False)

# 保存最终的推荐结果到 Hive 表
all_user_recommendations.write.mode("overwrite").saveAsTable("zky_rem.rem_result")

result = spark.sql("SELECT * FROM zky_rem.rem_result")
result.show(truncate=False)

spark.stop()