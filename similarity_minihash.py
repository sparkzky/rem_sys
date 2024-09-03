from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
from sparksession_init import get_or_create
from pyspark.sql.functions import col
import os

model_path = "hdfs://1.95.75.42:8020/user/models/minhash_model"

spark = get_or_create(re=True)

book_features_df = spark.sql("SELECT * FROM zky_rem.book_features limit 20000")
book_features_df.show(5, truncate=False)
book_features_df = book_features_df.repartition(200).cache()

# 使用 MinHash 计算近似相似度
# 如果模型已经存在，则加载，否则训练新模型并保存
if not os.path.exists(model_path):
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=2)
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
# similarity_df.show(5, truncate=False)
spark.stop()
