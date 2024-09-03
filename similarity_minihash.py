from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
from sparksession_init import get_or_create
from pyspark.sql.functions import col
import os

spark = get_or_create(re=True)
spark.sql("USE zky_rem")


# 读取数据和分组信息
user_data_counts = spark.sql("SELECT * FROM zky_rem.user_data_counts")
book_features_df = spark.sql("SELECT * FROM zky_rem.book_features")

# 计算每组的书籍相似度
for group_id in user_data_counts.select("group_id").distinct().rdd.flatMap(lambda x: x).collect():
    print(f"Processing group {group_id}")

    # 获取该组的用户数据和书籍特征
    group_data = user_data_counts.filter(col("group_id") == group_id)
    group_book_ids = group_data.select("book_id").distinct()

    # 根据 group_id 筛选书籍特征
    group_book_features_df = book_features_df.join(
        group_book_ids, on="numeric_book_id", how="inner"
    ).distinct()

    # 计算 MinHashLSH
    model_path = f"hdfs://1.95.75.42:8020/user/models/ZKY/minhash_model_group_{group_id}"

    if not os.path.exists(model_path):
        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=2)
        mh_model = mh.fit(group_book_features_df)
        mh_model.write().overwrite().save(model_path)
    else:
        mh_model = MinHashLSHModel.load(model_path)

    # 计算相似度
    similarity_df = mh_model.approxSimilarityJoin(
        group_book_features_df, group_book_features_df, 0.8, distCol="similarity"
    ).select(
        col("datasetA.numeric_book_id").alias("book1_id"),
        col("datasetB.numeric_book_id").alias("book2_id"),
        col("similarity")
    ).filter(col("book1_id") != col("book2_id"))

    # 保存到 Hive
    similarity_df.write.mode("overwrite").saveAsTable(f"zky_rem.book_similarity_group_{group_id}")

spark.stop()
