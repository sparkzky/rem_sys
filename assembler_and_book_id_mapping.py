from sparksession_init import get_or_create
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import rand, row_number
from pyspark.sql.window import Window

spark = get_or_create()
spark.sql("USE zky_rem")

# 指定读取的数据库
database = "grl_rem.user_behavior_matrix"

# 读取数据表
user_behavior_df = spark.sql(
    f"SELECT uid,book_id,total_reading_duration,is_favorited,rating,share_count,total_comments FROM {database}")
user_behavior_df.show(5)

# 为每个用户分配组 ID
# 随机打乱用户数据并分组
user_behavior_with_group = user_behavior_df \
    .withColumn("random_order", rand()) \
    .withColumn("group_id", row_number().over(Window.orderBy("random_order")) % 10 + 1) \
    .drop("random_order")

# 保存用户数据和分组信息到 Hive 表
user_behavior_with_group.write.mode("overwrite").saveAsTable("user_data_counts")

# 创建 StringIndexer 实例，将 uid 列从字符串转换为数值类型
uid_indexer = StringIndexer(inputCol="uid", outputCol="numeric_uid")
user_behavior_df_indexed = uid_indexer.fit(user_behavior_df).transform(user_behavior_df)

# 创建 StringIndexer 实例，将 book_id 列从字符串转换为数值类型
book_indexer = StringIndexer(inputCol="book_id", outputCol="numeric_book_id")
user_behavior_df_indexed = book_indexer.fit(user_behavior_df_indexed).transform(user_behavior_df_indexed)

# user_behavior_df_indexed.write.mode("overwrite").saveAsTable("zky_rem.user_behavior_indexed")

# 选择特征列并创建特征向量
assembler = VectorAssembler(
    inputCols=["rating", "is_favorited", "total_comments", "share_count", "total_reading_duration"],
    outputCol="features"
)

# 在这里根据 assembler 计算出每一个特征向量
user_behavior_df_features = assembler.transform(user_behavior_df_indexed)
user_behavior_df_features.show(5)

# 创建 ID 映射表
book_id_mapping = user_behavior_df_indexed.select("book_id", "numeric_book_id").distinct()
book_id_mapping = book_id_mapping.withColumnRenamed("numeric_book_id", "recommend_numeric_book_id")
book_id_mapping.write.mode("overwrite").saveAsTable("zky_rem.book_id_mapping")

# 计算每本书的特征向量
book_features_df = user_behavior_df_features.select("numeric_book_id", "features").distinct()
book_features_df.show(5)

book_features_df.write.mode("overwrite").saveAsTable("zky_rem.book_features")

spark.stop()

# 还有增量的
