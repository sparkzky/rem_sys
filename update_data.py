from sparksession_init import get_or_create
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import rand, row_number
from pyspark.sql.window import Window

# 初始化Spark会话
spark = get_or_create()
spark.sql("USE zky_rem")

# 指定读取的数据库
database = "grl_rem.user_behavior_matrix"
id = 1000
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


# 增量数据处理逻辑
def process_incremental_data(start_id):
    # 过滤增量数据
    incremental_data = user_behavior_df_indexed.filter(f"numeric_uid >= {start_id}")

    # 选择特征列并创建特征向量
    assembler = VectorAssembler(
        inputCols=["rating", "is_favorited", "total_comments", "share_count", "total_reading_duration"],
        outputCol="features"
    )
    incremental_features = assembler.transform(incremental_data)

    # 计算增量数据的特征向量和ID映射
    book_id_mapping_increment = incremental_data.select("book_id", "numeric_book_id").distinct()
    book_features_increment = incremental_features.select("numeric_book_id", "features").distinct()

    # 插入增量数据到现有的Hive表
    book_id_mapping_increment.write.mode("append").saveAsTable("zky_rem.book_id_mapping")
    book_features_increment.write.mode("append").saveAsTable("zky_rem.book_features")


# 假设指定的ID为1000，执行增量处理
process_incremental_data(id)

spark.stop()
