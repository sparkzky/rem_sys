from pyspark.sql.functions import col, collect_list, udf, row_number
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from pyspark.sql.window import Window

from ZKY.result_al.sparksession_init import get_or_create

spark = get_or_create()
spark.sql("USE zky_rem")

# 读取数据和相似度结果
user_data_counts = spark.sql("SELECT * FROM zky_rem.user_data_counts")
book_id_mapping = spark.sql("SELECT * FROM zky_rem.book_id_mapping")

# 获取所有用户ID
user_ids = user_data_counts.select("uid").distinct()

# 定义空 DataFrame 的 schema
schema = StructType([
    StructField("uid", StringType(), True),
    StructField("recommended_books", ArrayType(StringType()), True)
])

# 初始化空的 DataFrame，用于存储所有用户的推荐结果
all_user_recommendations = spark.createDataFrame([], schema)

# 为每个用户生成推荐
for user_id_row in user_ids.collect():
    user_id = user_id_row['uid']

    # 获取特定用户喜欢的书籍
    user_liked_books = user_data_counts \
        .filter((col("uid") == user_id) & (col("is_favorited") == 1)) \
        .select("numeric_book_id") \
        .distinct()

    # 获取用户所在的组
    user_group = user_data_counts.filter(col("uid") == user_id).select("group_id").distinct().collect()[0]["group_id"]

    # 获取该组的书籍相似度
    similarity_df = spark.sql(f"SELECT * FROM zky_rem.book_similarity_group_{user_group}")

    # 调整 join 操作，确保列名匹配
    user_liked_books_with_sim = user_liked_books.join(
        similarity_df,
        user_liked_books.numeric_book_id == similarity_df.book1_id
    )

    # 过滤掉用户已经喜欢的书籍
    filtered_df = user_liked_books_with_sim.filter(col("numeric_book_id") != col("book2_id"))

    # 定义窗口规格，按用户喜欢的书籍 ID 分组，并按照相似度降序排列
    window_spec = Window.partitionBy("numeric_book_id").orderBy(col("similarity").desc())

    # 使用窗口函数计算每对用户喜欢的书籍和推荐书籍的最大相似度
    ranked_df = filtered_df.withColumn("rank", row_number().over(window_spec))

    # 选择相似度最高的前 10 本推荐书籍
    user_recommendations = ranked_df.filter(col("rank") <= 10).select(
        col("book2_id").alias("recommended_book_id"),
        col("similarity")
    )

    # 进行 IO 映射
    recommendations_with_id = user_recommendations \
        .join(book_id_mapping,
              user_recommendations.recommended_book_id == book_id_mapping.recommend_numeric_book_id,
              how="left") \
        .select(
            col("book_id").alias("recommended_book_id")
        )

    # 将推荐的书籍ID收集为数组
    recommended_books_array = \
        recommendations_with_id.agg(collect_list("recommended_book_id").alias("recommended_books")).collect()[0][
            "recommended_books"]

    # 创建一个包含用户ID和推荐书籍ID数组的DataFrame
    user_recommendations_df = spark.createDataFrame([(user_id, recommended_books_array)], schema)

    # 将当前用户的推荐结果附加到总的结果DataFrame中
    all_user_recommendations = all_user_recommendations.union(user_recommendations_df)

# 显示所有用户的推荐结果
all_user_recommendations.show(truncate=False)

# 保存最终的推荐结果到 Hive 表
all_user_recommendations.write.mode("overwrite").saveAsTable("zky_rem.rem_result")

spark.stop()


