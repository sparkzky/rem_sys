from sparksession_init import get_or_create
from pyspark.sql.functions import col, collect_list, udf, size
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import random

spark = get_or_create()

user_behavior_df_indexed = spark.sql("SELECT * FROM zky_rem.user_behavior_indexed")
similarity_df = spark.sql("SELECT * FROM zky_rem.book_similarity")
book_id_mapping = spark.sql("SELECT * FROM zky_rem.book_similarity")

# 随机为用户分组
user_ids = user_behavior_df_indexed.select("uid").distinct()
user_ids = user_ids.withColumn("random_id", udf(lambda: random.randint(1, 1000), StringType())())
user_ids = user_ids.withColumn("row_number", row_number().over(Window.partitionBy("random_id").orderBy("uid")))

# # 获取所有用户ID
# user_ids = user_behavior_df_indexed.select("uid").distinct()

# 按组分配大小
partition_size = 20000
user_groups = user_ids.groupBy("random_id").agg(
    collect_list("uid").alias("user_list")
).filter(size(col("user_list")) <= partition_size)

# 定义空 DataFrame 的 schema
schema = StructType([
    StructField("uid", StringType(), True),
    StructField("recommended_books", ArrayType(StringType()), True)
])

# 初始化空的 DataFrame，用于存储所有用户的推荐结果
all_user_recommendations = spark.createDataFrame([], schema)

# 打印空的 DataFrame schema 以确认其结构
# all_user_recommendations.printSchema()

for group in user_groups.collect():
    user_list = group['user_list']

    # 创建一个DataFrame，其中包含当前组的所有用户数据
    current_users_df = user_behavior_df_indexed.filter(col("uid").isin(user_list))

    for user_id_row in user_list:
        user_id = user_id_row

        user_liked_books = current_users_df \
            .filter((col("uid") == user_id) & (col("is_favorited") == 1)) \
            .select("numeric_book_id") \
            .distinct()

        user_liked_books_with_sim = user_liked_books.join(
            similarity_df.hint("broadcast"),
            user_liked_books.numeric_book_id == similarity_df.book1_id
        )

        filtered_df = user_liked_books_with_sim.filter(col("numeric_book_id") != col("book2_id"))

        window_spec = Window.partitionBy("numeric_book_id").orderBy(col("similarity").desc())
        ranked_df = filtered_df.withColumn("rank", row_number().over(window_spec))
        user_recommendations = ranked_df.filter(col("rank") <= 50).select(
            col("book2_id").alias("recommended_book_id"),
            col("similarity")
        )

        recommendations_with_id = user_recommendations \
            .join(book_id_mapping.hint("broadcast"),
                  user_recommendations.recommended_book_id == book_id_mapping.recommend_numeric_book_id,
                  how="left") \
            .select(
            col("book_id").alias("recommended_book_id")
        )

        recommended_books_array = \
            recommendations_with_id.agg(collect_list("recommended_book_id").alias("recommended_books")).collect()[0][
                "recommended_books"]

        schema = StructType([
            StructField("uid", StringType(), True),
            StructField("recommended_books", ArrayType(StringType()), True)
        ])

        user_recommendations_df = spark.createDataFrame([(user_id, recommended_books_array)], schema)

        all_user_recommendations = all_user_recommendations.union(user_recommendations_df)

# 显示所有用户的推荐结果
all_user_recommendations.show(truncate=False)

# 保存最终的推荐结果到 Hive 表
all_user_recommendations.write.mode("overwrite").saveAsTable("zky_rem.rem_result")

# result = spark.sql("SELECT * FROM zky_rem.rem_result")
# result.show(truncate=False)

spark.stop()
