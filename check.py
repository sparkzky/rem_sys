from sparksession_init import get_or_create

spark = get_or_create()


spark.sql("USE businessdata")
spark.sql("SHOW TABLES").show()
table = spark.sql("SELECT * FROM expand_data")
table.show()
spark.stop()
