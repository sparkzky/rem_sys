from sparksession_init import get_or_create

spark = get_or_create()

spark.sql("SHOW DATABASES").show()
spark.sql("USE businessdata")
spark.sql("SHOW TABLES").show()
spark.sql("SELECT * FROM expand_data").show()

spark.stop()
