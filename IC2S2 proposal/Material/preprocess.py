from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from datetime import date

spark = SparkSession \
        .builder \
        .appName("Preprocess") \
        .getOrCreate()

# Load Spotify data
df_raw = spark.read.csv('/home/mikezhu/music/data/spotify2020.csv', header=True) 
print("number of rows (raw):",df_raw.count())
df_raw.show(5, truncate=False)
df_raw.printSchema()

# %% preprocess relevant columns
df=df_raw
# numeric columns
for col in ['popularity','duration_ms','tempo','key','time_signature','mode','danceability','energy','valence','acousticness','instrumentalness','loudness','speechiness','liveness']:
    df=df.withColumn(col,F.col(col).cast('float'))
# duration_ms unit to minutes
df=df.withColumn('duration',F.col('duration_ms')/60000).drop('duration_ms')
# date column, construct date range
df=df.withColumn('release_date',F.to_date(F.col('release_date'))).filter(
    (F.col("release_date") >= F.lit(date(1920, 1, 1))) &
    (F.col("release_date") <= F.lit(date(2020, 12, 31)))
).withColumn('year',F.year('release_date'))
# drop na
df=df.dropna()
# delete audio storys * this is a data-driven criterion
df = df.filter(
    ~((F.col('instrumentalness') < 0.001) & (F.col('speechiness') > 0.75))
)

# %% Save preprocessed dataframe as parquet
df.write.mode('overwrite').parquet('/home/mikezhu/music/data/spotify2020_preprocessed.parquet')