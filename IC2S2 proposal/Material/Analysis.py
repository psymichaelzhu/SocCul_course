#%%
import os
os.chdir('/home/mikezhu/Code')

from pyspark.sql import SparkSession, Row
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, ArrayType
from utilities.utilities_PCA import find_optimal_pca_components, analyze_pca_composition
from utilities.utilites_TimeSeriesAnalysis import plot_centroid_trajectory, plot_centroid_quiver, prepare_yearly_distributions, analyze_and_plot_regressors


spark = SparkSession \
        .builder \
        .appName("Clustering") \
        .getOrCreate()

# Load Spotify data
df = spark.read.parquet('/home/mikezhu/music/data/spotify2020_preprocessed.parquet', header=True) 
print(df.count())
df.printSchema()


# %% vectorize features

# identify potentially relevant features and add to a feature dataframe
feature_cols = [
    #'explicit',
    #'liveness', 
    'energy',
    'loudness',
    'danceability',
    'valence',
    'tempo',
    'time_signature',
    'acousticness',
    'instrumentalness',
    'mode',
    'duration',
    'key',
    'speechiness'
]

# select feature columns and numeric data as floats
df_features = df.select(*(F.col(c) for c in feature_cols),'id') \
                         .dropna()

df_features = df_features.withColumn('features', F.array(*[F.col(c) for c in feature_cols])) \
                         .select('id', 'features')

# convert features to dense vector format (expected by K-Means, PCA)
vectors = df_features.rdd.map(lambda row: Vectors.dense(row.features))
features = spark.createDataFrame(vectors.map(Row), ["features_unscaled"])

# scale features (some values like duration_ms are much larger than others)
standardizer = StandardScaler(inputCol="features_unscaled", outputCol="features")
model = standardizer.fit(features)
features = model.transform(features) \
                .select('features')

# persist in memory before fit model
features.persist()


# %% PCA: find optimal number of components
optimal_n, optimal_explained_variances, optimal_cumulative_variance, model_pca, features_pca = find_optimal_pca_components(features)
components_df = analyze_pca_composition(model_pca, feature_cols)

# merge PCA results
df_pca = features_pca.withColumn("tmp_id", F.monotonically_increasing_id()) \
            .join(df_features.withColumn("tmp_id", F.monotonically_increasing_id()), "tmp_id") \
            .drop("tmp_id") \
            .join(df,on=["id"],how="inner")

df_pca.show()
df_pca.count()
df_pca.printSchema()

# %% Time series analysis
# %% 1. Calculate centroids for each year

window_spec = Window.partitionBy('year')

# Convert pcaFeatures to array type for calculations
df_with_array = df_pca.withColumn('pca_array', F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))('pcaFeatures'))


# prepare data for time series analysis
# This funciton is put here because it needs spark session
def calculate_centroids(df_with_array, type='cohort', window_years=5):
    """
    Calculate centroids based on specified type:
    
    Args:
        df_with_array: DataFrame with PCA features as arrays
        type: Type of centroid calculation ('cohort', 'history', or 'recent')
        window_years: Number of years to look back for 'recent' type (default: 5)
        spark: SparkSession object (optional)
    
    Returns:
        Spark DataFrame with centroids
    """
    len_features = len(df_with_array.select('pca_array').first()[0])
    if type == 'cohort':
        # Original cohort-based calculation
        df_centroids = df_with_array.groupBy('year').agg(
            F.array([
                F.avg(F.element_at('pca_array', i+1)).alias(f'comp_{i}')
                for i in range(len_features)  
            ]).alias('centroid')
        )
    elif type == 'history':
        # Calculate historical centroids
        years = df_with_array.select('year').distinct().orderBy('year').collect()
        historical_centroids = []
        
        for year_row in years:
            year = year_row['year']
            historical_data = df_with_array.filter(F.col('year') <= year)
            
            centroid = historical_data.agg(
                F.array([
                    F.avg(F.element_at('pca_array', i+1)).alias(f'comp_{i}')
                    for i in range(len_features)
                ]).alias('centroid')
            ).first()['centroid']
            
            historical_centroids.append((year, centroid))
        
        # Convert back to Spark DataFrame
        df_centroids = spark.createDataFrame(historical_centroids, ['year', 'centroid'])
    
    elif type == 'recent':
        # Calculate recent centroids using sliding window
        years = df_with_array.select('year').distinct().orderBy('year').collect()
        recent_centroids = []
        
        for year_row in years[window_years:]:
            year = year_row['year']
            # Filter data for past X years (exclusive of current year)
            recent_data = df_with_array.filter(
                (F.col('year') <= year) & 
                (F.col('year') >= year - window_years)
            )
            
            centroid = recent_data.agg(
                F.array([
                    F.avg(F.element_at('pca_array', i+1)).alias(f'comp_{i}')
                    for i in range(len_features)
                ]).alias('centroid')
            ).first()['centroid']
            
            recent_centroids.append((year, centroid))
        
        # Convert back to Spark DataFrame
        df_centroids = spark.createDataFrame(recent_centroids, ['year', 'centroid'])
    
    return df_centroids


df_centroids = calculate_centroids(
    df_with_array=df_with_array,
    type='recent',
    window_years=2
)

# Plot the centroid trajectory
plot_centroid_trajectory(df_centroids)

# Plot the centroid quiver
plot_centroid_quiver(df_centroids)


# %% 2. Calculate distances
df_distance = df_with_array.join(df_centroids, on='year') \
    .withColumn('distance_to_centroid', euclidean_distance('pca_array', 'centroid'))

df_distance.orderBy('distance_to_centroid', ascending=True).show()

# %% 3. Create distance bins
interval_size = 1
df_distance_bin = df_distance.filter(F.col('distance_to_centroid') >= 1) \
    .withColumn(
        'distance_bin',
        F.concat(
            F.floor(F.col('distance_to_centroid') / F.lit(interval_size)) * F.lit(interval_size),
            F.lit(' - '),
            (F.floor(F.col('distance_to_centroid') / F.lit(interval_size)) + F.lit(1)) * F.lit(interval_size)
        )
    ).withColumn(
        'bin_index',
        F.floor(F.col('distance_to_centroid') / F.lit(interval_size)).cast('integer')
    )
# Show the result
df_distance_bin.select('name', 'artists', 'year', 'distance_to_centroid', 'distance_bin','bin_index').show(5)

# %% 4. Group by year and distance_bin to count tracks
NUM_bins=8
df_bin_counts = df_distance_bin.groupBy('year', 'bin_index','distance_bin') \
    .agg(F.count('*').alias('track_count')) \
    .orderBy('year', 'bin_index')

# Show the results
print("Track counts by year and distance bin:")
df_bin_counts.show(20)

yearly_distributions_pd = prepare_yearly_distributions(df_bin_counts)
print(yearly_distributions_pd.head())

# %% 5. Visualize yearly distributions
plot_yearly_distributions(yearly_distributions_pd)

# %% 6. fit Prophet model to conduct interrupted time series analysis
results = analyze_and_plot_regressors(yearly_distributions_pd)


# %%
