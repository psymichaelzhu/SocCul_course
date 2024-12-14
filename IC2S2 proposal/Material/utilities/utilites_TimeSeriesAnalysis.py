from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.types import DoubleType
from prophet import Prophet
from prophet.utilities import regressor_coefficients
import pandas as pd

# visualize centroid trajectory
def plot_centroid_trajectory(df_centroids):
    """
    Plot the trajectory of centroids over years in the first two PCA dimensions.
    Colors transition from blue (early years) to yellow (recent years).
    """
    # Convert to pandas for plotting
    centroids_pd = df_centroids.toPandas()
    
    # Extract first two dimensions from centroid arrays
    centroids_pd['x'] = centroids_pd['centroid'].apply(lambda x: x[0])
    centroids_pd['y'] = centroids_pd['centroid'].apply(lambda x: x[1])
    
    # Create color gradient based on years
    years = centroids_pd['year'].values
    min_year, max_year = years.min(), years.max()
    colors = plt.cm.viridis((years - min_year) / (max_year - min_year))
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(centroids_pd['x'], centroids_pd['y'], 
               c=colors, s=100, alpha=0.6)
    
    # Add year labels
    for i, row in centroids_pd.iterrows():
        plt.annotate(str(int(row['year'])), 
                    (row['x'], row['y']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    
    plt.xlabel('First PCA Component')
    plt.ylabel('Second PCA Component')
    plt.title('Centroid Trajectory Over Years')
    plt.grid(True, alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                label='Year', 
                ticks=[0, 1],
                boundaries=np.linspace(0, 1, 100),
                values=np.linspace(0, 1, 99))
    plt.tight_layout()
    plt.show()

def plot_centroid_quiver(df_centroids):
    """
    Plot quiver diagram showing centroid movements between consecutive years.
    Arrows start from origin, colors transition from blue (early years) to yellow (recent years).
    """
    # Convert to pandas for plotting
    centroids_pd = df_centroids.toPandas()
    
    # Extract first two dimensions from centroid arrays
    centroids_pd['x'] = centroids_pd['centroid'].apply(lambda x: x[0])
    centroids_pd['y'] = centroids_pd['centroid'].apply(lambda x: x[1])
    
    # Sort by year to ensure correct order
    centroids_pd = centroids_pd.sort_values('year')
    
    # Calculate yearly movement vectors
    movement_vectors = []
    movement_years = []
    
    for i in range(1, len(centroids_pd)):
        prev_year = centroids_pd['year'].iloc[i-1]
        curr_year = centroids_pd['year'].iloc[i]
        
        # Calculate yearly movement vector
        yearly_vector = np.array([
            centroids_pd['x'].iloc[i] - centroids_pd['x'].iloc[i-1],
            centroids_pd['y'].iloc[i] - centroids_pd['y'].iloc[i-1]
        ])
        movement_vectors.append(yearly_vector)
        movement_years.append(curr_year)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Create color gradient based on years
    num_years = len(movement_years)
    colors = plt.cm.viridis(np.linspace(0, 1, num_years))
    
    # Plot movement vectors from origin
    for i, vector in enumerate(movement_vectors):
        plt.quiver(0, 0, vector[0], vector[1], 
                  angles='xy', scale_units='xy', scale=1,
                  color=colors[i], alpha=0.7)
    
    # Add colorbar to show year progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(movement_years), 
                                               vmax=max(movement_years)))
    plt.colorbar(sm, label='Year')
    
    plt.xlabel('First Principal Component Change')
    plt.ylabel('Second Principal Component Change')
    plt.title('Yearly Movement Vectors in Music Style Space')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    
    plt.tight_layout()
    plt.show()



def prepare_yearly_distributions(df_bin_counts):
    """
    Convert Spark DataFrame with bin counts to pandas DataFrame format for analysis.
    Expected columns: year, bin, count
    Returns pandas DataFrame with yearly distributions in wide format
    """
    # Convert to pandas
    yearly_distributions_pd = df_bin_counts.toPandas()
    yearly_distributions_pd = yearly_distributions_pd.dropna()
    
    # Filter bin_index to only include first NUM_bins bins
    yearly_distributions_pd = yearly_distributions_pd[
        yearly_distributions_pd['bin_index'].between(0, NUM_bins)
    ]
    
    # Sort by year and bin for consistency
    yearly_distributions_pd = yearly_distributions_pd.sort_values(['year', 'bin_index'])
    
    # Pivot table using distance_bin instead of bin_index
    yearly_distributions_wide = yearly_distributions_pd.pivot(
        index='year',
        columns='distance_bin',
        values='track_count'
    ).reset_index()
    
    # Rename columns for clarity
    yearly_distributions_wide.columns.name = None
    
    return yearly_distributions_wide


def plot_yearly_distributions(yearly_distributions_pd):
    plt.figure(figsize=(12, 6))
    
    # Get number of bins (excluding year column)
    n_bins = len(yearly_distributions_pd.columns) - 1
    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))
    
    # Plot each distance bin as a separate line
    for i, col in enumerate(yearly_distributions_pd.columns):
        if col != 'year':
            plt.plot(yearly_distributions_pd['year'], 
                    yearly_distributions_pd[col],
                    marker='o',
                    linewidth=2,
                    color=colors[i-1],
                    label=col)  # Use actual distance range as label

    plt.xlabel('Year')
    plt.ylabel('Track Count')
    plt.title('Track Counts by Distance Range Over Years')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Distance Range')
    plt.grid(True, alpha=0.3)
    # Add gray background for 1999-2001 streaming transition period
    plt.axvspan(1999, 2001, color='lightgray', alpha=0.3, label='Streaming Transition')
    plt.tight_layout()
    plt.show()


# Prophet model for interrupted time series analysis
def analyze_and_plot_regressors(yearly_distributions_pd):
    # Store results
    results = {
        'intercepts': [],
        'intercepts_lower': [],
        'intercepts_upper': [],
        'slopes': [],
        'slopes_lower': [],
        'slopes_upper': [],
        'levels': [],
        'models': []  # Add storage for model objects
    }

    # Define holidays for known events
    holidays_df = pd.DataFrame({
        'holiday': 'innovation_peak',
        'ds': pd.to_datetime(['1966-01-01', '1993-01-01']),
        'lower_window': -2,
        'upper_window': 2,
    })

    

    # Get distance bins (excluding 'year' column)
    distance_bins = [col for col in yearly_distributions_pd.columns if col != 'year']
    
    # Iterate over each distance bin
    for distance_bin in distance_bins:
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(yearly_distributions_pd['year'].astype(str) + '-01-01'),
            'y': yearly_distributions_pd[distance_bin]
        })
        # Add streaming regressors
        df_prophet['streaming'] = (df_prophet['ds'].dt.year >= 2000).astype(int)
        df_prophet['year_since_streaming'] = (df_prophet['ds'].dt.year - 2000).clip(lower=0)
        
        # Initialize and fit Prophet model
        model = Prophet(
            mcmc_samples=500,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.001,
        )
        
        # Add seasonality and regressors
        model.add_seasonality(
            name='40_year_cycle',
            period=40*365.25,
            fourier_order=3
        )
        model.add_regressor('streaming')
        model.add_regressor('year_since_streaming')

        model.fit(df_prophet)
        
        # Store model object
        results['models'].append(model)

        # Extract regressor coefficients with confidence intervals
        coef_df = regressor_coefficients(model)
        
        # Store streaming effect (intercept)
        streaming_coef = coef_df.loc[coef_df['regressor'] == 'streaming'].iloc[0]
        results['intercepts'].append(streaming_coef['coef'])
        results['intercepts_lower'].append(streaming_coef['coef_lower'])
        results['intercepts_upper'].append(streaming_coef['coef_upper'])
        
        # Store year_since_streaming effect (slope)
        slope_coef = coef_df.loc[coef_df['regressor'] == 'year_since_streaming'].iloc[0]
        results['slopes'].append(slope_coef['coef'])
        results['slopes_lower'].append(slope_coef['coef_lower'])
        results['slopes_upper'].append(slope_coef['coef_upper'])
        
        results['levels'].append(distance_bin)

    # Create visualization for each distance bin
    for i, distance_bin in enumerate(distance_bins):
        model = results['models'][i]
        
        # Make predictions
        future = model.make_future_dataframe(periods=0, freq='Y')
        future['streaming'] = (future['ds'].dt.year >= 2000).astype(int)
        future['year_since_streaming'] = (future['ds'].dt.year - 2000).clip(lower=0)
        forecast = model.predict(future)
        
        # Plot components
        fig = model.plot_components(forecast)
        fig.set_size_inches(12, 8)
        plt.suptitle(f'Model Components for Distance Range {distance_bin}')
        plt.tight_layout()
        plt.show()
        
        # Plot predictions vs actual values
        model.plot(forecast, figsize=(12, 8))
        plt.xlabel('Year')
        plt.ylabel('Track Count')
        plt.title(f'Track Count Trajectory for Distance Range {distance_bin}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot streaming effects with confidence intervals
    plt.figure(figsize=(12, 6))

    # Plot intercept effect with confidence interval
    plt.fill_between(results['levels'], 
                    results['intercepts_lower'], 
                    results['intercepts_upper'], 
                    alpha=0.2, 
                    color='#5185B8',
                    label='Intercept 95% CI')
    plt.plot(results['levels'], 
            results['intercepts'], 
            '#5185B8', 
            label='Streaming Intercept Effect', 
            marker='o')

    # Plot slope effect with confidence interval
    plt.fill_between(results['levels'], 
                    results['slopes_lower'], 
                    results['slopes_upper'], 
                    alpha=0.2, 
                    color='#EF8636',
                    label='Slope 95% CI')
    plt.plot(results['levels'], 
            results['slopes'], 
            '#EF8636', 
            label='Streaming Slope Effect', 
            marker='o')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Distance Range')
    plt.ylabel('Coefficient Value')
    plt.title('Streaming Effects Across Distance Ranges with 95% Confidence Intervals')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results
