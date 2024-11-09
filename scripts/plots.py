from collections import Counter
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from wordcloud import WordCloud

#------------------------------------------------------------------------------
# Download required data and configure plot styling.
#------------------------------------------------------------------------------

# Download required NLTK data.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set plot styling.
sns.set_theme(palette='Greys_r')
sns.set_style("whitegrid")

#------------------------------------------------------------------------------
# Load input data.
#------------------------------------------------------------------------------

full_data = pd.read_csv('../data/data_60_21.csv')
data_60_83 = pd.read_csv('../data/data_60_83.csv')
data_84_21 = pd.read_csv('../data/data_84_21.csv')
yearly_counts = pd.read_csv('../data/speech_counts.csv', index_col='year')
aggregate_data = pd.read_csv('../data/aggregate_data.csv', index_col='year_month')
text_data = pd.read_csv('../data/text.csv')
labels = pd.read_csv('../data/labels.csv')

#------------------------------------------------------------------------------
# Define functions.
#------------------------------------------------------------------------------

def generate_histogram(counts: pd.DataFrame) -> None:
    """
    Generate histogram of speech counts by year.

    Args:
        counts: DataFrame containing yearly speech counts.
    """
    plt.figure(figsize=(12, 5))
    plt.bar(counts.index, yearly_counts['counts'], color='black', width=0.8)
    plt.ylabel('Count')
    plt.xlabel('Year')
    plt.xticks(rotation=45, ha='right')
    plt.xlim(counts.index.min() - 0.5, yearly_counts.index.max() + 0.5)
    plt.tight_layout()
    plt.savefig('../results/plots/speech_count.eps')


def truncate_colormap(cmap,
                      min_val: float = 0.0,
                      max_val: float = 1.0,
                      n: int = 100) -> LinearSegmentedColormap:
    """
    Truncate a colormap to specified range.

    Args:
        cmap: Original colormap.
        min_val: Minimum value in range [0.0, 1.0].
        max_val: Maximum value in range [0.0, 1.0].
        n: Number of colors in truncated map.
        
    Returns:
        Truncated colormap.
    """
    return LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )


def get_concern(df: pd.DataFrame,
                start_year: int,
                end_year: int,
                term: str,
                color: str = None) -> None:
    """
    Generate word cloud visualization of speaker concerns.
    
    Args:
        df: DataFrame containing text data.
        start_year: Start year for filtering.
        end_year: End year for filtering.
        term: Column name for filtering data.
        color: Optional colormap name.
    """
    # Filter data.
    data_filtered = df[df[term] < 0.0].copy()
    concern_data = data_filtered[
        (data_filtered['year'] >= start_year) &
        (data_filtered['year'] <= end_year)
    ].copy()
   
    # Process text.
    concern_text = ' '.join(concern_data['concern_type'].astype(str))
    tokens = nltk.word_tokenize(concern_text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize tokens.
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Generate n-grams.
    n_gram_ranges = [1, 2, 3]
    all_grams = []
    for n in n_gram_ranges:
        all_grams.extend([
            ' '.join(gram) for gram in ngrams(lemmatized_tokens, n) 
            if all(word not in stop_words for word in gram)
        ])

    # Count frequencies.
    gram_frequencies = Counter(all_grams)

    # Set up colormap.
    base_cmap = plt.get_cmap('gray')
    cmap = truncate_colormap(base_cmap, 0.0, 0.8) if not color else color

    # Generate word cloud.
    wordcloud = WordCloud(
        width=3000,
        height=1000,
        random_state=1,
        background_color='white',
        colormap=cmap,
        collocations=False,
        stopwords=stop_words
    ).generate_from_frequencies(gram_frequencies)

    # Plot and save.
    plt.figure(figsize=(30, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    
    filename = f'../results/plots/ngram_cloud_{start_year}_{end_year}_{term.replace(" ", "_")}.eps'
    plt.savefig(filename)


def plot_series(df: pd.DataFrame,
                term: str,
                short: bool = False) -> None:
    """
    Plot time series with rolling mean.

    Args:
        df: DataFrame containing time series data.
        term: Column name to plot.
        short: If True, use shorter time range.
    """
    plt.figure()
    df.index = pd.to_datetime(df.index)
    
    # Standardize values.
    df[term] = (df[term] - df[term].mean()) / df[term].std()
    
    # Plot based on time range.
    if not short:
        df[term].rolling(24).mean().plot(
            figsize=(15,5),
            xlim=(pd.Timestamp('1962-01-01'), pd.Timestamp('2020-01-01'))
        )
    else:
        df[term].rolling(24).mean().plot(
            figsize=(15,5),
            xlim=(pd.Timestamp('1963-04-01'), df.index[-1].strftime('%Y-%m-%d'))
        )
        
    plt.xlabel('Year')
    plt.ylabel('Standardized Index Level')
    plt.savefig(f'../results/plots/{term.replace(" ","_")}.eps')


def plot_comparison(df: pd.DataFrame,
                    term1: str,
                    term2: str,
                    period: int) -> None:
    """
    Plot comparison between two time series.

    Args:
        df: DataFrame containing time series data.
        term1: First term to compare.
        term2: Second term to compare.
        period: Rolling window size.
    """
    plt.figure()
    
    # Prepare data.
    plot_df = df[[term1, term2]].copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    
    # Plot rolling means.
    plot_df.rolling(period, min_periods=5).mean().plot(
        figsize=(15,5),
        xlim=(pd.Timestamp('1960-01-01'), pd.Timestamp('2020-01-01'))
    )
    
    plt.xlabel('Year')
    plt.ylabel('Standardized Classification Score')
    plt.savefig(f'../results/plots/compare_{term1.replace(" ","_")}_{term2.replace(" ","_")}.eps')

def prepare_shap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Shapley analysis.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with transformed variables.
    """
    # Define variable list.
    variables = [
        'financial_crisis', 'inflation', 'employment', 'past',
        'present', 'future', 'academic', 'bank_capital', 'bank_liquidity', 'crisisJST',
        'lag_cpi_inflation', 'lag_output_gap', 'lag_stir', 'lag_debtgdp', 'lag_ltd',
        'total_loans', 'house_prices'
    ]

    # Transform variables.
    df['total_loans'] = np.log(df['lag_tloans'])
    df['house_prices'] = np.log(df['lag_hpnom'])

    # Include all dependent variables.
    dep_vars = ['financial_stability', 'cosine_bankreg_finstab', 'cosine_monetary_finstab']
    data_shap = df[variables + dep_vars].copy()

    # Rename columns.
    variable_mapping = {
        'financial_crisis': 'financial_crisis_text',
        'inflation': 'inflation_text', 
        'employment': 'employment_text',
        'past': 'past_text',
        'present': 'present_text',
        'future': 'future_text',
        'academic': 'academic_text',
        'bank_capital': 'bank_capital_text',
        'bank_liquidity': 'bank_liquidity_text',
        'crisisJST': 'crisis_dummy',
        'lag_cpi_inflation': 'cpi_inflation',
        'lag_output_gap': 'output_gap', 
        'lag_stir': 'short_rate',
        'lag_debtgdp': 'debt_to_gdp',
        'lag_ltd': 'loan_to_deposit',
        'total_loans': 'total_loans',
        'house_prices': 'house_prices'
    }

    return data_shap.rename(columns=variable_mapping)

def estimate_shap_model(data: pd.DataFrame,
                        dependent_variable: str) -> tuple:
    """
    Estimate model for Shapley analysis.
    
    Args:
        data: Prepared DataFrame from prepare_shap_data
        dependent_variable: Dependent variable name
        
    Returns:
        reg_model: Estimated model
    """
    reg_model = sklearn.linear_model.LinearRegression()
    
    # Get relevant columns.
    dep_vars = ['financial_stability', 'cosine_bankreg_finstab', 'cosine_monetary_finstab']
    variable_cols = [col for col in data.columns if col not in dep_vars]
    
    # Create working copy with only relevant columns.
    working_data = data[[dependent_variable] + variable_cols].copy()
    
    # Drop NaNs.
    working_data.dropna(subset=[dependent_variable] + variable_cols, inplace=True)
    
    # Split explanatory and dependent variables.
    X = working_data[variable_cols]
    y = working_data[dependent_variable]
    
    # Estimate model.
    reg_model.fit(X, y)
    
    return reg_model

def gen_bee_plot(dependent_variable: str,
                 reg_model,
                 df: pd.DataFrame,
                 n_variables: int = 5) -> None:
    """
    Generate SHAP beeswarm plot for most important variables.
    """
    # Get variables, excluding dependent variables.
    dep_vars = ['financial_stability', 'cosine_bankreg_finstab', 'cosine_monetary_finstab']
    variable_cols = [col for col in df.columns if col not in dep_vars]
    
    # Use only explanatory variable columns for SHAP analysis.
    X = df[variable_cols].copy()
    explainer = shap.LinearExplainer(reg_model, X, seed=42)
    shap_values = explainer(X)
    
    # Calculate mean absolute SHAP values for variable importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    variable_importance = pd.Series(mean_abs_shap, index=variable_cols)
    top_variables = variable_importance.nlargest(n_variables).index
    
    # Filter SHAP values to recover top variables.
    variable_indices = [variable_cols.index(f) for f in top_variables]
    filtered_shap_values = shap.Explanation(
        values=shap_values.values[:, variable_indices],
        base_values=shap_values.base_values,
        data=shap_values.data[:, variable_indices],
        feature_names=list(top_variables)
    )

    # Generate plot.
    plt.figure(figsize=(12, 12))
    plt.style.use('default')
    shap.plots.beeswarm(
        filtered_shap_values,
        show=False,
        color='black'
    )

    # Save plot.
    plt.tight_layout()
    plt.savefig(f'../results/plots/beeswarm_shap_{dependent_variable}.pdf')
    plt.close()

#------------------------------------------------------------------------------
# Generate descriptive statistics.
#------------------------------------------------------------------------------

# Generate descriptive statistics for full sample.
full_data[['financial_stability', 'inflation', 'employment', 'financial_crisis',
      'bank_liquidity', 'bank_capital', 'past', 'present', 'future', 'academic',
      'cosine_monetary_finstab', 'cosine_bankreg_finstab']].describe().round(3).to_csv(
          '../results/tables/text_feature_descriptives.csv'
)

# Generate district-level statistics.
full_data.groupby('district_code')[
    ['past', 'present', 'future', 'financial_stability', 'academic']
].mean().round(3).to_csv('../results/tables/district_descriptives.csv')


#------------------------------------------------------------------------------
# Generate figures.
#------------------------------------------------------------------------------

# Generate histogram of yearly speech counts.
generate_histogram(yearly_counts)

# Generate text feature time series plots.
plot_series(aggregate_data, 'dual_mandate_score')
plot_series(aggregate_data, 'cosine_monetary_finstab')
plot_series(aggregate_data, 'cosine_bankreg_finstab')
plot_series(aggregate_data, 'financial_crisis')

# Plot future-past difference.
plt.figure(figsize=(15,5))
(aggregate_data['future'] - aggregate_data['past']).rolling(24, min_periods=5).mean().plot()
plt.xlabel('Year')
plt.xlim(pd.Timestamp('1962-01-01'), pd.Timestamp('2018-01-01'))
plt.ylim(-2, 1.25)
plt.ylabel('Difference in Standardized Index Levels')
plt.savefig('../results/plots/diff_future_past.eps')

# Plot financial stability comparison.
plt.figure(figsize=(15,5))
for series, style in [
    ('financial_stability', '-'),
    ('inflation', '--'),
    ('output_growth', ':')
]:
    sns.lineplot(
        data=aggregate_data[series].rolling(24, min_periods=5).mean(),
        label=series,
        linestyle=style
    )
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.xlabel('Year')
plt.xlim(pd.Timestamp('1962-01-01'), pd.Timestamp('2018-01-01'))
plt.ylabel('Standardized Index Levels')
plt.savefig('../results/plots/index_plots_finstab_inflation_output.eps', bbox_inches='tight')

# Plot tense comparison.
plt.figure(figsize=(15,5))
for series, style in [('past', '-'), ('present', '--'), ('future', ':')]:
    sns.lineplot(
        data=aggregate_data[series].rolling(24, min_periods=5).mean(),
        label=series,
        linestyle=style
    )
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.xlabel('Year')
plt.xlim(pd.Timestamp('1962-01-01'), pd.Timestamp('2018-01-01'))
plt.ylim(-1.5, 1.0)
plt.ylabel('Standardized Index Levels')
plt.savefig('../results/plots/tense_plots.eps', bbox_inches='tight')

# Generate word clouds.
get_concern(text_data, 1960, 1983, 'dual_mandate_score')
get_concern(text_data, 1984, 2017, 'dual_mandate_score')

# Define dependent variables for SHAP analysis.
DEP_VARS = [
    'financial_stability',
    'cosine_bankreg_finstab',
    'cosine_monetary_finstab'
]

# Prepare data.
prepared_data = prepare_shap_data(full_data)

# Estimate model for each dependent variable.
for dep_var in DEP_VARS:
    shap_reg_model = estimate_shap_model(prepared_data, dep_var)
    gen_bee_plot(dep_var, shap_reg_model, prepared_data, n_variables=5)
