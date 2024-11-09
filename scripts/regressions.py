import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col

#------------------------------------------------------------------------------
# Load input data and define regression variables.
#------------------------------------------------------------------------------

full_data = pd.read_csv('../data/data_60_21.csv')
data_60_83 = pd.read_csv('../data/data_60_83.csv')
data_84_21 = pd.read_csv('../data/data_84_21.csv')

# Define regression variable ordering for output tables.
text_regression_vars = [
    'inflation', 'employment', 'financial_stability', 'financial_crisis',
    'bank_liquidity', 'bank_capital', 'past', 'present', 'future',
    'past*financial_stability', 'present*financial_stability',
    'future*financial_stability', 'academic', 'lag_debtgdp', 'lag_ltd'
]

returns_regression_vars = text_regression_vars.copy()

# Common base regression specification.
base_regression_spec = """
    C(crisisJST) + lag_cpi_inflation + lag_output_gap + lag_stir + lag_debtgdp + 
    np.log(lag_tloans) + np.log(lag_hpnom) + lag_ltd + financial_crisis + 
    inflation + employment + past + present + future + academic + 
    bank_capital + bank_liquidity + C(district_code)-1
"""

#------------------------------------------------------------------------------
# Financial stability text regressions.
#------------------------------------------------------------------------------

stability_full = sm.ols(
    f"financial_stability ~ {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

stability_early = sm.ols(
    f"financial_stability ~ {base_regression_spec}", 
    data=data_60_83
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_60_83['year_month'],
                'groups': data_60_83['district_code'],
                'maxlags': 4})

stability_late = sm.ols(
    f"financial_stability ~ {base_regression_spec}",
    data=data_84_21
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_84_21['year_month'],
                'groups': data_84_21['district_code'],
                'maxlags': 4})

#------------------------------------------------------------------------------
# Monetary policy advocacy regressions.
#------------------------------------------------------------------------------

monetary_full = sm.ols(
    f"cosine_monetary_finstab ~ {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

monetary_early = sm.ols(
    f"cosine_monetary_finstab ~ {base_regression_spec}",
    data=data_60_83
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_60_83['year_month'],
                'groups': data_60_83['district_code'],
                'maxlags': 4})

monetary_late = sm.ols(
    f"cosine_monetary_finstab ~ {base_regression_spec}",
    data=data_84_21
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_84_21['year_month'],
                'groups': data_84_21['district_code'],
                'maxlags': 4})

#------------------------------------------------------------------------------
# Bank regulation advocacy regressions.
#------------------------------------------------------------------------------

bankreg_full = sm.ols(
    f"cosine_bankreg_finstab ~ {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

bankreg_early = sm.ols(
    f"cosine_bankreg_finstab ~ {base_regression_spec}",
    data=data_60_83
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_60_83['year_month'],
                'groups': data_60_83['district_code'],
                'maxlags': 4})

bankreg_late = sm.ols(
    f"cosine_bankreg_finstab ~ {base_regression_spec}",
    data=data_84_21
).fit(cov_type='hac-panel',
      cov_kwds={'time': data_84_21['year_month'],
                'groups': data_84_21['district_code'],
                'maxlags': 4})

# Generate and save summary table of text regressions.
text_regression_table = summary_col(
    [stability_full, stability_early, stability_late,
     monetary_full, monetary_early, monetary_late,
     bankreg_full, bankreg_early, bankreg_late],
    float_format='%0.4f',
    stars=True,
    info_dict={
        'R2': lambda x: f"{x.rsquared_adj:.4f}",
        'N': lambda x: f"{x.nobs:.4f}"
    },
    regressor_order=text_regression_vars
).as_latex()

np.savetxt('../results/regressions/text_regressions.tex', [text_regression_table], fmt='%s')

#------------------------------------------------------------------------------
# Asset return regressions.
#------------------------------------------------------------------------------

# Equity returns.
equity_base = sm.ols(
    f"lead_eq_tr ~ financial_stability + {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

equity_interaction = sm.ols(
    f"""lead_eq_tr ~ financial_stability + {base_regression_spec} + 
    past*financial_stability + present*financial_stability + future*financial_stability""",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

# Bonds returns.
bond_returns = sm.ols(
    f"lead_bond_tr ~ financial_stability + {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

# Risky returns.
risky_returns = sm.ols(
    f"lead_risky_tr ~ financial_stability + {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

# Safe returns.
safe_returns = sm.ols(
    f"lead_safe_tr ~ financial_stability + {base_regression_spec}",
    data=full_data
).fit(cov_type='hac-panel',
      cov_kwds={'time': full_data['year_month'],
                'groups': full_data['district_code'],
                'maxlags': 4})

# Generate and save summary table of returns prediction regressions.
returns_regression_table = summary_col(
    [equity_base, equity_interaction, bond_returns, risky_returns, safe_returns],
    float_format='%0.4f',
    stars=True,
    info_dict={
        'R2': lambda x: f"{x.rsquared_adj:.4f}",
        'N': lambda x: f"{x.nobs:.4f}"
    },
    regressor_order=returns_regression_vars
).as_latex()

np.savetxt('../results/regressions/returns_regressions.tex', [returns_regression_table], fmt='%s')
