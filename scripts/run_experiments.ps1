poetry shell

# Normal Normal experiments
poetry run python .\experiments\conjugate_prior\normal_normal\normal_normal.py -title Default
poetry run python .\experiments\conjugate_prior\normal_normal\normal_normal.py -title PriorDataConflict --mu 5.0 --sigma_squared 1.0 --tau_0 1.0 --mu_0 1.0
poetry run python .\experiments\conjugate_prior\normal_normal\normal_normal.py -title PriorDataConflict_UnsureData --mu 5.0 --sigma_squared 4.0 --tau_0 1.0 --mu_0 1.0
poetry run python .\experiments\conjugate_prior\normal_normal\normal_normal.py -title PriorDataConflict_UnsurePrior --mu 5.0 --sigma_squared 1.0 --tau_0 4.0 --mu_0 1.0

# Poisson Gamma experiments
poetry run python .\experiments\conjugate_prior\poi_gamma\poisson_model.py -title Default
poetry run python .\experiments\conjugate_prior\poi_gamma\poisson_model.py -title PriorDataConflict --lam 3.0 --alpha_0 10. --beta_0 10.
poetry run python .\experiments\conjugate_prior\poi_gamma\poisson_model.py -title PriorDataConflict_UnsurePrior --lam 3.0 --alpha_0 .1 --beta_0 .1

# Bayesian Linear Regression experiments
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title Default 
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title DefaultSeed43 --seed 43 
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title DefaultSeed44 --seed 44

poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title N1000 --n  1000
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title N1000_Seed43 --n  1000 --seed 43
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title N1000_Seed44 --n  1000 --seed 44

## Varying sigma
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title VaryingSigma1 -s 1
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title VaryingSigma10 -s 10
poetry run python .\experiments\conjugate_prior\bayesian_linear_regression\bayesian_linear_regression.py -title VaryingSigma1000 -s 1000

# Heteroscedastic Regression experiment
poetry run python .\experiments\heteroscedastic_regression\heteroscedastic_regression.py

# UCI ML Repository experiments
poetry run python .\experiments\uciml\uci_experiment.py