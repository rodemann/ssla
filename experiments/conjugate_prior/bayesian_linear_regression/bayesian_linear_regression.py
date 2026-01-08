import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from scipy.stats import entropy, gaussian_kde
from torch.distributions import Normal, Uniform
from torch.distributions.multivariate_normal import MultivariateNormal

from src.latex_utils import convert_states
from src.utils import generate_readme

A4_DIMS = (12, 8)
rcParams["figure.figsize"] = A4_DIMS

state = []


def generate_data(
    n: torch.Tensor = torch.tensor(100),
    n_test: torch.Tensor = torch.tensor(50),
    theta_true: torch.Tensor = torch.tensor([3.0, 5.0]),
    sigma_true: torch.Tensor = torch.tensor(10.0),
):
    # train
    X = Uniform(low=-5, high=5).sample((n,))
    X_design = torch.cat((torch.ones(n, 1), X.reshape(-1, 1)), dim=1)
    noise = Normal(0, sigma_true).sample((n,))
    y = torch.matmul(X_design, theta_true) + noise

    # test
    X_design_test = torch.cat(
        (torch.ones(n_test, 1), torch.linspace(-5, 5, steps=n_test).reshape(-1, 1)),
        dim=1,
    )
    y_test = torch.matmul(X_design_test, theta_true)

    return X_design, y, X_design_test, y_test


def prior_fac(
    theta_0: torch.Tensor = torch.tensor([3.0, 5.0]),
    Sigma_0: torch.Tensor = torch.eye(2),
):
    prior_dist = MultivariateNormal(theta_0, Sigma_0)
    return prior_dist


def posterior_fac(
    X_design: torch.Tensor,
    y: torch.Tensor,
    X_new: torch.Tensor,
    theta_0: torch.Tensor = torch.tensor([3.0, 5.0]),
    Sigma_0: torch.Tensor = torch.eye(2),
    sigma_true: torch.Tensor = torch.tensor(10.0),
):
    # posterior
    Sigma_n = torch.inverse(
        torch.inverse(Sigma_0) + X_design.T @ X_design / sigma_true**2
    )
    theta_n = Sigma_n @ (
        torch.inverse(Sigma_0) @ theta_0 + 1 / sigma_true**2 * X_design.T @ y
    )
    posterior_dist = MultivariateNormal(theta_n, Sigma_n)

    # posterior predictive
    predictive_mean = X_new @ theta_n
    predictive_var = sigma_true**2 + X_new @ Sigma_n @ X_new.T
    predictive_dist = Normal(predictive_mean.flatten(), predictive_var.sqrt().flatten())
    return posterior_dist, predictive_dist


def ell(X, y, theta, sigma):
    return -X.shape[0] / 2 * torch.log(2 * torch.pi * sigma**2) - 1 / (
        2 * sigma**2
    ) * (y - X @ theta).T @ (y - X @ theta)


def theta(X, y):
    return torch.inverse(X.T @ X) @ X.T @ y


def fisher(X, sigma):
    return 1 / sigma**2 * X.T @ X


def log_ppd(
    y: torch.Tensor,
    y_new: torch.Tensor,
    X: torch.Tensor,
    X_new: torch.Tensor,
    sigma: torch.Tensor,
    prior: torch.distributions.Distribution,
):
    augmented_y = torch.cat((y, y_new))
    augmented_X = torch.cat((X, X_new), dim=0)

    tilde_theta = theta(augmented_X, augmented_y)
    hat_theta = theta(X, y)

    ell_D = ell(X, y, tilde_theta, sigma)
    ell_np1 = ell(X_new, y_new, tilde_theta, sigma)
    # tilde_ell = ell(augmented_X, augmented_y, tilde_theta, sigma)
    tilde_ell = ell_D + ell_np1
    hat_ell = ell(X, y, hat_theta, sigma)

    tilde_prior = prior.log_prob(tilde_theta)
    hat_prior = prior.log_prob(hat_theta)

    fisher_D = fisher(X, sigma)
    fisher_np1 = fisher(X_new, sigma)
    tilde_fisher = fisher_D + fisher_np1
    # tilde_fisher = fisher(augmented_X, sigma)
    hat_fisher = fisher(X, sigma)

    delta_ell = tilde_ell - hat_ell
    delta_prior = tilde_prior - hat_prior
    delta_fisher = 0.5 * torch.log(torch.det(hat_fisher)) - 0.5 * torch.log(
        torch.det(tilde_fisher)
    )
    log_posteriorpredicitive = delta_ell + delta_prior + delta_fisher
    global state
    state.append(
        {
            r"$\tilde{\theta}$": tilde_theta.numpy().tolist(),
            r"$\hat{\theta}$": hat_theta.numpy().tolist(),
            r"$\tilde{\ell}(\cdot)$": tilde_ell.item(),
            r"$\hat{\ell}(\cdot)$": hat_ell.item(),
            r"$\tilde{\pi}(\cdot)$": tilde_prior.item(),
            r"$\hat{\pi}(\cdot)$": hat_prior.item(),
            r"$\tilde{\mathcal{J}}(\cdot)$": tilde_fisher.numpy().tolist(),
            r"$\hat{\mathcal{J}}(\cdot)$": hat_fisher.numpy().tolist(),
            r"$\Delta \ell(\cdot)$": delta_ell.item(),
            r"$\Delta \pi(\cdot)$": delta_prior.item(),
            r"$\Delta \mathcal{J}(\cdot)$": delta_fisher.item(),
            r"$\log(p(\hat{y}_{n+1}|x_{n+1}, D))$": log_posteriorpredicitive.item(),
            r"$p(\hat{y}_{n+1}|x_{n+1}, D)$": log_posteriorpredicitive.exp().item(),
        }
    )
    return delta_ell + delta_prior + delta_fisher


def log_ppd_assla(
    y: torch.Tensor,
    y_new: torch.Tensor,
    X: torch.Tensor,
    X_new: torch.Tensor,
    sigma: torch.Tensor,
):
    augmented_y = torch.cat((y, y_new))
    augmented_X = torch.cat((X, X_new), dim=0)

    # tilde_theta = theta(augmented_X, augmented_y)
    hat_theta = theta(X, y)

    tilde_ell = ell(augmented_X, augmented_y, hat_theta, sigma)
    hat_ell = ell(X, y, hat_theta, sigma)

    # tilde_prior = prior.log_prob(tilde_theta)
    # hat_prior = prior.log_prob(hat_theta)

    tilde_fisher = fisher(augmented_X, sigma)
    hat_fisher = fisher(X, sigma)

    return (
        tilde_ell
        - hat_ell
        + 0.5 * torch.log(torch.det(hat_fisher))
        - 0.5 * torch.log(torch.det(tilde_fisher))
    )


def compute_credible_intervals_kde(X_test, y_test, y, X, sigma, prior, method):
    lower_bound, upper_bound = [], []

    for x_test in X_test:
        y_new_range = torch.linspace(
            y_test.min().item() - 3 * sigma.item(),
            y_test.max().item() + 3 * sigma.item(),
            100,
        )

        if method == "ssla":
            log_probs = torch.tensor(
                [
                    log_ppd(
                        y, torch.tensor([y_new]), X, x_test.unsqueeze(0), sigma, prior
                    ).item()
                    for y_new in y_new_range
                ]
            )
        elif method == "assla":
            log_probs = torch.tensor(
                [
                    log_ppd_assla(
                        y, torch.tensor([y_new]), X, x_test.unsqueeze(0), sigma
                    ).item()
                    for y_new in y_new_range
                ]
            )
        else:
            raise ValueError("Unsupported method. Use 'ssla' or 'assla'.")

        densities = log_probs.exp().numpy()
        densities /= densities.sum()
        kde = gaussian_kde(y_new_range.numpy(), weights=densities)

        xs = torch.linspace(y_new_range.min(), y_new_range.max(), 1000).numpy()
        pdf = kde(xs)
        cdf = pdf.cumsum() / pdf.sum()
        lower_idx = (cdf >= 0.025).nonzero()[0][0]
        upper_idx = (cdf >= 0.975).nonzero()[0][0]
        lower_bound.append(xs[lower_idx])
        upper_bound.append(xs[upper_idx])

    return torch.tensor(lower_bound), torch.tensor(upper_bound)


def perform_experiment(
    n: torch.Tensor,
    n_test: torch.Tensor,
    theta_true: torch.Tensor,
    sigma_true: torch.Tensor,
    graphics: Path,
    source: Path,
):

    X_design, y, X_design_test, y_test = generate_data(
        n=n, n_test=n_test, theta_true=theta_true, sigma_true=sigma_true
    )

    ## Confidence Interval
    hat_theta = theta(X_design, y)
    X_test_mean = torch.matmul(X_design_test, hat_theta)  # MLE Predictions
    X_test_cov = sigma_true**2 * torch.inverse(X_design.T @ X_design)  # Variance of MLE

    X_test_std = torch.sqrt(torch.diag(X_design_test @ X_test_cov @ X_design_test.T))
    upper_mle = X_test_mean + 1.96 * X_test_std
    lower_mle = X_test_mean - 1.96 * X_test_std
    fig = plt.figure()
    plt.scatter(X_design[:, 1].numpy(), y.numpy(), label="Observed Data", alpha=0.6)
    plt.plot(
        X_design_test[:, 1].numpy(), X_test_mean.numpy(), label="MLE Fit", color="blue"
    )
    plt.fill_between(
        X_design_test[:, 1].numpy(),
        lower_mle.numpy(),
        upper_mle.numpy(),
        color="blue",
        alpha=0.3,
        label="MLE 95% CI",
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"Generated Data, n = {n}", fontsize=14)
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend()
    fig.savefig(graphics / f"blm_mle_CI_n{n.item()}_seed{seed}.png", dpi=300)

    X_new = Uniform(low=-5, high=5).sample((1,))
    X_new = torch.cat((torch.tensor([1.0]), X_new)).reshape(-1, 2)

    priors = [
        (torch.zeros(2), torch.eye(2)),  # N((0,0), ((1, 0), (0, 1)))
        (torch.zeros(2), torch.eye(2) * 0.2),  # N((0,0), ((.1, 0), (0, .1)))
        (torch.zeros(2), torch.eye(2) * 10.0),  # N((0,0), ((10, 0), (0, 10)))
        (torch.tensor([3.0, 5.0]), torch.eye(2)),  # N((3,5), ((1, 0), (0, 1)))
        (torch.tensor([3.0, 5.0]), torch.eye(2) * 0.2),  # N((3,5), ((.1, 0), (0, .1)))
        (torch.tensor([3.0, 5.0]), torch.eye(2) * 10.0),  # N((3,5), ((10, 0), (0, 10)))
    ]
    fig, axes = plt.subplots(2, 3)
    j = 0
    for i, (theta_0, Sigma_0) in enumerate(priors):
        if i == 3:
            j = 1
        prior_dist = prior_fac(theta_0=theta_0, Sigma_0=Sigma_0)
        posterior_dist, predictive_dist = posterior_fac(
            X_design,
            y,
            X_new,
            theta_0=theta_0,
            Sigma_0=Sigma_0,
            sigma_true=sigma_true,
        )

        y_hat_values = torch.linspace(
            predictive_dist.mean.item() - 3 * predictive_dist.variance.sqrt().item(),
            predictive_dist.mean.item() + 3 * predictive_dist.variance.sqrt().item(),
            50,
        )
        log_ppd_values = torch.tensor(
            [
                log_ppd(
                    y, torch.tensor([y_val]), X_design, X_new, sigma_true, prior_dist
                ).item()
                for y_val in y_hat_values
            ]
        )
        log_ppd_assla_values = torch.tensor(
            [
                log_ppd_assla(
                    y, torch.tensor([y_val]), X_design, X_new, sigma_true
                ).item()
                for y_val in y_hat_values
            ]
        )

        # state
        global state
        with open(
            source
            / f"{theta_0[0].item()}_{theta_0[1].item()}_{Sigma_0[0,0].item():.2f}_n{n.item()}_seed{seed}.json",
            "w+",
        ) as f:
            json.dump(state, f)
        state = []

        posterior_pred_density = predictive_dist.log_prob(y_hat_values).exp()

        ssla_entropy = entropy(
            log_ppd_values.exp().numpy(), posterior_pred_density.numpy()
        )
        assla_entropy = entropy(
            log_ppd_assla_values.exp().numpy(), posterior_pred_density.numpy()
        )

        axes[j, i % 3].plot(
            y_hat_values,
            log_ppd_values.exp(),
            label=f"SSLA (Entropy={ssla_entropy:.4f})",
            linestyle="--",
            color="green",
            marker="o",
        )
        axes[j, i % 3].plot(
            y_hat_values,
            log_ppd_assla_values.exp(),
            label=f"ASSLA (Entropy={assla_entropy:.4f})",
            linestyle="--",
            color="black",
            marker="x",
        )
        axes[j, i % 3].plot(
            y_hat_values,
            posterior_pred_density,
            label="Posterior Predictive Density",
            color="red",
        )
        axes[j, i % 3].set_title(
            rf"$N({tuple(map(lambda x: x.item(), tuple(prior_dist.mean.flatten())))}^T, {Sigma_0[0,0].item():.2f} \cdot I_2)$",
            fontsize=10,
        )
        axes[j, i % 3].legend(loc="upper left", fontsize=8)
        axes[j, i % 3].grid(True, linestyle="--", alpha=0.6)

    fig.text(0.5, 0.01, r"Value of New Observation $\hat{y}$", ha="center", fontsize=12)
    fig.text(
        0.0,
        0.5,
        "Posterior Predictive Density (PPD)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    fig.subplots_adjust(bottom=0.06, left=0.06)

    fig.suptitle(
        rf"Comparison of SSLA, ASSLA and Analytic Posterior Predictive Distributions for a Random Point $x={X_new[:, 1].item():.4f}$",
        fontsize=14,
    )
    plt.savefig(graphics / f"blm_pointdist_n{n.item()}_seed{seed}.png", dpi=300)

    """ ## Confidence Interval
    hat_theta = theta(X_design, y)
    X_test_mean = torch.matmul(X_design_test, hat_theta)  # MLE Predictions
    X_test_cov = SIGMA_TRUE**2 * torch.inverse(X_design.T @ X_design)  # Variance of MLE

    X_test_std = torch.sqrt(torch.diag(X_design_test @ X_test_cov @ X_design_test.T))
    upper_mle = X_test_mean + 1.96 * X_test_std
    lower_mle = X_test_mean - 1.96 * X_test_std

    plt.figure()
    plt.scatter(X_design[:, 1].numpy(), y.numpy(), label="Observed Data", alpha=0.6)
    plt.plot(
        X_design_test[:, 1].numpy(), X_test_mean.numpy(), label="MLE Fit", color="red"
    )
    plt.fill_between(
        X_design_test[:, 1].numpy(),
        lower_mle.numpy(),
        upper_mle.numpy(),
        color="red",
        alpha=0.3,
        label="MLE 95% CI",
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("MLE Fit with Confidence Intervals")
    plt.legend()
    plt.savefig(GRAPHICS_PATH / f"blm_mle_CI_seed{seed}.png", dpi=300) """

    ## Credable Interval
    fig, axes = plt.subplots(2, 3)
    j = 0
    for i, (theta_0, Sigma_0) in enumerate(priors):
        if i == 3:
            j = 1
        prior_dist = prior_fac(theta_0=theta_0, Sigma_0=Sigma_0)
        posterior_dist, predictive_dist = posterior_fac(
            X_design,
            y,
            X_new,
            theta_0=torch.zeros(2),
            Sigma_0=torch.eye(2),
            sigma_true=sigma_true,
        )
        ssla_lower, ssla_upper = compute_credible_intervals_kde(
            X_design_test, y_test, y, X_design, sigma_true, prior_dist, "ssla"
        )
        assla_lower, assla_upper = compute_credible_intervals_kde(
            X_design_test, y_test, y, X_design, sigma_true, prior_dist, "assla"
        )
        # Compute analytic predictive mean and variance
        analytic_mean = X_design_test @ posterior_dist.mean  # Predictive mean
        analytic_var = (
            sigma_true**2
            + (
                X_design_test
                @ torch.inverse(posterior_dist.precision_matrix)
                @ X_design_test.T
            ).diag()
        )  # Predictive variance

        # Compute analytic credible intervals
        analytic_lower = analytic_mean - 1.96 * analytic_var.sqrt()
        analytic_upper = analytic_mean + 1.96 * analytic_var.sqrt()

        axes[j, i % 3].scatter(
            X_design[:, 1], y, label="Observed Data", color="orange", alpha=0.6
        )
        axes[j, i % 3].scatter(
            X_design_test[:, 1], y_test, label="True Test Data", color="blue", alpha=0.6
        )
        axes[j, i % 3].plot(
            X_design_test[:, 1],
            X_test_mean,
            label="Posterior Predictive Mean",
            color="red",
        )

        # Add Analytic CI
        axes[j, i % 3].fill_between(
            X_design_test[:, 1],
            analytic_lower.numpy(),
            analytic_upper.numpy(),
            color="blue",
            alpha=0.3,
            label="Analytic Predictive 95% CI",
        )

        # SSLA credible intervals
        axes[j, i % 3].fill_between(
            X_design_test[:, 1],
            ssla_lower,
            ssla_upper,
            color="green",
            alpha=0.3,
            label="SSLA 95% CI",
        )

        # ASSLA credible intervals
        axes[j, i % 3].fill_between(
            X_design_test[:, 1],
            assla_lower,
            assla_upper,
            color="black",
            alpha=0.3,
            label="ASSLA 95% CI",
        )
        axes[j, i % 3].set_title(
            rf"$N({tuple(map(lambda x: x.item(), tuple(prior_dist.mean.flatten())))}^T, {Sigma_0[0,0].item():.2f} \cdot I_2)$",
            fontsize=10,
        )
        axes[j, i % 3].legend(loc="upper left", fontsize=8)
        axes[j, i % 3].grid(True, linestyle="--", alpha=0.6)
    fig.text(0.5, 0.01, r"x", ha="center", fontsize=12)
    fig.text(
        0.01,
        0.5,
        "y",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    fig.subplots_adjust(bottom=0.06, left=0.06)

    fig.suptitle(
        f"Comparison of SSLA, ASSLA and Analytic 95% Credible Interval",
        fontsize=14,
    )
    plt.savefig(graphics / f"blm_CI_n{n.item()}_seed{seed}.png", dpi=300)


def run_experiment(
    title: str,
    seed: int,
    n: torch.Tensor,
    n_test: torch.Tensor,
    theta_true: torch.Tensor,
    sigma_true: torch.Tensor,
):
    EXPERIMENT_PATH = Path(
        f"./experiments/conjugate_prior/bayesian_linear_regression/{title}"
    )
    JSON_PATH = EXPERIMENT_PATH / "states"
    LATEX_PATH = EXPERIMENT_PATH / "latex"
    GRAPHICS_PATH = EXPERIMENT_PATH / "graphics"
    if EXPERIMENT_PATH.exists():
        warnings.warn(f"Experiment with name {title} already exists. Overwriting")
    EXPERIMENT_PATH.mkdir(exist_ok=True)
    JSON_PATH.mkdir(exist_ok=True)
    LATEX_PATH.mkdir(exist_ok=True)

    GRAPHICS_PATH.mkdir(exist_ok=True)

    # 0. Set Seed
    torch.manual_seed(seed)

    # 1. Perform standard experiment
    perform_experiment(
        n=n,
        n_test=n_test,
        theta_true=theta_true,
        sigma_true=sigma_true,
        source=JSON_PATH,
        graphics=GRAPHICS_PATH,
    )

    # 2. Convert states to table
    convert_states(source=JSON_PATH, target=LATEX_PATH)

    # 3. Check change from joint to conditional SSLA

    # 4. Generate ReadME
    generate_readme(
        title=title,
        dgp=f"theta: {theta_true.tolist()}\n\tsigma_sq: {torch.pow(sigma_true, 2).item()}",
        prior="Auto Generated",
        experiment_path=EXPERIMENT_PATH,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_title", "-title", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    dgp_parser = parser.add_argument_group("Data-Generating Process")
    dgp_parser.add_argument("--n", default=100, type=int)
    dgp_parser.add_argument("--n_test", default=50, type=int)
    dgp_parser.add_argument("--theta", "-t", nargs=2, default=[3.0, 5.0], type=float)
    dgp_parser.add_argument("--sigma_squared", "-s", default=100.0, type=float)
    args = parser.parse_args()

    title = args.experiment_title
    seed = args.seed
    # dgp
    n = torch.tensor(args.n)
    n_test = torch.tensor(args.n_test)
    theta_true = torch.tensor(args.theta)
    sigma_true = torch.sqrt(torch.tensor(args.sigma_squared))

    run_experiment(
        title=title,
        seed=seed,
        n=n,
        n_test=n_test,
        theta_true=theta_true,
        sigma_true=sigma_true,
    )
