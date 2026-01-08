import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from scipy.stats import entropy, nbinom
from torch.distributions import Gamma, Poisson

from src.latex_utils import convert_states
from src.utils import generate_readme

A4_DIMS = (12, 8)
rcParams["figure.figsize"] = A4_DIMS

ssla_state = []
# ssla_joint_state = []
assla_state = []


def fisher(obs, lam):
    return torch.sum(obs) / torch.pow(lam, 2)


def ell(obs: torch.Tensor, theta):
    return torch.sum(torch.log(theta) * obs - theta) - torch.sum(torch.lgamma(obs + 1))


def theta(obs: torch.Tensor):
    return torch.sum(obs) / obs.numel()


def log_ppd(
    obs: torch.Tensor,
    new_obs: torch.Tensor,
    prior: torch.distributions.Distribution,
):
    augmented_obs = torch.cat((obs, new_obs))
    hat_theta = theta(obs)
    tilde_theta = theta(augmented_obs)

    # ell
    hat_ell = ell(obs, hat_theta)
    ell_D = ell(obs, tilde_theta)
    ell_np1 = ell(new_obs, tilde_theta)
    tilde_ell = ell_D + ell_np1

    # prior
    hat_prior = prior.log_prob(hat_theta)
    tilde_prior = prior.log_prob(tilde_theta)

    # fisher
    hat_fisher = 1 / 2 * torch.log(fisher(obs, hat_theta))
    fisher_D = fisher(obs, tilde_theta)
    fisher_np1 = fisher(new_obs, tilde_theta)
    tilde_fisher = 1 / 2 * torch.log(fisher_D + fisher_np1)

    delta_ell = tilde_ell - hat_ell
    delta_prior = tilde_prior - hat_prior
    delta_fisher = hat_fisher - tilde_fisher

    log_posteriorpredicitive = delta_ell + delta_prior + delta_fisher

    global ssla_state
    ssla_state.append(
        {
            r"$\tilde{\theta}$": tilde_theta.item(),
            r"$\hat{\theta}$": hat_theta.item(),
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

    return log_posteriorpredicitive


def log_ppd_assla(
    obs: torch.Tensor,
    new_obs: torch.Tensor,
):
    augmented_obs = torch.cat((obs, new_obs))
    hat_theta = theta(obs)

    # ell
    hat_ell = ell(obs, hat_theta)
    tilde_ell = ell(augmented_obs, hat_theta)

    # fisher
    hat_fisher = 1 / 2 * torch.log(fisher(obs, hat_theta))
    tilde_fisher = 1 / 2 * torch.log(fisher(augmented_obs, hat_theta))

    delta_ell = tilde_ell - hat_ell
    delta_fisher = hat_fisher - tilde_fisher
    log_posteriorpredicitive = delta_ell + delta_fisher

    global assla_state
    assla_state.append(
        {
            r"$\hat{\theta}$": hat_theta.item(),
            r"$\tilde{\ell}(\cdot)$": tilde_ell.item(),
            r"$\hat{\ell}(\cdot)$": hat_ell.item(),
            r"$\tilde{\mathcal{J}}(\cdot)$": tilde_fisher.numpy().tolist(),
            r"$\hat{\mathcal{J}}(\cdot)$": hat_fisher.numpy().tolist(),
            r"$\Delta \ell(\cdot)$": delta_ell.item(),
            r"$\Delta \mathcal{J}(\cdot)$": delta_fisher.item(),
            r"$\log(p(\hat{y}_{n+1}|x_{n+1}, D))$": log_posteriorpredicitive.item(),
            r"$p(\hat{y}_{n+1}|x_{n+1}, D)$": log_posteriorpredicitive.exp().item(),
        }
    )

    return log_posteriorpredicitive


def perform_experiment(
    dgp: torch.distributions.Distribution,
    prior: torch.distributions.Distribution,
    alpha_0: torch.Tensor,
    beta_0: torch.Tensor,
    source: Path,
    graphics: Path,
):
    lin = torch.arange(0, 11, 1)
    fig, axes = plt.subplots(2, 3)
    j = 0
    for i, n in enumerate([2e1, 1e2, 1e3, 1e4, 1e5, 1e6]):
        if i == 3:
            j = 1
        observations = dgp.sample((int(n),))

        # Analytic posterior predictive distribution
        alpha_n = torch.sum(observations) + alpha_0
        beta_n = int(n) + beta_0
        posterior_dist = Gamma(concentration=alpha_n, rate=beta_n)

        # Analytic posterior predictive
        r = torch.sum(observations) + alpha_0
        p = (int(n) + beta_0) / (int(n) + beta_0 + 1)
        # ppd_dist = NegativeBinomial(total_count=r, probs=p) # Not usable: https://github.com/pytorch/pytorch/issues/62178
        ppd_dist = nbinom(r, p)

        prob = []
        ll_probs = []
        # ll_joint_probs = []
        assla_probs = []
        for v in lin:
            prob.append(ppd_dist.pmf(v))
            ll_probs.append(log_ppd(observations, v.reshape(1), prior).exp())
            # ll_joint_probs.append(
            #     log_joint_ppd(observations, v.reshape(1), prior).exp()
            # )
            assla_probs.append(log_ppd_assla(observations, v.reshape(1)).exp())
        # entropy
        ssla_entropy = entropy(ll_probs, prob)
        # ssla_joint_entropy = entropy(ll_joint_probs, prob)
        assla_entropy = entropy(assla_probs, prob)

        axes[j, i % 3].plot(
            assla_probs,
            marker="x",
            label=f"ASSLA (Entropy={assla_entropy:.2f})",
            c="black",
            markersize=4,
        )
        axes[j, i % 3].plot(
            ll_probs,
            marker="o",
            label=f"SSLA (Entropy={ssla_entropy:.2f})",
            c="red",
            markersize=4,
        )
        # axes[j, i % 3].plot(
        #     ll_joint_probs,
        #     marker="o",
        #     label=f"SSLA (Entropy={ssla_joint_entropy:.2f})",
        #     c="orange",
        #     markersize=4,
        # )
        axes[j, i % 3].plot(prob, linestyle="--", label="Analytic", c="green")
        axes[j, i % 3].set_title(f"n = {int(n)}", fontsize=10)
        axes[j, i % 3].legend(loc="upper right", fontsize=8)
        axes[j, i % 3].grid(True, linestyle="--", alpha=0.6)
        axes[j, i % 3].set_ylim(0, 0.35)

        # state
        global ssla_state
        # global ssla_joint_state
        global assla_state
        with open(
            source / f"ssla_poi_gamma_n{n}.json",
            "w+",
        ) as f:
            json.dump(ssla_state, f)
        ssla_state = []

        # with open(
        #     source / f"ssla_joint_poi_gamma_n{n}.json",
        #     "w+",
        # ) as f:
        #     json.dump(ssla_joint_state, f)
        # ssla_joint_state = []

        with open(
            source / f"assla_poi_gamma_n{n}.json",
            "w+",
        ) as f:
            json.dump(assla_state, f)
        assla_state = []

    fig.text(0.5, 0.01, r"$x_{new}$", ha="center", fontsize=12)  # X-axis label
    fig.text(
        0.01,
        0.5,
        "Posterior Predictive Density (PPD)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )  # Y-axis label

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    fig.subplots_adjust(bottom=0.06, left=0.06)

    fig.suptitle(
        "Comparison of SSLA and Analytic Posterior Predictive Distributions",
        fontsize=14,
    )
    fig.savefig(graphics / "Poisson_Gamma_Model.png", dpi=300)
    # plt.show()


def run_experiment(
    title: str,
    seed: int,
    dgp: torch.distributions.Distribution,
    prior: torch.distributions.Distribution,
    alpha_0: torch.Tensor,
    beta_0: torch.Tensor,
):
    EXPERIMENT_PATH = Path(f"./experiments/conjugate_prior/poi_gamma/{title}")
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
        dgp=dgp,
        prior=prior,
        alpha_0=alpha_0,
        beta_0=beta_0,
        source=JSON_PATH,
        graphics=GRAPHICS_PATH,
    )

    # 2. Convert states to table
    convert_states(source=JSON_PATH, target=LATEX_PATH)

    # 3. Check change from joint to conditional SSLA

    # 4. Generate ReadME
    generate_readme(
        title=title,
        dgp=f"lambda: {dgp.mean}",
        prior=f"alpha_0: {alpha_0.item()}\n\tbeta_0: {beta_0.item()}",
        experiment_path=EXPERIMENT_PATH,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_title", "-title", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    dgp_parser = parser.add_argument_group("Data-Generating Process")
    dgp_parser.add_argument("--lam", "-l", default=3.0, type=float)
    prior_parser = parser.add_argument_group("Prior Configuration")
    prior_parser.add_argument("--alpha_0", "-a0", default=6.0, type=float)
    prior_parser.add_argument("--beta_0", "-b0", default=2.0, type=float)
    args = parser.parse_args()

    title = args.experiment_title
    seed = args.seed

    # dgp
    data_dist = Poisson(rate=args.lam)

    # prior
    # E(X) = \alpha / \beta , X \sim Gamma(\alpha, \beta)
    alpha_0 = torch.tensor(args.alpha_0)
    beta_0 = torch.tensor(args.beta_0)
    prior_dist = Gamma(concentration=alpha_0, rate=beta_0)

    run_experiment(
        title=title,
        seed=seed,
        dgp=data_dist,
        prior=prior_dist,
        alpha_0=alpha_0,
        beta_0=beta_0,
    )
