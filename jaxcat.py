import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
import numpyro.distributions as dist

from constants import *

from typing import NamedTuple
import numpy as np

class Params(NamedTuple):
    # --- view params ----
    # gamma hyperprior on alpha
    alpha_v: jnp.float32
    beta_v: jnp.float32
    # beta stickbreaking params
    gam_v_1: jnp.ndarray
    gam_v_2: jnp.ndarray
    # feature-view assignments
    pi: jnp.ndarray
    # ----

    # --- cat params ---
    # gamma hyperprior on alpha
    alpha_v_k: jnp.ndarray
    beta_v_k: jnp.ndarray
    # beta stickbreaking params
    gam_v_k_1: jnp.ndarray
    gam_v_k_2: jnp.ndarray
    # row-cluster assignments
    eta: jnp.ndarray
    # ---

    # normal prior
    m: jnp.ndarray
    k: jnp.ndarray
    sigma_sq: jnp.ndarray
    v: jnp.ndarray


def view_stickbreak(arr):
    cumprod_one_minus_arr = jnp.cumprod(1 - arr)
    arr_one = jnp.pad(arr, (0, 1), constant_values=1) # add a one to the end
    one_c = jnp.pad(cumprod_one_minus_arr, (1, 0), constant_values=1) # add a one to the beginning
    
    return arr_one * one_c

cat_stickbreak = jax.vmap(view_stickbreak, 0)

def update_normal(view_assignments, cat_assignments):
    """https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf"""
        
    # calc sufficient statistics
    E_N = jnp.tensordot(X_G_PRESENT, cat_assignments, (0, 1)) * view_assignments[..., None]
    N_E_X = jnp.tensordot(X_G, cat_assignments, (0, 1)) * view_assignments[..., None]
    E_X_sq = jnp.tensordot(X_G_SQ, cat_assignments, (0, 1)) * view_assignments[..., None]
    
    post_k = K + E_N
    post_m = (M*K + N_E_X) / post_k
    post_v = V + E_N
    post_sigma_sq = (V*SS + K*(M**2) + E_X_sq - post_k*(post_m**2)) / post_v

    # t-distribution log likelihood
    _mean_diff = X_G[..., None, None] - post_m[None, ...]
    log_prob = -0.5 * (
        _mean_diff**2 / post_sigma_sq
        + 1 / post_k
        + jnp.log(post_sigma_sq)
        + jnp.log(post_v / 2)
        - digamma(post_v / 2)
        + jnp.log(2 * jnp.pi)
    )

    return post_m, post_k, post_sigma_sq, post_v, log_prob


def update_view_sticks(prior_alpha, prior_beta, gam_1, gam_2, view_assignments, n_views):
    """
    update hyperprior, prior on view stickbreaking 
    
    Args:
        prior_alpha (jnp.int32): prior shape of Gamma dist 
        prior_beta (jnp.int32): prior rate of Gamma dist
        gam_1 (jnp.ndarray): current estimate (V,) 
        gam_2 (jnp.ndarray): current estimate (V,)
        assignments (jnp.ndarray): (D, V) 
        V (jnp.int32): scalar
    
    Returns:
        posterior_alpha, posterior_beta, posterior_gam_1, posterior_gam_2
    """
    
    post_alpha = prior_alpha + n_views - 1
    post_beta = prior_beta - jnp.sum((digamma(gam_2) - digamma(gam_1 + gam_2))[:n_views-1])

    # E_q(Gamma(post_alpha, post_beta))
    view_concentration = post_alpha / post_beta
    
    # view_concentration = 1. # uncomment to remove hyperprior on views/clusters

    post_gam_1 = 1 + jnp.sum(view_assignments, axis=0)
    select_views = jnp.tril(jnp.ones((n_views, n_views)), k=-1)
    post_gam_2 = view_concentration + jnp.sum(view_assignments @ select_views, axis=0)

    return post_alpha, post_beta, post_gam_1, post_gam_2

update_cat_sticks = jax.vmap(update_view_sticks, (None, None, 0, 0, 0, None), 0)

def update_view_assignment(gam_1, gam_2, log_prob):
    """   
    https://web.archive.org/web/20160811014115/http://scikit-learn.org/stable/modules/dp-derivation.html
    
    assign objs to clusters under the DP, with stickbreaking process $\prod{Beta(gam_1_i, gam_2_i)}$

    Args:
        gam_1 (jnp.ndarray): (V,) 
        gam_2 (jnp.ndarray): (V,) 
        log_prob (jnp.ndarray): (D, V) 
        V (jnp.int32): scalar

    Returns:
        posterior_assignments (jnp.ndarray): assignment of features to views 
    """
    digamma_sum = digamma(gam_1 + gam_2)
    priors = digamma(gam_1) - digamma_sum + jnp.sum((digamma(gam_2) - digamma_sum)[:-1])    
    
    posterior_assignments = priors + log_prob

    return jax.nn.softmax(posterior_assignments, axis=-1) 

update_cat_assignment = jax.vmap(update_view_assignment, (0, 0, 1), 0)     
    
def init(rng_key):
    pi_key, eta_key = jax.random.split(rng_key, 2)

    pi = dist.Dirichlet(jnp.ones(N_VIEWS)).sample(pi_key, (D_G,))
    eta = dist.Dirichlet(jnp.ones(N_CATS)).sample(eta_key, (N_VIEWS, N))

    params = Params(ALPHA_V, 
                    BETA_V,
                    GAM_V_1,
                    GAM_V_2,
                    pi,
                    ALPHA_V_K,
                    BETA_V_K,
                    GAM_V_K_1,
                    GAM_V_K_2,
                    eta,
                    M,
                    K,
                    SS,
                    V)

    return params


@jax.jit
def do_iter(params):
    post_m, post_k, post_sigma_sq, post_v, log_prob = update_normal(params.pi,
                                                                    params.eta)
        
    post_alpha_v, post_beta_v, post_gam_v_1, post_gam_v_2 = update_view_sticks(ALPHA_V,
                                                                               BETA_V,
                                                                               params.gam_v_1,
                                                                               params.gam_v_2,
                                                                               params.pi,
                                                                               N_VIEWS)
    
    post_alpha_v_k, post_beta_v_k, post_gam_v_k_1, post_gam_v_k_2 = update_cat_sticks(ALPHA_V_K,
                                                                                      BETA_V_K,
                                                                                      params.gam_v_k_1,
                                                                                      params.gam_v_k_2,
                                                                                      params.eta,
                                                                                      N_CATS)
        
    post_pi = update_view_assignment(post_gam_v_1,
                                     post_gam_v_2,
                                     jnp.sum(log_prob, axis=(0, 3)))
    
    post_eta = update_cat_assignment(params.gam_v_k_1,
                                     params.gam_v_k_2,
                                     jnp.sum(log_prob, axis=1))
    
    post_m, post_k, post_sigma_sq, post_v, log_prob = update_normal(params.pi,
                                                                      params.eta)
    
    post_pi = post_pi[:, jnp.flip(jnp.argsort(jnp.sum(post_pi, axis=0)))]
    
    post_params = Params(post_alpha_v,
                         post_beta_v,  
                         post_gam_v_1, 
                         post_gam_v_2, 
                         post_pi,       
                         post_alpha_v_k,
                         post_beta_v_k,
                         post_gam_v_k_1,
                         post_gam_v_k_2,
                         post_eta,
                         post_m,
                         post_k,
                         post_sigma_sq,
                         post_v)

    # (obs, feats, views, clusters)
    prob = jnp.exp(log_prob)
    view_weighted_log_prob = jnp.log(post_pi[None, :, :, None]*prob)
    view_weighted_log_prob = jnp.nan_to_num(view_weighted_log_prob, neginf=0.)

    joint_lik = jnp.sum(view_weighted_log_prob, axis=1)
    
    alt_post_eta = jnp.swapaxes(post_eta, 0, 1)
    weighted_joint = jnp.exp(joint_lik)*alt_post_eta

    lik = jnp.sum(jnp.log(jnp.sum(weighted_joint, axis=2)))    

    return lik, post_params

def fit(params, n_iter=20):
    liks = np.zeros(n_iter)
    params_over_time = []
    
    for i in range(n_iter):
        lik, params = do_iter(params)
        liks[i] = jnp.sum(lik)
        params_over_time.append(params)
                
    return liks, params, params_over_time
    
if __name__ == '__main__':
    from tqdm import tqdm    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    n_init = 20
    n_iter = 100
    keys = jax.random.split(jax.random.PRNGKey(0), n_init)
    
    losses = np.zeros((n_init, n_iter))
    all_params = []
    params_over_time = []
    
    for i in tqdm(range(n_init)):
        params = init(keys[i])
        loss, params, param_over_time = fit(params, n_iter)
        losses[i] = loss
        all_params.append(params)
        params_over_time.append(param_over_time)

    worst_ind = np.argmin(losses[:, -1])
    best_ind = np.argmax(losses[:, -1])
    best_params = all_params[best_ind]    


    sns.heatmap(best_params.pi)
    plt.title('Feature View Assignments')
    plt.xlabel('Features')
    plt.ylabel('Views');
    
    top_n_views = min(4, N_VIEWS)

    fig, axs = plt.subplots(
        nrows=1, ncols=top_n_views, figsize=(20, 5)
    )

    for view_i, ax in enumerate(axs.flatten()):
        sns.heatmap(best_params.eta[view_i], ax=ax, vmin=0, vmax=1)

        ax.set_xlabel("cluster i")
        ax.set_ylabel("row i")
        ax.set_title(f"View {view_i}")

    fig.tight_layout()
    plt.show()
    
    start = 0

    plt.plot(losses.T[start:, :], alpha=.1)
    plt.plot(losses.T[start:,best_ind])
    plt.title('logp(data | params)')
    plt.show()