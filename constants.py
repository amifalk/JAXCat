import jax.numpy as jnp
import utils

# --- specify data and size to fit ---
N = 100
N_VIEWS = 5
N_CATS = 10
X = utils.gen_views([(2, 5), (3, 5)], N)
X = jnp.asarray(X)
# ---

#---- don't touch these, useful constants    
N, D_G = X.shape
X_G_PRESENT = 1 - jnp.isnan(X)
X_G = jnp.nan_to_num(X)
X_G_SQ = X_G**2
#----

# --- normal-inverse-chi-squared prior ---
var_f = jnp.var(X_G, axis=0)
E_X_f = jnp.mean(X_G, axis=0)
log_n = jnp.log(N)

# prior mean M ~ Normal(M_M, 1/M_SS), unknown mean, known precision
M_M = E_X_f       
M_SS = var_f
# prior belief in prior mean K ~ Gamma(K_A, K_B), known shape A, unknown rate B
K_A = 1.0
K_B = 1.0
# prior variance SS ~ invGamma(SS_A, SS_B), known shape A, unknown rate B
SS_A = log_n
SS_B = log_n
# prior belief in prior variance V ~ invGamma(V_A, V_B) known shape A, unknown rate B
V_A = 1.0
V_B = var_f
# ----

# gamma hyperprior on view concentration
ALPHA_V = 1.0
BETA_V = 1.0

# --- view ---
GAM_V_1 = jnp.ones(N_VIEWS)
GAM_V_2 = jnp.ones(N_VIEWS)    
# ---

# gamma hyperprior on cat concentration
ALPHA_V_K = 1.0
BETA_V_K = 1.0 

# --- cluster ---
GAM_V_K_1 = jnp.ones((N_VIEWS, N_CATS))
GAM_V_K_2 = jnp.ones((N_VIEWS, N_CATS))
# ---

# --- set priors as posteriors from the hyperpriors ---
E_N_f = jnp.sum(X_G_PRESENT, axis=0)
N_E_X_f = jnp.sum(X_G, axis=0)

TAU_f = 1/jnp.var(X_G, ddof=1, axis=0) # empirical precision
ALPHA_f = E_X_f/K_A # empirical shape of gamma dist, TODO this is a slightly biased estimate but it will do for now -> https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimators        

post_K_A = K_A + E_N_f*ALPHA_f
post_K_B = K_B + N_E_X_f

M = (((1/M_SS)*M_M + TAU_f*N_E_X_f)/((1/M_SS) + E_N_f*TAU_f))[:, None, None] # posterior M 
K = (post_K_A/post_K_B)[:, None, None]
SS = jnp.ones(D_G)[:, None, None]
V = jnp.array([1e-4])[:, None, None]