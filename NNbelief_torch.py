import torch
from itertools import chain, combinations



def beta_0(b):  # Equ.38
    """[summary]

    Args:
        b ([type]): shape (K,)

    Returns:
        [type]: shape (K,)
    """
    return b - b.mean()


def beta_jk(w):  # appendix B.3
    """[summary]

    Args:
        w ([type]): shape (J, K)

    Returns:
        beta_jk_ast [type]: shape (J, K)
    """
    beta_mean = torch.mean(w, dim=1, keepdims=True)  # shape (J, 1)
    beta_jk_ast = w - beta_mean  # shape (J, K)
    return beta_jk_ast


def alpha_jk(beta, beta_0, phix):  # Equ.38
    """[summary]

    Args:
        beta ([type]): shape (J, K)
        beta_0 ([type]): shape (K,)
        phix ([type]): shape (N, J)

    Returns:
        [type]: [description] shape (J, K)
    """
    J, K = beta.shape
    mu = torch.mean(phix, dim=0, keepdims=True)  # shape (1, J)
    first_term = (beta_0.reshape(1, K) + torch.mm(mu, beta)) / J  # shape (1, K)
    second_term = beta * mu.reshape(J, 1)  # shape (J, K)
    alpha_jk_ast = first_term - second_term  # shape (J, K)
    return alpha_jk_ast


def belief_weight(beta, alpha, phix):  # next equ of Equ.35
    """[summary]

    Args:
        beta ([type]): shape (J, K)
        alpha ([type]): shape (J, K)
        phix ([type]): shape (N, J)

    Returns:
        [type]: shape (N, J, K)
    """
    J, K = beta.shape
    weight_jk = beta.reshape(1, J, K) * phix.reshape(-1, J, 1) + alpha.reshape(
        1, J, K)  # shape (N, J, K)
    return weight_jk


def relu(x):
    return x * (x > 0)


def weight_pos(beta, alpha, x):  # Equ.27
    """[summary]

    Args:
        beta ([type]): shape (J, K)
        alpha ([type]): shape (J, K)
        x ([type]): shape (N, J)

    Returns:
        [type]: shape (N, K)
    """
    weight = belief_weight(beta, alpha, x)  # shape (N, J, K)
    pos_weight = relu(weight).sum(dim=1)  # shape (N, K)
    return pos_weight


def weight_neg(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): shape (J, K)
        alpha ([type]): shape (J, K)
        x ([type]): shape (N, J)

    Returns:
        [type]: shape (N, K)
    """
    weight = belief_weight(beta, alpha, x)  # shape (N, J, K)
    neg_weight = relu(-weight).sum(dim=1)  # shape (N, K)
    return neg_weight


def eta_pos(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    w_pos = weight_pos(beta, alpha, x)  # shape (N, K)
    K = w_pos.shape[-1]
    eta_pos = 1 / (torch.exp(w_pos).sum(dim=1) - K + 1)  # shape (N,)
    return eta_pos


def eta_neg(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    w_neg = weight_neg(beta, alpha, x)  # shape (N, K)
    eta_neg = 1 / (1 - torch.prod(1 - torch.exp(-w_neg), dim=1))  # shape (N,)
    return eta_neg


def conflict(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    w_pos = weight_pos(beta, alpha, x)  # shape (N, K)
    w_neg = weight_neg(beta, alpha, x)  # shape (N, K)
    eta_pos_temp = eta_pos(beta, alpha, x)  # shape (N,)
    eta_neg_temp = eta_neg(beta, alpha, x)  # shape (N,)
    kappa = torch.sum(eta_pos_temp.reshape(-1, 1) * (torch.exp(w_pos) - 1) *
                   (1 - eta_neg_temp.reshape(-1, 1) * torch.exp(-w_neg)),
                   dim=1)  # shape (N,)
    return kappa


def eta(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    return 1 / (1 - conflict(beta, alpha, x))


def ignorance(beta, alpha, x):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    w_neg = weight_neg(beta, alpha, x)  # shape (N, K)
    eta_pos_temp = eta_pos(beta, alpha, x)  # shape (N,)
    eta_neg_temp = eta_neg(beta, alpha, x)  # shape (N,)
    eta_temp = eta(beta, alpha, x)  # shape (N,)
    ig_temp = eta_temp * eta_pos_temp * eta_neg_temp * torch.exp(
        -torch.sum(w_neg, dim=1))
    return ig_temp


def m_theta_k(beta, alpha, x):  # return the masses for all k
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: shape (N, K)
    """
    w_pos = weight_pos(beta, alpha, x)  # shape (N, K)
    w_neg = weight_neg(beta, alpha, x)  # shape (N, K)
    eta_pos_temp = eta_pos(beta, alpha, x)  # shape (N,)
    eta_neg_temp = eta_neg(beta, alpha, x)  # shape (N,)
    eta_temp = eta(beta, alpha, x)  # shape (N,)

    eta_mul = eta_temp * eta_pos_temp * eta_neg_temp  # shape (N,)
    first_term = eta_mul.reshape(-1, 1) * torch.exp(-w_neg)  # shape (N, K)
    prod_term = torch.prod(1 - torch.exp(-w_neg), dim=1, keepdims=True) / (
        1 - torch.exp(-w_neg))  # shape (N, 1) / shape (N, K) = shape (N, K)
    second_term = torch.exp(w_pos) - 1 + prod_term  # shape (N, K)
    m_theta = first_term * second_term  # shape (N, K)

    return m_theta


def my_floor(a, precision=0):
    """[summary]
        理论上1-sum(m(\theta_k))应该≤1，但计算中会有浮点误差问题，会出现0或负值
    Args:
        a ([type]): [description]
        precision (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """    
    return torch.round(a - 0.5 * 10**(-precision), precision)

def get_k_ig(beta, alpha, x):
    weight = belief_weight(beta, alpha, x)  # shape (N, J, K)
    
    # weight = weight / 0.4
    w_pos = relu(weight).sum(dim=1)  # shape (N, K)
    w_neg = relu(-weight).sum(dim=1)  # shape (N, K)
    K = w_pos.shape[-1]

    eta_pos_temp = 1 / (torch.exp(w_pos).sum(dim=1) - K + 1)  # shape (N,)
    eta_neg_temp = 1 / (1 - torch.prod(1 - torch.exp(-w_neg), dim=1))  # shape (N,)
    kappa = torch.sum(eta_pos_temp.reshape(-1, 1) * (torch.exp(w_pos) - 1) *
                   (1 - eta_neg_temp.reshape(-1, 1) * torch.exp(-w_neg)),
                   dim=1)  # shape (N,)
    
    eta_temp = 1 / (1 - kappa)  # shape (N,)
    ig_temp = eta_temp * eta_pos_temp * eta_neg_temp * torch.exp(
        -torch.sum(w_neg, dim=1))
    
    # eta_mul = eta_temp * eta_pos_temp * eta_neg_temp  # shape (N,)
    # first_term = eta_mul.reshape(-1, 1) * torch.exp(-w_neg)  # shape (N, K)
    # prod_term = torch.prod(1 - torch.exp(-w_neg), dim=1, keepdims=True) / (
    #     1 - torch.exp(-w_neg))  # shape (N, 1) / shape (N, K) = shape (N, K)
    # second_term = torch.exp(w_pos) - 1 + prod_term  # shape (N, K)
    # m_theta_k = first_term * second_term  # shape (N, K)
    # non_specificity_simple = 1 - my_floor(m_theta_k, precision=15).sum(dim=1)
    return kappa, ig_temp

def get_k_ig_ns(beta, alpha, x):
    weight = belief_weight(beta, alpha, x)  # shape (N, J, K)
    w_pos = relu(weight).sum(dim=1)  # shape (N, K)
    w_neg = relu(-weight).sum(dim=1)  # shape (N, K)
    # import pdb; pdb.set_trace()
    K = w_pos.shape[-1]
    eta_pos_temp = 1 / (torch.exp(w_pos).sum(dim=1) - K + 1)  # shape (N,)
    eta_neg_temp = 1 / (1 - torch.prod(1 - torch.exp(-w_neg), dim=1))  # shape (N,)
    kappa = torch.sum(eta_pos_temp.reshape(-1, 1) * (torch.exp(w_pos) - 1) *
                   (1 - eta_neg_temp.reshape(-1, 1) * torch.exp(-w_neg)),
                   dim=1)  # shape (N,)
    
    eta_temp = 1 / (1 - kappa)  # shape (N,)
    ig_temp = eta_temp * eta_pos_temp * eta_neg_temp * torch.exp(
        -torch.sum(w_neg, dim=1))
    
    eta_mul = eta_temp * eta_pos_temp * eta_neg_temp  # shape (N,)
    first_term = eta_mul.reshape(-1, 1) * torch.exp(-w_neg)  # shape (N, K)
    prod_term = torch.prod(1 - torch.exp(-w_neg), dim=1, keepdims=True) / torch.maximum((1 - torch.exp(-w_neg)), 1e-25)  # shape (N, 1) / shape (N, K) = shape (N, K)
    # prod_term = torch.prod(1 - torch.exp(-w_neg), dim=1, keepdims=True) / (
    #     1 - torch.exp(-w_neg))  # shape (N, 1) / shape (N, K) = shape (N, K)
    second_term = torch.exp(w_pos) - 1 + prod_term  # shape (N, K)
    m_theta_k = first_term * second_term  # shape (N, K)
    # non_specificity_simple = 1 - my_floor(m_theta_k, precision=15).sum(dim=1)
    non_specificity_simple = 1 - m_theta_k.sum(dim=1)
    # print(non_specificity_simple.min())
    # import pdb; pdb.set_trace()
    
    return kappa, ig_temp, non_specificity_simple

def m_theta_A(beta, alpha, x, A):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]
        A ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    K = alpha.shape[1]
    u_set = torch.arange(K)  # Universal set
    A_comp = complement(u_set, A)

    w_neg = weight_neg(beta, alpha, x)
    eta_pos_temp = eta_pos(beta, alpha, x)
    eta_neg_temp = eta_neg(beta, alpha, x)
    eta_temp = eta(beta, alpha, x)

    w = torch.prod(torch.exp(-w_neg[:, A]), dim=1)
    _w = torch.prod(1 - torch.exp(-w_neg[:, A_comp]), dim=1)

    return eta_temp * eta_pos_temp * eta_neg_temp * w * _w


def PowerSetsBinary(set_):
    s = list(set_)
    powsets = chain.from_iterable(
        combinations(s, r) for r in range(1,
                                          len(s) + 1))
    return list(powsets)


def complement(u_set, A):
    if not isinstance(A, list):
        A = [A]
    A_comp = list(torch.delete(u_set, A))
    return A_comp


def bel(beta, alpha, x, A):
    """[summary]

    Args:
        beta ([type]): [description]
        alpha ([type]): [description]
        x ([type]): [description]
        A ([type]): [description]

    Returns:
        [type]: shape (N,)
    """
    N = x.shape[0]
    subsets = PowerSetsBinary(A)
    bel_temp = torch.zeros(N)
    for subset in subsets:
        if (len(subset) == 1):
            # use m_k when the subset has only one element
            bel_temp = bel_temp + m_theta_k(beta, alpha, x)[:, subset[0]]
        else:
            # use m_A when the subset has more than one element
            bel_temp = bel_temp + m_theta_A(beta, alpha, x, subset)
    return bel_temp


def plausibility(beta, alpha, x, u_set, A):
    com_A = complement(u_set, A)
    # round number to avoid computational error(NAN)
    pl = 1 - torch.round(bel(beta, alpha, x, com_A), 14)
    return pl


def plausibility_transform(beta, alpha, x, u_set):
    # contour transformation
    pl = []
    for theta_k in u_set:
        pl.append(plausibility(beta, alpha, x, u_set, [theta_k]))
    pl = torch.from_array(pl)
    pl_sum = pl.sum(dim=0, keepdims=True)
    pm = pl / pl_sum
    return pm


def H_conflict(beta, alpha, x, u_set):
    # plausibility transform (Voorbraak,1989; Shenoy,2006)
    N = x.shape[0]
    sum_pl = torch.zeros(N)  # normalization constant for plausibility transform
    p_m = []
    for theta_k in u_set:
        sum_pl = sum_pl + plausibility(beta, alpha, x, u_set, [theta_k])
    for theta_k in u_set:
        p_m.append(plausibility(beta, alpha, x, u_set, [theta_k]) / sum_pl)
    p_m = torch.from_array(p_m)
    # Shannon's entropy
    h_temp = (p_m * torch.log2(1 / p_m)).sum(dim=0)
    return h_temp


def H_non_specificity(beta, alpha, x, u_set):
    powset = PowerSetsBinary(u_set)
    h_temp = []
    for subset in powset:
        set_size = len(subset)
        if set_size > 1:
            h_temp.append(
                m_theta_A(beta, alpha, x, subset) * torch.log2(set_size))
    h_temp = torch.from_array(h_temp)
    h_temp = h_temp.sum(dim=0)
    return h_temp


def H_total(beta, alpha, x, u_set):
    H_total = H_conflict(beta, alpha, x, u_set) + H_non_specificity(
        beta, alpha, x, u_set)
    return H_total
