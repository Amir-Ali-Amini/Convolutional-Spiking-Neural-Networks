def soft_hard_bound(w, w_min, w_max):
    return (w > w_min) * (w < w_max) * (w - w_min) * (w_max - w)
