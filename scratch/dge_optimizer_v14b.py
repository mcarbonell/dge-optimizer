import numpy as np
import math

class DGEOptimizerV14b:
    """
    Denoised Gradient Estimation (DGE) v14b — Orthogonal Blocks.
    
    Cambios vs v13:
    1. Bloques ORTOGONALES (partición sin solape) en vez de muestreo con reemplazo.
       → Cada parámetro recibe exactamente 1 medición por step.
       → Reduce ruido cruzado por sqrt(group_size) de verdad.
    2. lr_scale = 1.0 (eliminado el 1/sqrt(k) que asumía independencia falsa).
    3. Tracking de SNR: correlación entre gradiente instantáneo y EMA.
       → Si el "temporal denoiser" funciona, debe subir con t.
    
    Coste: 2*k forwards por step, con k = ceil(log2(D)).
    """
    def __init__(self, dim: int, lr: float = 1.0, delta: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 lr_decay: float = 0.01, delta_decay: float = 0.05,
                 total_steps: int = 1000, greedy_w: float = 0.0,
                 clip_norm: float = 1.0, seed: int | None = None,
                 track_snr: bool = False):
        self.dim = dim
        self.lr0 = lr
        self.delta0 = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay = lr_decay
        self.delta_decay = delta_decay
        self.total_steps = total_steps
        self.greedy_w = greedy_w
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.track_snr = track_snr
        
        self.k = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        # group_size se calcula dinámicamente con array_split (maneja restos)
        
        # FIX #2: sin penalización por k, los bloques ortogonales ya son insesgados
        self.lr_scale = 1.0
        
        # Adam state
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0
        
        # Diagnóstico
        self.snr_history = []

    def _cosine(self, v0, decay):
        frac = min(self.t / max(self.total_steps, 1), 1.0)
        return v0 * (decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * frac)))

    def step(self, f, x):
        """
        f: callable(x) -> scalar loss. DEBE usar el mismo batch en todas las llamadas
           dentro de este step (crítico para zeroth-order estocástico).
        x: parameter vector (flat, float32).
        Returns: (x_new, num_evaluations)
        """
        self.t += 1
        lr    = self._cosine(self.lr0, self.lr_decay) * self.lr_scale
        delta = self._cosine(self.delta0, self.delta_decay)

        # FIX #1: PARTICIÓN ORTOGONAL en vez de choice con reemplazo
        # Cada parámetro aparece en exactamente 1 bloque por step.
        perm = self.rng.permutation(self.dim)
        groups = np.array_split(perm, self.k)
        
        signs = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

        g = np.zeros(self.dim, dtype=np.float32)
        
        best_s = -1.0
        best_dir = np.zeros(self.dim, dtype=np.float32)

        # Evaluar cada bloque ortogonal
        for idx in groups:
            pert = np.zeros(self.dim, dtype=np.float32)
            pert[idx] = signs[idx] * delta
            
            fp = f(x + pert)
            fm = f(x - pert)
            
            # Estimador SPSA restringido al bloque
            sg = (fp - fm) / (2.0 * delta)
            
            # Asignación: g_i ≈ g_true_i + ruido_cruzado_dentro_del_bloque
            # El ruido cruzado tiene media 0 y se cancela vía Adam EMA.
            g[idx] = sg * signs[idx]
            
            # Greedy direction tracking
            if abs(sg) > best_s:
                best_s = abs(sg)
                d = np.zeros(self.dim, dtype=np.float32)
                d[idx] = signs[idx]
                dn = np.linalg.norm(d)
                best_dir = -np.sign(sg) * d / (dn + 1e-12)

        # YA NO necesitamos g_cnt ni promediado: cada param tiene exactamente 1 medida.
        # YA NO existe el caso "not_ev": todos los parámetros se actualizan cada step.
        
        # FIX #3: Diagnóstico SNR — ¿el EMA está limpiando o solo promediando ruido?
        if self.track_snr and self.t > 1:
            mh_prev = self.m / (1 - self.beta1 ** (self.t - 1) + 1e-30)
            # Correlación entre señal instantánea y memoria acumulada
            if np.std(g) > 1e-12 and np.std(mh_prev) > 1e-12:
                corr = float(np.corrcoef(g, mh_prev)[0, 1])
                self.snr_history.append(corr)
        
        # Adam EMA update (ahora denso, sin máscaras)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2

        mh = self.m / (1 - self.beta1 ** self.t + 1e-30)
        vh = self.v / (1 - self.beta2 ** self.t + 1e-30)

        upd = lr * mh / (np.sqrt(vh) + self.eps)
        
        # Clipping
        un = np.linalg.norm(upd)
        if un > self.clip_norm:
            upd *= self.clip_norm / un

        x_new = x - upd - self.greedy_w * lr * best_dir
        
        return x_new, 2 * self.k
    
    def get_snr_trend(self):
        """Devuelve (corr_inicial, corr_final). Si denoising funciona: final >> inicial."""
        if len(self.snr_history) < 20:
            return None
        early = np.mean(self.snr_history[:10])
        late = np.mean(self.snr_history[-10:])
        return float(early), float(late)