"""
deepguard/models_deep.py
========================
Deep one-class detectors introduced in Phase 1.

All detectors are trained on BENIGN data only and share the common interface:
fit / score / predict / evaluate / save / load. Window-based autoencoders
additionally expose ``score_flows()`` which maps per-timestep reconstruction
errors back to individual flows (see deepguard.sequences for why this matters).

Detectors
---------
TransformerAEDetector : attention encoder-decoder autoencoder over windows.
USADEtector           : USAD-style adversarial autoencoder (Kandula et al., KDD'20).
DeepSVDDetector       : Deep SVDD one-class deep model at flow level.

create_detector(name)  : factory used by pipelines/config.
"""

from __future__ import annotations

import abc
import pathlib
from typing import Dict, Optional, Union

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from deepguard.sequences import build_eval_windows, scatter_window_errors_to_flows

# ──────────────────────────────────────────────────────────────────────────────
# Shared plumbing for window-based autoencoders
# ──────────────────────────────────────────────────────────────────────────────

class _WindowAEDetector(abc.ABC):
    """Common scoring/threshold/persistence behaviour for window AEs."""

    def __init__(self, window_size: int, threshold_pct: float = 95.0) -> None:
        self.window_size = window_size
        self.threshold_pct = threshold_pct
        self.model = None
        self.threshold: Optional[float] = None
        self.flow_threshold: Optional[float] = None
        self._feature_dim: Optional[int] = None

    # -- to be provided by subclasses ----------------------------------------
    @abc.abstractmethod
    def _build(self, window_size: int, n_features: int):
        """Construct and compile self.model."""

    @abc.abstractmethod
    def _fit_model(self, X_train, X_val, epochs, batch_size, patience) -> None: ...

    @abc.abstractmethod
    def name(self) -> str: ...

    # -- shared ---------------------------------------------------------------
    def fit(
        self,
        X_train_seq: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
    ) -> "_WindowAEDetector":
        import tensorflow as tf

        tf.random.set_seed(42)
        self._feature_dim = X_train_seq.shape[2]
        if self.model is None:
            self.model = self._build(self.window_size, self._feature_dim)
        self._fit_model(X_train_seq, X_val_seq, epochs, batch_size, patience)
        return self

    def score(self, X_seq: np.ndarray) -> np.ndarray:
        """Per-window anomaly score = mean MSE over timesteps and features."""
        if self.model is None:
            raise RuntimeError("Call fit() or load() before score().")
        recon = self.model.predict(np.asarray(X_seq, dtype=np.float32),
                                   batch_size=512, verbose=0)
        return np.mean(np.square(np.asarray(X_seq, dtype=np.float32) - recon),
                       axis=(1, 2))

    def _per_timestep_err(self, X_seq: np.ndarray) -> np.ndarray:
        recon = self.model.predict(np.asarray(X_seq, dtype=np.float32),
                                   batch_size=512, verbose=0)
        return np.mean(np.square(np.asarray(X_seq, dtype=np.float32) - recon), axis=2)

    def score_flows(self, X: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Flow-level anomaly scores.

        Builds overlapping windows over the flow pool, reconstructs every
        timestep, and averages each flow's error across all covering windows.
        This avoids the dilution that made whole-window MSE blind to isolated
        anomalous flows.
        """
        windows, coverage = build_eval_windows(np.asarray(X), self.window_size, stride)
        errs = self._per_timestep_err(windows)
        return scatter_window_errors_to_flows(errs, coverage, len(X))

    def set_threshold(self, X_val_benign_seq: np.ndarray, percentile: float = 95) -> float:
        val_scores = self.score(X_val_benign_seq)
        self.threshold = float(np.percentile(val_scores, percentile))
        return self.threshold

    def set_flow_threshold(self, X_benign: np.ndarray, percentile: float = 95,
                           stride: int = 1) -> float:
        """Calibrate the flow-level decision threshold on benign calibration flows."""
        s = self.score_flows(X_benign, stride=stride)
        self.flow_threshold = float(np.percentile(s, percentile))
        return self.flow_threshold

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Set threshold via set_threshold() before predict().")
        return (self.score(X_seq) > self.threshold).astype(int)

    def predict_flows(self, X: np.ndarray, stride: int = 1) -> np.ndarray:
        if self.flow_threshold is None:
            raise RuntimeError("Set flow_threshold via set_flow_threshold() first.")
        return (self.score_flows(X, stride=stride) > self.flow_threshold).astype(int)

    def evaluate(self, X_seq: np.ndarray, y_seq: np.ndarray) -> Dict[str, float]:
        scores = self.score(X_seq)
        y_pred = (scores > self.threshold).astype(int)
        return {
            "precision": float(precision_score(y_seq, y_pred, zero_division=0)),
            "recall": float(recall_score(y_seq, y_pred, zero_division=0)),
            "f1": float(f1_score(y_seq, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_seq, scores)),
            "pr_auc": float(average_precision_score(y_seq, scores)),
            "threshold": self.threshold,
        }

    def evaluate_flows(self, X: np.ndarray, y_flow: np.ndarray, stride: int = 1) -> Dict[str, float]:
        scores = self.score_flows(X, stride=stride)
        y_pred = (scores > self.flow_threshold).astype(int)
        cm = confusion_matrix(y_flow, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return {
            "precision": float(precision_score(y_flow, y_pred, zero_division=0)),
            "recall": float(recall_score(y_flow, y_pred, zero_division=0)),
            "f1": float(f1_score(y_flow, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_flow, scores)),
            "pr_auc": float(average_precision_score(y_flow, scores)),
            "fpr": fp / (tn + fp) if (tn + fp) else 0.0,
            "fnr": fn / (tp + fn) if (tp + fn) else 0.0,
            "flow_threshold": self.flow_threshold,
        }

    def _save_common(self, path: pathlib.Path, extra_params: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path) + "_model.keras")
        joblib.dump({
            "window_size": self.window_size,
            "threshold_pct": self.threshold_pct,
            "_feature_dim": self._feature_dim,
            "threshold": self.threshold,
            "flow_threshold": self.flow_threshold,
            **extra_params,
        }, str(path) + "_params.pkl")

    def _load_common(self, path: pathlib.Path) -> Dict:
        import tensorflow as tf

        params = joblib.load(str(path) + "_params.pkl")
        self.window_size = params["window_size"]
        self.threshold_pct = params["threshold_pct"]
        self._feature_dim = params["_feature_dim"]
        self.threshold = params.get("threshold")
        self.flow_threshold = params.get("flow_threshold")
        self.model = tf.keras.models.load_model(str(path) + "_model.keras")
        return params


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

def _make_positional_encoding_layer():
    """
    Build the registered PositionalEncoding layer class.

    Defined via a factory so TensorFlow is only imported when this is first
    needed; registration lets keras.load_model resolve the custom layer.
    """
    import numpy as _np
    import tensorflow as _tf
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.utils import register_keras_serializable

    @register_keras_serializable(package="deepguard")
    class PositionalEncoding(Layer):
        """Fixed sinusoidal positional encoding, broadcast over the batch."""

        def __init__(self, window_size: int, d_model: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self.window_size = window_size
            self.d_model = d_model
            pos = _np.arange(window_size)[:, None]
            i = _np.arange(d_model)[None, :]
            angle = pos / _np.power(10000.0, (2 * (i // 2)) / max(d_model, 1))
            pe = _np.zeros((window_size, d_model), dtype=_np.float32)
            pe[:, 0::2] = _np.sin(angle[:, 0::2])
            pe[:, 1::2] = _np.cos(angle[:, 1::2])
            self._pe = pe

        def call(self, x):
            return x + _tf.constant(self._pe)[None, :, :]

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"window_size": self.window_size, "d_model": self.d_model})
            return cfg

    return PositionalEncoding


PositionalEncoding = _make_positional_encoding_layer()


class TransformerAEDetector(_WindowAEDetector):
    """
    Attention-based autoencoder over flow windows.

    Learned-representation pipeline: linear projection + sinusoidal positional
    encoding → MHSA/FFN encoder block → token-wise latent compression →
    decoder block → per-timestep reconstruction. Attention captures long-range
    temporal patterns within a window that an LSTM summarises only sequentially.
    """

    def __init__(
        self,
        window_size: int = 50,
        d_model: int = 64,
        num_heads: int = 4,
        latent_dim: int = 32,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        threshold_pct: float = 95.0,
    ) -> None:
        super().__init__(window_size, threshold_pct)
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.lr = lr

    def name(self) -> str:
        return "transformer_ae"

    def _build(self, window_size: int, n_features: int):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Add,
            Dense,
            Dropout,
            Input,
            LayerNormalization,
            MultiHeadAttention,
            TimeDistributed,
        )
        from tensorflow.keras.models import Model as KModel

        def enc_block(x, tag):
            h = LayerNormalization(name=f"{tag}_ln1")(x)
            att = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate, name=f"{tag}_mha",
            )(h, h)
            x = Add(name=f"{tag}_add1")([x, att])
            h = LayerNormalization(name=f"{tag}_ln2")(x)
            f = Dense(self.d_model * 2, activation="gelu", name=f"{tag}_ff1")(h)
            f = Dropout(self.dropout_rate, name=f"{tag}_ffd")(f)
            f = Dense(self.d_model, name=f"{tag}_ff2")(f)
            return Add(name=f"{tag}_add2")([x, f])

        W, F, dm = window_size, n_features, self.d_model
        inp = Input(shape=(W, F), name="input")
        x = Dense(dm, name="proj")(inp)
        x = PositionalEncoding(W, dm, name="pos_enc")(x)

        x = enc_block(x, "enc")
        z = Dense(self.latent_dim, activation="relu", name="latent")(x)
        x = Dense(dm, name="dec_proj")(z)
        x = enc_block(x, "dec")
        out = TimeDistributed(Dense(F, activation="linear"), name="output")(x)

        model = KModel(inp, out, name="transformer_autoencoder")
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr, clipnorm=1.0), loss="mse")
        return model

    def _fit_model(self, X_train, X_val, epochs, batch_size, patience) -> None:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2),
                              min_lr=1e-6, verbose=0),
        ]
        val = (X_val, X_val) if X_val is not None else None
        self.model.fit(X_train, X_train, validation_data=val, epochs=epochs,
                       batch_size=batch_size, callbacks=callbacks, verbose=0)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        self._save_common(path, {
            "d_model": self.d_model, "num_heads": self.num_heads,
            "latent_dim": self.latent_dim, "dropout_rate": self.dropout_rate,
            "lr": self.lr,
        })
        print(f"TransformerAEDetector saved → {path}[_*]")

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "TransformerAEDetector":
        params = joblib.load(str(pathlib.Path(path)) + "_params.pkl")
        obj = cls(
            window_size=params["window_size"], d_model=params["d_model"],
            num_heads=params["num_heads"], latent_dim=params["latent_dim"],
            dropout_rate=params["dropout_rate"], lr=params["lr"],
            threshold_pct=params["threshold_pct"],
        )
        obj._load_common(pathlib.Path(path))
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# USAD — UnSupervised Anomaly Detection (adversarial AE)
# ──────────────────────────────────────────────────────────────────────────────

class USADEtector(_WindowAEDetector):
    """
    USAD-style adversarial autoencoder (Audor et al., KDD 2020) on flattened
    windows. Encoder E compresses a window; decoder D reconstructs it; a
    discriminator D_f learns to tell originals from reconstructions. Training
    alternates a reconstruction phase and an adversarial phase in which E,D are
    pushed to fool D_f. Score combines reconstruction error with the
    discriminator's rejection of the reconstruction.
    """

    def __init__(
        self,
        window_size: int = 50,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        alpha: float = 0.5,
        lr: float = 1e-3,
        threshold_pct: float = 95.0,
    ) -> None:
        super().__init__(window_size, threshold_pct)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.lr = lr
        self.E = self.D = self.Df = None

    def name(self) -> str:
        return "usad"

    def _build(self, window_size: int, n_features: int):
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Flatten, Input
        from tensorflow.keras.models import Model as KModel

        W, F, L = window_size, n_features, self.latent_dim
        H = self.hidden_dim

        inp = Input(shape=(W, F))
        flat = Flatten()(inp)
        z = Dense(L, activation="relu")(Dense(H, activation="relu")(flat))
        self.E = KModel(inp, z, name="usad_encoder")

        zin = Input(shape=(L,))
        xr = Dense(W * F)(Dense(H, activation="relu")(zin))
        self.D = KModel(zin, xr, name="usad_decoder")

        rin = Input(shape=(W * F,))
        verdict = Dense(1, activation="sigmoid")(Dense(H, activation="relu")(rin))
        self.Df = KModel(rin, verdict, name="usad_discriminator")

        opt = tf.keras.optimizers.Adam(self.lr)
        bce = tf.keras.losses.BinaryCrossentropy()
        mae = tf.keras.losses.MeanAbsoluteError()
        self._opt_g = opt                      # updates E and D
        self._opt_d = tf.keras.optimizers.Adam(self.lr)  # updates Df
        self._bce = bce
        self._mae = mae
        return None  # USAD manages its own modules; base-class self.model unused

    def _fit_model(self, X_train, X_val, epochs, batch_size, patience) -> None:
        import tensorflow as tf

        assert self.E is not None, "_build() must run before _fit_model()"
        X = np.concatenate([X_train, X_val]) if X_val is not None else X_train
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        ds = ds.shuffle(min(len(X), 10000)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        a = self.alpha
        for _ in range(max(1, epochs)):
            for x_batch in ds:
                B = tf.shape(x_batch)[0]
                xf = tf.reshape(x_batch, (B, -1))
                e_vars = self.E.trainable_variables + self.D.trainable_variables

                # Phase 1 — reconstruction
                with tf.GradientTape() as t1:
                    rec = self.D(self.E(x_batch, training=True), training=True)
                    loss_rec = self._mae(xf, rec)
                grads = t1.gradient(loss_rec, e_vars)
                self._opt_g.apply_gradients(zip(grads, e_vars))

                # Phase 2 — adversarial (generator side)
                with tf.GradientTape() as t2:
                    z = self.E(x_batch, training=True)
                    rec = self.D(z, training=True)
                    g_loss = a * self._mae(xf, rec) + (1 - a) * self._bce(
                        tf.ones_like(self.Df(rec, training=True)),
                        self.Df(rec, training=True),
                    )
                grads = t2.gradient(g_loss, e_vars)
                self._opt_g.apply_gradients(zip(grads, e_vars))

                # Phase 2 — adversarial (discriminator side)
                d_vars = self.Df.trainable_variables
                with tf.GradientTape() as t3:
                    d_loss = self._bce(tf.ones_like(self.Df(xf, training=True)),
                                       self.Df(xf, training=True)) + \
                             self._bce(tf.zeros_like(self.Df(rec, training=True)),
                                       self.Df(tf.stop_gradient(rec), training=True))
                grads = t3.gradient(d_loss, d_vars)
                self._opt_d.apply_gradients(zip(grads, d_vars))

    def score(self, X_seq: np.ndarray) -> np.ndarray:
        if self.E is None:
            raise RuntimeError("Call fit() or load() before score().")
        import tensorflow as tf

        X = np.asarray(X_seq, dtype=np.float32)
        outs = []
        for i in range(0, len(X), 512):
            xb = tf.convert_to_tensor(X[i : i + 512])
            B = tf.shape(xb)[0]
            xf = tf.reshape(xb, (B, -1))
            rec = self.D(self.E(xb, training=False), training=False)
            l1 = tf.reduce_mean(tf.abs(xf - rec), axis=1)
            adv = -tf.math.log(self.Df(rec, training=False)[:, 0] + 1e-8)
            outs.append(
                (self.alpha * l1 + (1 - self.alpha) * adv).numpy()
            )
        return np.concatenate(outs)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.E.save(str(path) + "_E.keras")
        self.D.save(str(path) + "_D.keras")
        self.Df.save(str(path) + "_Df.keras")
        joblib.dump({
            "window_size": self.window_size, "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim, "alpha": self.alpha, "lr": self.lr,
            "threshold_pct": self.threshold_pct, "threshold": self.threshold,
            "flow_threshold": self.flow_threshold,
            "_feature_dim": self._feature_dim,
        }, str(path) + "_params.pkl")
        print(f"USADEtector saved → {path}[_E/_D/_Df/_params]")

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "USADEtector":
        import tensorflow as tf

        path = pathlib.Path(path)
        p = joblib.load(str(path) + "_params.pkl")
        obj = cls(
            window_size=p["window_size"], hidden_dim=p["hidden_dim"],
            latent_dim=p["latent_dim"], alpha=p["alpha"], lr=p["lr"],
            threshold_pct=p["threshold_pct"],
        )
        obj._feature_dim = p["_feature_dim"]
        obj.threshold = p.get("threshold")
        obj.flow_threshold = p.get("flow_threshold")
        obj.E = tf.keras.models.load_model(str(path) + "_E.keras")
        obj.D = tf.keras.models.load_model(str(path) + "_D.keras")
        obj.Df = tf.keras.models.load_model(str(path) + "_Df.keras")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Deep SVDD — flow-level one-class deep model
# ──────────────────────────────────────────────────────────────────────────────

class DeepSVDDetector:
    """
    Deep Support Vector Data Description (Ruff et al., ICML 2018), MLP variant.

    Trains an encoder φ so that benign flows map tightly around a center c;
    score(x) = ||φ(x) − c||². Operates directly on single flows — a natural
    non-Gaussian complement to the GMM at flow level.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 512,
        patience: int = 10,
        threshold_pct: float = 95.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.threshold_pct = threshold_pct
        self.center: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self._encoder = None
        self._feature_dim: Optional[int] = None

    def name(self) -> str:
        return "deep_svdd"

    def _build_encoder(self, n_features: int):
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model as KModel

        inp = Input(shape=(n_features,))
        x = Dense(self.hidden_dim, activation="relu")(inp)
        z = Dense(self.latent_dim, use_bias=False)(x)   # bias-free last layer (SVDD)
        return KModel(inp, z, name="svdd_encoder")

    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            epochs: Optional[int] = None, batch_size: Optional[int] = None,
            patience: Optional[int] = None) -> "DeepSVDDetector":
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(42)
        X = np.asarray(X_train, dtype=np.float32)
        self._feature_dim = X.shape[1]
        self._encoder = self._build_encoder(self._feature_dim)

        # Center = mean initial embedding (paper: set c before optimization)
        self.center = self._encoder.predict(X[: min(len(X), 50_000)], verbose=0).mean(axis=0)
        c = tf.constant(self.center, dtype=tf.float32)

        def svdd_loss(_y_true, y_pred):
            return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - c), axis=1))

        self._encoder.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=svdd_loss)
        dummy = np.zeros((len(X), 1), dtype=np.float32)
        val = None
        cb = []
        if X_val is not None and len(X_val):
            Xv = np.asarray(X_val, dtype=np.float32)
            val = (Xv, np.zeros((len(Xv), 1), dtype=np.float32))
            cb.append(EarlyStopping(monitor="val_loss", patience=patience or self.patience,
                                    restore_best_weights=True, verbose=0))
        self._encoder.fit(
            X, dummy, validation_data=val,
            epochs=epochs or self.epochs, batch_size=batch_size or self.batch_size,
            callbacks=cb, verbose=0,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None or self.center is None:
            raise RuntimeError("Call fit() or load() before score().")
        Z = self._encoder.predict(np.asarray(X, dtype=np.float32), batch_size=1024, verbose=0)
        return np.sum((Z - self.center[None, :]) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Set threshold via set_threshold() first.")
        return (self.score(X) > self.threshold).astype(int)

    def set_threshold(self, X_benign: np.ndarray, percentile: float = 95) -> float:
        self.threshold = float(np.percentile(self.score(X_benign), percentile))
        return self.threshold

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        scores = self.score(X)
        y_pred = (scores > self.threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return {
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
            "fpr": fp / (tn + fp) if (tn + fp) else 0.0,
            "fnr": fn / (tp + fn) if (tp + fn) else 0.0,
            "threshold": self.threshold,
        }

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._encoder.save_weights(str(path) + "_enc.weights.h5")
        joblib.dump({
            "hidden_dim": self.hidden_dim, "latent_dim": self.latent_dim,
            "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size,
            "patience": self.patience, "threshold_pct": self.threshold_pct,
            "center": self.center, "threshold": self.threshold,
            "_feature_dim": self._feature_dim,
        }, str(path) + "_params.pkl")
        print(f"DeepSVDDetector saved → {path}[_enc/_params]")

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "DeepSVDDetector":
        path = pathlib.Path(path)
        p = joblib.load(str(path) + "_params.pkl")
        obj = cls(
            hidden_dim=p["hidden_dim"], latent_dim=p["latent_dim"], lr=p["lr"],
            epochs=p["epochs"], batch_size=p["batch_size"], patience=p["patience"],
            threshold_pct=p["threshold_pct"],
        )
        obj.center = p["center"]
        obj.threshold = p.get("threshold")
        obj._feature_dim = p["_feature_dim"]
        obj._encoder = obj._build_encoder(obj._feature_dim)
        obj._encoder.load_weights(str(path) + "_enc.weights.h5")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    "lstm_ae": "deepguard.models:LSTMAEDetector",
    "transformer_ae": TransformerAEDetector,
    "usad": USADEtector,
    "deep_svdd": DeepSVDDetector,
}


def available_models() -> list:
    return sorted(_REGISTRY)


def create_detector(name: str, **kwargs):
    """Instantiate a detector by config name ('lstm_ae'|'transformer_ae'|'usad'|'deep_svdd')."""
    target = _REGISTRY.get(name)
    if target is None:
        raise KeyError(f"Unknown detector '{name}'. Available: {available_models()}")
    if isinstance(target, str):
        module, _, attr = target.partition(":")
        import importlib

        cls = getattr(importlib.import_module(module), attr)
    else:
        cls = target
    return cls(**kwargs)


def load_detector(name: str, path: Union[str, pathlib.Path]):
    """Load a persisted detector by name."""
    det = create_detector(name)
    return det.load(path)
