from __future__ import annotations
from typing import Dict, List, Any, Optional, Iterable, Tuple, Union, Callable
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer


# ======================================================
# 0) OPTIONAL: Pandas-friendly transformers for CAPPING
# ======================================================
ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


class DomainClipper(BaseEstimator, TransformerMixin):
    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None, add_flag: bool = True):
        self.bounds = bounds
        self.add_flag = add_flag
        self._seen_cols: Optional[List[str]] = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._seen_cols = list(X.columns)
        return self

    def transform(self, X):
        was_nd = isinstance(X, np.ndarray)
        was_series = isinstance(X, pd.Series)
        if was_series:
            X = X.to_frame()
        if was_nd:
            return X
        Xc = X.copy()
        bounds = self.bounds or {}
        for col, (mn, mx) in bounds.items():
            if col in Xc.columns:
                before = Xc[col]
                Xc[col] = before.clip(lower=mn, upper=mx)
                if self.add_flag:
                    Xc[f"{col}__domain_clipped"] = ((before < mn) | (before > mx)).astype(np.uint8)
        return Xc

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self._seen_cols or []
        output_features = list(input_features)
        if self.add_flag and self.bounds:
            for col in self.bounds:
                if col in input_features:
                    output_features.append(f"{col}__domain_clipped")
        return np.array(output_features, dtype=object)


class QuantileCapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: Iterable[str],
        lower_q: Optional[float] = None,
        upper_q: Optional[float] = 0.99,
        add_flag: bool = True,
        clip_first: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        # penting: tuple agar aman di-clone
        self.columns = tuple(columns) if columns is not None else tuple()
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.add_flag = add_flag
        self.clip_first = clip_first
        self.quantiles_: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._seen_cols: Optional[List[str]] = None

    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, pd.Series):
            return X.to_frame()
        return pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

    def fit(self, X, y=None):
        Xdf = self._as_df(X)
        self._seen_cols = Xdf.columns.tolist()
        Xc = Xdf.copy()

        # opsional: clip terlebih dulu dengan domain bounds agar quantile stabil
        clip_first = self.clip_first or {}
        for col, (mn, mx) in clip_first.items():
            if col in Xc.columns:
                Xc[col] = Xc[col].clip(lower=mn, upper=mx)

        # hitung quantile untuk kolom yang ada
        self.quantiles_.clear()
        for col in self.columns:
            if col not in Xc.columns:
                continue
            lo = np.nanquantile(Xc[col], self.lower_q, method="linear") if self.lower_q is not None else None
            hi = np.nanquantile(Xc[col], self.upper_q, method="linear") if self.upper_q is not None else None
            self.quantiles_[col] = (lo, hi)
        return self

    def transform(self, X):
        Xdf = self._as_df(X)
        Xc = Xdf.copy()
        for col in self.columns:
            if col not in Xc.columns:
                continue
            lo, hi = self.quantiles_.get(col, (None, None))
            before = Xc[col]
            capped = before
            if lo is not None:
                capped = np.maximum(capped, lo)
            if hi is not None:
                capped = np.minimum(capped, hi)
            Xc[col] = capped
            if self.add_flag:
                flag_col = f"{col}__capped"
                Xc[flag_col] = (((before < lo) if lo is not None else False) |
                                ((before > hi) if hi is not None else False)).astype(np.uint8)
        return Xc

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self._seen_cols or []
        output_features = list(input_features)
        if self.add_flag:
            for col in self.columns:
                if col in input_features:
                    output_features.append(f"{col}__capped")
        return np.array(output_features, dtype=object)


# ======================================================
# 0.5) GENERIC DERIVED-FEATURES TRANSFORMER (DECLARATIVE)
# ======================================================
Spec = Dict[str, Any]


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """
    specs item:
      {
        "name": "feature_a",
        "expr": "( `b` * `c` ) / `x`",   # atau pakai C('b') dsb; juga boleh operator ==, !=, <, >, <=, >=, &, |, ~
        "func": Callable[[pd.DataFrame], pd.Series] | None,
        "requires": ["b","c","x"],       # opsional
        "type": "numeric" | "nominal" | "ordinal",
        "clip": (lo, hi) | None,
        "fillna": value | None,
        "on_missing": "skip" | "raise"
      }
    - Backtick `...` akan diubah menjadi C("...") agar kolom ber-spasi/koma/kurung bisa direferensikan aman.
    """
    _BT = re.compile(r"`([^`]+)`")  # backtick matcher

    def __init__(self, specs: Optional[List[Dict[str, Any]]] = None, strict: bool = False, max_pass: int = 4, verbose: bool = False):
        self.specs = specs or []
        self.strict = strict
        self.max_pass = max_pass
        self.verbose = verbose
        self._seen_cols: Optional[List[str]] = None
        self._made_cols: List[str] = []
        self._colmap: Dict[str, str] = {}   # casefold -> actual

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._seen_cols = list(X.columns)
            self._colmap = {str(c).casefold(): str(c) for c in X.columns}
        else:
            self._seen_cols = None
            self._colmap = {}
        return self

    @staticmethod
    def _preprocess_expr(expr: str) -> str:
        def repl(m): return f'C("{m.group(1)}")'
        return DerivedFeatures._BT.sub(repl, expr or "")

    def _safe_eval_expr(self, df: pd.DataFrame, expr: str) -> pd.Series:
        # izinkan operator aritmetika, perbandingan, logika, kurung, koma, titik, tanda kutip
        allowed = re.compile(r"^[\w\s\+\-\*\/\%\(\)\.\,\[\]\'\"\<\>\=\!\&\|\~]+$")
        e = (expr or "").replace("\n", " ").strip()
        if not allowed.match(e):
            raise ValueError(f"Ekspresi mengandung token tidak diizinkan: {expr!r}")

        # resolver kolom berbasis df.columns saat ini (case-insensitive)
        def C(name: str) -> pd.Series:
            zmap = {str(c).casefold(): str(c) for c in df.columns}
            actual = zmap.get(str(name).casefold(), name)
            if actual not in df.columns:
                raise KeyError(f"C('{name}') tidak ditemukan di kolom input.")
            return df[actual]

        local_ns = {"np": np, "pd": pd, "C": C}
        out = eval(e, {"__builtins__": {}}, local_ns)  # aman karena namespace terkontrol
        s = out if isinstance(out, pd.Series) else pd.Series(out, index=df.index)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s

    def transform(self, X):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Z = Xdf.copy()

        pending = list(self.specs)
        self._made_cols = []
        made_any = True
        passes = 0

        while pending and made_any and passes < self.max_pass:
            made_any = False
            passes += 1

            # refresh peta kolom berdasarkan Z terbaru
            self._colmap = {str(c).casefold(): str(c) for c in Z.columns}

            next_pending = []
            for spec in pending:
                name    = spec["name"]
                expr    = spec.get("expr")
                func    = spec.get("func")
                req     = spec.get("requires")
                clip    = spec.get("clip")
                fillna  = spec.get("fillna")
                on_miss = spec.get("on_missing", "raise" if self.strict else "skip")

                # cek requirement (case-insensitive) terhadap Z saat ini
                def has_req() -> bool:
                    if not req:
                        return True
                    for r in req:
                        key = str(r).casefold()
                        actual = self._colmap.get(key, None)
                        if actual not in Z.columns:
                            return False
                    return True

                if not has_req():
                    if on_miss == "skip":
                        next_pending.append(spec)
                        if self.verbose:
                            print(f"[DerivedFeatures] (pass {passes}) '{name}' ditunda (dependency belum ada).")
                        continue
                    if self.strict:
                        missing = [r for r in (req or []) if self._colmap.get(str(r).casefold(), None) not in Z.columns]
                        raise KeyError(f"[DerivedFeatures] missing {missing} for '{name}'")
                    next_pending.append(spec)
                    continue

                try:
                    if func is not None:
                        s = func(Z)
                        if not isinstance(s, pd.Series):
                            s = pd.Series(s, index=Z.index)
                    elif expr is not None:
                        pre = self._preprocess_expr(expr)
                        s = self._safe_eval_expr(Z, pre)
                    else:
                        raise ValueError(f"[DerivedFeatures] '{name}' requires expr or func")

                    if clip is not None:
                        lo, hi = clip
                        s = s.clip(lower=lo, upper=hi)
                    if fillna is not None:
                        s = s.fillna(fillna)
                    if s.dtype.kind in "iub":
                        s = s.astype(float)

                    Z[name] = s
                    self._made_cols.append(name)
                    made_any = True

                    # update peta kolom segera setelah fitur dibuat
                    self._colmap[str(name).casefold()] = name

                except Exception as ex:
                    if self.verbose:
                        print(f"[DerivedFeatures] '{name}' gagal: {ex}")
                    if self.strict:
                        raise
                    next_pending.append(spec)

            pending = next_pending

        if pending and self.strict:
            names = [p["name"] for p in pending]
            raise RuntimeError(f"[DerivedFeatures] unresolved derived features after {self.max_pass} passes: {names}")
        return Z

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else (self._seen_cols or [])
        return np.array(base + self._made_cols, dtype=object)


# ======================================================
# 1) SCHEMA (EDIT SESUAI DATA) — with 'derived' inside
# ======================================================
def get_feature_schema() -> Dict[str, Any]:
    s = {
        # Continuous numbers
        "numeric": [
            "tenure", 
            "MonthlyCharges", 
            "TotalCharges"
        ],

        # For your custom BinaryMapTransformer
        # Note: I added SeniorCitizen here just to ensure it stays an integer, 
        # even though it's already 1 and 0 in the raw data.
        "binary_map": {
            "gender": {"female": 1, "male": 0},
            "Partner": {"yes": 1, "no": 0},
            "Dependents": {"yes": 1, "no": 0},
            "PhoneService": {"yes": 1, "no": 0},
            "PaperlessBilling": {"yes": 1, "no": 0},
            "SeniorCitizen": {"1": 1, "0": 0} 
        },

        # Features with 3+ text categories that need One-Hot Encoding
        "categorical_nominal": [
            "MultipleLines", 
            "InternetService", 
            "OnlineSecurity", 
            "OnlineBackup", 
            "DeviceProtection", 
            "TechSupport", 
            "StreamingTV", 
            "StreamingMovies", 
            "Contract", 
            "PaymentMethod"
        ],

        # No strict ordinal columns in the basic telecom churn dataset
        "categorical_ordinal": {},

        "drop": ["id"],
        "target": "Churn",

        # Optional: Cap the top 1% of TotalCharges so ultra-long tenure 
        # customers don't skew your distance-based models (like KNN/Logistic Regression)
        "capping": {
            "domain_bounds": {},
            "quantile": {
                "columns": ["TotalCharges", "MonthlyCharges"],
                "lower_q": None,
                "upper_q": 0.99,
            },
        },

        # ==== DERIVED DECLARATIONS ====
        # Let's put your generic expression engine to work!
        "derived": [
            {
                # Creates a new feature converting months to years
                "name": "tenure_years",
                "expr": "`tenure` / 12.0",
                "requires": ["tenure"],
                "type": "numeric",
                "on_missing": "skip"
            },
            {
                # Checks if the customer is paying more than their base monthly rate 
                # (e.g., did they get hit with extra usage fees this month?)
                "name": "monthly_vs_average",
                "expr": "`MonthlyCharges` - (`TotalCharges` / `tenure`)",
                "requires": ["MonthlyCharges", "TotalCharges", "tenure"],
                "type": "numeric",
                "on_missing": "skip"
            }
        ],
        
        # Optional: ensure all text in these columns is lowercase before OHE to prevent duplicates
        "lowercase_value_cols": [
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
            "Contract", "PaymentMethod"
        ]
    }

    # Add derived numeric features to the numeric list so they get scaled/imputed
    derived_numeric = [d["name"] for d in s.get("derived", []) if d.get("type") == "numeric"]
    s["numeric"] = list(s.get("numeric", [])) + derived_numeric
    return s


# =========================================
# 2) HELPERS
# =========================================
def _lowercase_values(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    Z = df.copy()
    for c in cols:
        if c in Z.columns and Z[c].dtype == "object":
            Z[c] = Z[c].astype(str).str.strip().str.lower()
    return Z


def _astype_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    Z = df.copy()
    for c in cols:
        if c in Z.columns:
            Z[c] = Z[c].astype(str)
    return Z


def _normalize_binary_map(df_bin: pd.DataFrame, column_configs: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    out = {}
    col_cfg = {str(k).lower(): {str(kk).lower(): int(vv) for kk, vv in v.items()} for k, v in (column_configs or {}).items()}
    for c in df_bin.columns:
        s = df_bin[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float)
            continue
        v = s.astype(str).str.strip().str.lower()
        cfg = col_cfg.get(c.lower(), None)
        if cfg is None:
            raise ValueError(f"[binary_map] kolom '{c}' tidak punya mapping. Tambahkan di schema['binary_map'].")
        mapped = v.map(cfg).astype(float)
        if mapped.isna().any():
            bad = v[mapped.isna()].unique().tolist()
            raise ValueError(f"[binary_map] Nilai tak terpetakan pada kolom '{c}': {bad}. Lengkapi mapping.")
        out[c] = mapped
    return pd.DataFrame(out, index=df_bin.index)


class BinaryMapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_configs: Optional[Dict[str, Dict[str, int]]] = None):
        self.column_configs = column_configs or {}
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        return _normalize_binary_map(X, self.column_configs)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ or input_features, dtype=object)


# =========================================
# 3) FACTORY: MAKE_PREPROCESSOR (Derived inside schema; step optional)
# =========================================
def make_preprocessor(
    schema: Optional[Dict[str, Any]] = None,
    use_imputers: bool = False,
    scale_numeric: bool = False,
    ohe_dense: bool = False,
    use_capping: bool = True,
    add_capping_flags: bool = True,
) -> Pipeline:
    schema = schema or get_feature_schema()

    num_cols = list(schema.get("numeric", []))
    bin_map_cols = list(schema.get("binary_map", {}).keys())
    ord_map: Dict[str, List[Any]] = schema.get("categorical_ordinal", {})
    ord_cols = list(ord_map.keys())
    nom_cols = list(schema.get("categorical_nominal", []))
    lower_cols = list(schema.get("lowercase_value_cols", []))

    cap_cfg = (schema.get("capping") or {}) if use_capping else {}
    domain_bounds: Dict[str, Tuple[float, float]] = cap_cfg.get("domain_bounds", {})
    q_cfg: Dict[str, Any] = cap_cfg.get("quantile", {}) or {}
    q_cols: List[str] = list(q_cfg.get("columns", []) or [])
    q_lower = q_cfg.get("lower_q", None)
    q_upper = q_cfg.get("upper_q", None)

    # Cleaner depan (opsional)
    lc = FunctionTransformer(_lowercase_values, kw_args={"cols": lower_cols}, feature_names_out="one-to-one")

    # ===== OPTIONAL: DERIVED FEATURES STEP (only if specs exist) =====
    derived_specs = list(schema.get("derived", []) or [])
    steps: List[Tuple[str, Any]] = [("clean_values", lc)]
    if derived_specs:
        steps.append(("derive_features", DerivedFeatures(specs=derived_specs, strict=False, max_pass=4, verbose=False)))

    # ===== Numeric pipeline =====
    numeric_steps: List[Tuple[str, Any]] = []
    if use_capping and domain_bounds:
        numeric_steps.append(("domain_clip", DomainClipper(bounds=domain_bounds, add_flag=add_capping_flags)))
    if use_capping and q_cols:
        numeric_steps.append(("quantile_cap", QuantileCapper(columns=q_cols, lower_q=q_lower, upper_q=q_upper,
                                                             add_flag=add_capping_flags, clip_first=domain_bounds)))
    if use_imputers:
        numeric_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    # ===== Binary pipeline =====
    binary_pipe = Pipeline([("binmap", BinaryMapTransformer(column_configs=schema.get("binary_map", {})))])

    # ===== Ordinal pipeline (cast to str → encode) =====
    if ord_cols:
        ord_categories = [[str(v) for v in ord_map[c]] for c in ord_cols]
        ord_steps: List[Tuple[str, Any]] = []
        ord_steps.append(("to_str", FunctionTransformer(_astype_str, kw_args={"cols": ord_cols}, feature_names_out="one-to-one")))
        if use_imputers:
            ord_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        ord_steps.append(("ord", OrdinalEncoder(
            categories=ord_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )))
        ordinal_pipe = Pipeline(ord_steps)
    else:
        ordinal_pipe = None

    # ===== Nominal pipeline (OHE) =====
    ohe_kwargs = dict(handle_unknown="ignore")
    if ohe_dense:
        try:
            ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
        except TypeError:
            ohe = OneHotEncoder(sparse=False, **ohe_kwargs)
    else:
        try:
            ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
        except TypeError:
            ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

    cat_nom_steps: List[Tuple[str, Any]] = []
    if use_imputers:
        cat_nom_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    cat_nom_steps.append(("ohe", ohe))
    cat_nom_pipe = Pipeline(cat_nom_steps)

    # ===== ColumnTransformer =====
    transformers: List[Tuple[str, Any, List[str]]] = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if bin_map_cols:
        transformers.append(("bin", binary_pipe, bin_map_cols))
    if ord_cols:
        transformers.append(("ord", ordinal_pipe, ord_cols))
    if nom_cols:
        transformers.append(("nom", cat_nom_pipe, nom_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    steps.append(("ct", ct))
    preprocessor = Pipeline(steps=steps)
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass
    return preprocessor


# =========================================
# 4) UTIL: FEATURE NAMES SAFE FALLBACK
# =========================================
def safe_feature_names(preprocessor: Pipeline, nom_cols: Optional[List[str]] = None) -> List[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names: List[str] = []
        ct: ColumnTransformer = preprocessor.named_steps["ct"]
        if "num" in ct.named_transformers_:
            num = ct.transformers_[[t[0] for t in ct.transformers_].index("num")][2]
            names += list(num)
        if "bin" in ct.named_transformers_:
            bin_cols = ct.transformers_[[t[0] for t in ct.transformers_].index("bin")][2]
            names += list(bin_cols)
        if "ord" in ct.named_transformers_:
            ord_cols = ct.transformers_[[t[0] for t in ct.transformers_].index("ord")][2]
            names += [f"{c}_ord" for c in ord_cols]
        if "nom" in ct.named_transformers_:
            nom_cols = nom_cols or ct.transformers_[[t[0] for t in ct.transformers_].index("nom")][2]
            ohe = ct.named_transformers_["nom"].named_steps["ohe"]
            names += ohe.get_feature_names_out(nom_cols).tolist()
        if not names:
            # fallback terakhir (jarang kepakai)
            return [f"f{i}" for i in range(ct.transform(pd.DataFrame()).shape[1])]
        return names
