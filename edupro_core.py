"""
EduPro – Student Segmentation & Personalized Course Recommendation engine.
Shared by the Jupyter notebook and the Streamlit app.

Pipeline: load -> (synthetic fill-in if sheets missing) -> learner features ->
preprocess -> K-Means (+ hierarchical validation) -> segment naming ->
hybrid cluster-aware recommender -> evaluation.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
LEVELS = ["Beginner", "Intermediate", "Advanced"]

# --------------------------------------------------------------------------- #
# 1. LOADING
# --------------------------------------------------------------------------- #
_CANON = {
    "userid": "UserID", "username": "UserName", "age": "Age", "gender": "Gender",
    "email": "Email", "courseid": "CourseID", "coursename": "CourseName",
    "coursecategory": "CourseCategory", "coursetype": "CourseType",
    "courselevel": "CourseLevel", "courserating": "CourseRating",
    "courseprice": "CoursePrice", "transactionid": "TransactionID",
    "transactiondate": "TransactionDate", "amount": "Amount",
}


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_CANON.get(re.sub(r"[^a-z]", "", str(c).lower()), str(c).strip()) for c in df.columns]
    return df


def level_score(level) -> int:
    s = str(level).lower()
    if "begin" in s:
        return 1
    if "adv" in s:
        return 3
    return 2


def _clean_level(level) -> str:
    return LEVELS[level_score(level) - 1] if re.search(r"begin|inter|adv", str(level).lower()) else str(level)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = _canon(users).drop_duplicates("UserID")
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    users["Age"] = users["Age"].fillna(users["Age"].median())
    users["Gender"] = users["Gender"].fillna("Unknown").astype(str).str.title()
    return users.reset_index(drop=True)


def clean_courses(courses: pd.DataFrame) -> pd.DataFrame:
    courses = _canon(courses).drop_duplicates("CourseID")
    if "CourseName" not in courses:
        courses["CourseName"] = courses["CourseID"].astype(str)
    courses["CourseRating"] = pd.to_numeric(courses["CourseRating"], errors="coerce")
    courses["CourseRating"] = courses["CourseRating"].fillna(courses["CourseRating"].median())
    courses["CourseLevel"] = courses["CourseLevel"].map(_clean_level)
    courses["LevelScore"] = courses["CourseLevel"].map(level_score)
    return courses.reset_index(drop=True)


def clean_transactions(tx: pd.DataFrame, users: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    tx = _canon(tx)
    tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce").fillna(0.0)
    tx = tx.dropna(subset=["UserID", "CourseID", "TransactionDate"])
    tx = tx[tx.UserID.isin(users.UserID) & tx.CourseID.isin(courses.CourseID)]
    return tx.drop_duplicates(["UserID", "CourseID"]).reset_index(drop=True)


def _find(data_dir: str, key: str):
    hits = [p for p in glob.glob(os.path.join(data_dir, "*.csv")) if key in os.path.basename(p).lower()]
    return sorted(hits)[0] if hits else None


def load_data(data_dir: str = "data", users_df=None, courses_df=None, tx_df=None) -> dict:
    """Load Users/Courses/Transactions. Missing sheets are synthesised (and flagged)."""
    users = users_df if users_df is not None else (
        pd.read_csv(_find(data_dir, "user")) if _find(data_dir, "user") else None)
    courses = courses_df if courses_df is not None else (
        pd.read_csv(_find(data_dir, "course")) if _find(data_dir, "course") else None)
    tx = tx_df if tx_df is not None else (
        pd.read_csv(_find(data_dir, "transaction")) if _find(data_dir, "transaction") else None)
    if users is None:
        raise FileNotFoundError("Users CSV not found (expected data/Users.csv).")

    users = clean_users(users)
    synthetic = []
    if courses is None:
        courses = generate_courses()
        synthetic.append("Courses")
    courses = clean_courses(courses)
    if "CourseType" not in courses:
        courses["CourseType"] = "Paid"
    if tx is None:
        tx = generate_transactions(users, courses)
        synthetic.append("Transactions")
    tx = clean_transactions(tx, users, courses)
    return {"users": users, "courses": courses, "tx": tx, "synthetic": synthetic}


# --------------------------------------------------------------------------- #
# 1b. SYNTHETIC DATA (only used when the real sheets are absent)
# --------------------------------------------------------------------------- #
_TOPICS = {
    "Data Science": ["Python for Data Analysis", "Machine Learning", "Statistics", "Data Visualization"],
    "Web Development": ["HTML & CSS", "JavaScript", "React", "Backend with Node"],
    "Business": ["Management Essentials", "Entrepreneurship", "Project Management", "Business Strategy"],
    "Finance": ["Personal Finance", "Financial Modeling", "Investing", "Accounting"],
    "Design": ["UI/UX Design", "Graphic Design", "Design Thinking", "Motion Graphics"],
    "Digital Marketing": ["SEO", "Social Media Marketing", "Content Marketing", "Analytics"],
    "Cybersecurity": ["Network Security", "Ethical Hacking", "Cloud Security", "Risk & Compliance"],
    "Personal Development": ["Public Speaking", "Time Management", "Communication", "Leadership"],
}


def generate_courses(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, i = [], 1
    for cat, topics in _TOPICS.items():
        for lvl in LEVELS:
            for t in topics:
                free = rng.random() < 0.35
                rows.append({
                    "CourseID": f"C{i:04d}", "CourseName": f"{t} – {lvl}",
                    "CourseCategory": cat, "CourseType": "Free" if free else "Paid",
                    "CourseLevel": lvl,
                    "CourseRating": round(float(np.clip(rng.normal(4.2, 0.35), 3.0, 5.0)), 1),
                    "CoursePrice": 0.0 if free else float(round(rng.uniform(499, 4999), -1)),
                })
                i += 1
    return pd.DataFrame(rows)


def generate_transactions(users: pd.DataFrame, courses: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Latent-archetype simulator: casual / explorer / specialist / career."""
    rng = np.random.default_rng(seed)
    price = courses.get("CoursePrice", pd.Series(0.0, index=courses.index)).fillna(0).values
    cat = courses.CourseCategory.values
    cats = sorted(set(cat))
    lvl = courses.LevelScore.values
    paid = (courses.CourseType.str.lower() != "free").values
    rating = courses.CourseRating.values
    pop = rng.lognormal(0, 0.5, len(courses))
    career_cats = [c for c in ["Business", "Finance", "Data Science", "Cybersecurity"] if c in cats] or cats[:2]
    rows, tid = [], 1
    for uid, age in zip(users.UserID, users.Age):
        if rng.random() < 0.04:
            continue
        p_career = 0.28 if age >= 25 else 0.14
        arche = rng.choice(["casual", "explorer", "specialist", "career"],
                           p=np.array([0.25, 0.25, 0.30, p_career - 0.0]) / (0.8 + p_career))
        w = pop * (rating / 4.2) ** 2
        if arche == "casual":
            n = rng.integers(1, 3); w = w * np.select([lvl == 1, lvl == 2], [4, 1], 0.3) * np.where(paid, 0.6, 1.4)
            gap = 60
        elif arche == "explorer":
            n = rng.integers(4, 10); w = w * np.select([lvl == 1, lvl == 2], [5, 2], 0.4) * np.where(paid, 0.8, 1.2)
            gap = 25
        elif arche == "specialist":
            n = rng.integers(4, 10); home = rng.choice(cats)
            w = w * np.where(cat == home, 12, 1) * np.select([lvl == 1, lvl == 2], [0.4, 1.5], 1.5)
            gap = 30
        else:
            n = rng.integers(3, 7); fav = rng.choice(career_cats, size=2, replace=False)
            w = w * np.where(np.isin(cat, fav), 6, 1) * np.select([lvl == 1, lvl == 2], [0.3, 1.2], 1.8) \
                * np.where(paid, 4, 0.3) * (rating / 4.2) ** 4
            gap = 45
        w = w.astype(float).copy()
        date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 600)))
        chosen = []
        for _ in range(int(n)):
            pr = w / w.sum()
            j = rng.choice(len(w), p=pr)
            chosen.append(j)
            w[j] = 0
            if arche == "explorer":
                w[cat == cat[j]] *= 0.3
            if w.sum() <= 0:
                break
            date = date + pd.Timedelta(days=int(rng.exponential(gap)) + 1)
            rows.append({"TransactionID": f"T{tid:06d}", "UserID": uid, "CourseID": courses.CourseID[j],
                         "TransactionDate": date,
                         "Amount": float(round(price[j] * (1 - rng.uniform(0, 0.2)), 2)) if paid[j] else 0.0})
            tid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING (learner-level aggregation)
# --------------------------------------------------------------------------- #
NUM_FEATURES = ["total_courses", "avg_courses_per_category", "enroll_frequency", "avg_spend",
                "avg_rating", "n_categories", "learning_depth_index", "paid_share", "Age"]
LOG_FEATURES = ["total_courses", "avg_courses_per_category", "enroll_frequency", "avg_spend",
                "learning_depth_index"]
FEATURE_LABELS = {
    "total_courses": "Total courses enrolled", "avg_courses_per_category": "Avg courses / category",
    "enroll_frequency": "Enrollments / month", "avg_spend": "Avg spend / course",
    "avg_rating": "Avg rating enrolled", "n_categories": "Diversity (categories)",
    "learning_depth_index": "Learning depth index", "paid_share": "Paid-course share", "Age": "Age",
}


def build_features(users: pd.DataFrame, courses: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    cols = ["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "LevelScore"]
    t = tx.merge(courses[cols], on="CourseID", how="inner")
    t["is_paid"] = (t.CourseType.astype(str).str.lower() != "free").astype(int)
    g = t.groupby("UserID")
    f = pd.DataFrame({
        "total_courses": g.CourseID.nunique(),
        "n_categories": g.CourseCategory.nunique(),
        "total_spend": g.Amount.sum(),
        "avg_spend": g.Amount.mean(),
        "avg_rating": g.CourseRating.mean(),
        "avg_level": g.LevelScore.mean(),
        "paid_share": g.is_paid.mean(),
        "first_date": g.TransactionDate.min(),
        "last_date": g.TransactionDate.max(),
    })
    f["avg_courses_per_category"] = f.total_courses / f.n_categories
    months = ((f.last_date - f.first_date).dt.days / 30.4).clip(lower=1.0)
    f["enroll_frequency"] = f.total_courses / months
    lv = pd.crosstab(t.UserID, t.LevelScore).reindex(columns=[1, 2, 3], fill_value=0)
    f["n_beginner"], f["n_intermediate"], f["n_advanced"] = lv[1], lv[2], lv[3]
    f["beginner_ratio"] = f.n_beginner / f.total_courses
    f["advanced_ratio"] = f.n_advanced / f.total_courses
    f["learning_depth_index"] = (f.n_advanced + 1) / (f.n_beginner + 1)  # >1 => advanced-leaning

    def _mode(s):
        vc = s.value_counts()
        return vc.index[0]
    f["preferred_category"] = g.CourseCategory.agg(_mode)
    f["preferred_level"] = lv.idxmax(axis=1).map({1: "Beginner", 2: "Intermediate", 3: "Advanced"})

    f = users.set_index("UserID")[["UserName", "Age", "Gender"]].join(f, how="left")
    zero = ["total_courses", "n_categories", "total_spend", "avg_spend", "paid_share", "enroll_frequency",
            "avg_courses_per_category", "n_beginner", "n_intermediate", "n_advanced",
            "beginner_ratio", "advanced_ratio"]
    f[zero] = f[zero].fillna(0)
    f["learning_depth_index"] = f["learning_depth_index"].fillna(1.0)
    f["avg_rating"] = f["avg_rating"].fillna(f["avg_rating"].median())
    f["avg_level"] = f["avg_level"].fillna(1.0)
    f["preferred_category"] = f["preferred_category"].fillna("None")
    f["preferred_level"] = f["preferred_level"].fillna("None")
    f["total_courses"] = f["total_courses"].astype(int)
    return f


# --------------------------------------------------------------------------- #
# 3. PREPROCESSING
# --------------------------------------------------------------------------- #
def prepare(feat: pd.DataFrame, min_enroll: int = 2, clip: float = 3.5):
    """Log-transform skewed features, standardise, one-hot categoricals.
    Sparse learners (< min_enroll) are excluded from *fitting* (noise reduction)
    but still receive a segment via nearest centroid."""
    num = feat[NUM_FEATURES].astype(float).copy()
    for c in LOG_FEATURES:
        num[c] = np.log1p(num[c])
    active = (feat.total_courses >= min_enroll).values
    if active.sum() < 50:
        active = (feat.total_courses >= 1).values
    scaler = StandardScaler().fit(num[active])
    Z = np.clip(scaler.transform(num), -clip, clip)
    parts = [Z]
    for col, w in [("Gender", 0.4), ("preferred_level", 0.6), ("preferred_category", 0.5)]:
        parts.append(pd.get_dummies(feat[col].astype(str)).astype(float).values * w)
    return np.hstack(parts), active


def sweep_k(Xa: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xa)
        sil = silhouette_score(Xa, km.labels_, sample_size=min(3000, len(Xa)), random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def choose_k(sweep: pd.DataFrame, k_min: int = 3) -> int:
    s = sweep[sweep.k >= k_min]
    return int(s.loc[s.silhouette.idxmax(), "k"])


def fit_kmeans(X: np.ndarray, active: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X[active])
    return km, km.predict(X)


def hierarchical_validation(Xa: np.ndarray, km_labels: np.ndarray, k: int, max_n: int = 1500) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xa), min(max_n, len(Xa)), replace=False)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xa[idx])
    return {"hier_silhouette": float(silhouette_score(Xa[idx], hc)),
            "kmeans_silhouette_same_sample": float(silhouette_score(Xa[idx], km_labels[idx])),
            "adjusted_rand_index": float(adjusted_rand_score(km_labels[idx], hc)), "sample_size": len(idx)}


# --------------------------------------------------------------------------- #
# 4. SEGMENT NAMING & PROFILING
# --------------------------------------------------------------------------- #
_ARCHETYPES = {
    "Career-Focused Achievers": {"avg_spend": 1, "paid_share": 1, "avg_rating": 0.7, "learning_depth_index": 0.5},
    "Deep Specialists": {"learning_depth_index": 1, "avg_courses_per_category": 1.2, "n_categories": -1},
    "Beginner Explorers": {"n_categories": 1.2, "learning_depth_index": -1, "total_courses": 0.4},
    "Casual Browsers": {"total_courses": -1.2, "enroll_frequency": -0.5, "n_categories": -0.5},
    "Engaged Power Learners": {"total_courses": 1, "enroll_frequency": 1, "n_categories": 0.5},
}


def name_segments(feat: pd.DataFrame, labels: np.ndarray) -> dict:
    df = feat.assign(_c=labels)
    prof = df.groupby("_c")[NUM_FEATURES].mean()
    z = (prof - feat[NUM_FEATURES].mean()) / feat[NUM_FEATURES].std().replace(0, 1)
    ks = list(z.index)
    score = pd.DataFrame({a: [sum(w * z.loc[c, f] for f, w in ws.items()) for c in ks]
                          for a, ws in _ARCHETYPES.items()}, index=ks)
    names, s = {}, score.copy()
    while len(names) < min(len(ks), len(_ARCHETYPES)):
        c, a = np.unravel_index(np.argmax(s.values), s.shape)
        names[s.index[c]] = s.columns[a]
        s.iloc[c, :] = -1e9
        s.iloc[:, a] = -1e9
    for c in ks:
        if c not in names:
            top = z.loc[c].abs().idxmax()
            names[c] = f"Mixed Segment {c} ({'high' if z.loc[c, top] > 0 else 'low'} {FEATURE_LABELS[top].lower()})"
    return names


def segment_profile(feat: pd.DataFrame) -> pd.DataFrame:
    cols = NUM_FEATURES + ["beginner_ratio", "advanced_ratio", "total_spend"]
    p = feat.groupby("Segment")[cols].mean()
    p.insert(0, "Learners", feat.groupby("Segment").size())
    return p


def intra_cluster_similarity(X: np.ndarray, labels: np.ndarray) -> tuple[pd.Series, float]:
    """Mean pairwise cosine similarity within each cluster vs. over all learners."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    def mean_pair(A):
        n = len(A)
        return float((np.linalg.norm(A.sum(0)) ** 2 - n) / (n * (n - 1))) if n > 1 else np.nan
    per = pd.Series({c: mean_pair(Xn[labels == c]) for c in np.unique(labels)})
    return per, mean_pair(Xn)


# --------------------------------------------------------------------------- #
# 5. RECOMMENDER (content + cluster popularity + similar learners + rating)
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {"content": 0.40, "cluster_popularity": 0.30, "similar_learners": 0.30}


def _mm(a: np.ndarray) -> np.ndarray:
    m = a.max()
    return a / m if m > 0 else a


class Recommender:
    def __init__(self, courses, tx, feat, X, labels, seg_names, k_neighbors=30):
        self.courses = courses.reset_index(drop=True)
        self.feat, self.labels, self.seg_names = feat, np.asarray(labels), seg_names
        self.uid_idx = {u: i for i, u in enumerate(feat.index)}
        self.cid_idx = {c: i for i, c in enumerate(self.courses.CourseID)}
        M = np.zeros((len(feat), len(self.courses)), dtype=np.float32)
        for u, c in zip(tx.UserID.map(self.uid_idx), tx.CourseID.map(self.cid_idx)):
            if not (np.isnan(u) or np.isnan(c)):
                M[int(u), int(c)] = 1
        self.M = M
        self.Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        self.k_nb = k_neighbors
        cat_codes, self.cat_names = pd.factorize(self.courses.CourseCategory)
        self.c_cat, self.n_cat = cat_codes, len(self.cat_names)
        self.c_lvl = self.courses.LevelScore.values.astype(float)
        r = self.courses.CourseRating.values.astype(float)
        self.rating_n = (r - r.min()) / (r.max() - r.min() + 1e-9)
        self.global_pop = _mm(M.sum(0))
        self.cluster_pop = {}
        for c in np.unique(self.labels):
            self.cluster_pop[c] = _mm(M[self.labels == c].sum(0))

    # component scores ------------------------------------------------------
    def components(self, u: int) -> dict:
        owned = self.M[u] > 0
        if owned.any():
            cc = np.bincount(self.c_cat[owned], minlength=self.n_cat).astype(float)
            aff = (cc + 0.1) / (cc.sum() + 0.1 * self.n_cat)
            cat_aff = _mm(aff[self.c_cat])
            target = min(3.0, self.c_lvl[owned].mean() + 0.3)   # nudge toward next level
        else:                                                     # cold start
            cat_aff, target = np.ones(len(self.courses)), 1.0
        lvl_match = np.exp(-((self.c_lvl - target) ** 2) / (2 * 0.8 ** 2))
        content = 0.65 * cat_aff + 0.35 * lvl_match
        pop = self.cluster_pop[self.labels[u]]
        mask = self.labels == self.labels[u]
        mask[u] = False
        idx = np.where(mask)[0]
        sim = np.zeros(len(self.courses))
        if len(idx):
            s = self.Xn[idx] @ self.Xn[u]
            top = np.argsort(-s)[: self.k_nb]
            w = np.clip(s[top], 0, None)
            if w.sum() > 0:
                sim = _mm((w @ self.M[idx[top]]) / w.sum())
        return {"content": content, "cluster_popularity": pop, "similar_learners": sim, "owned": owned}

    def score(self, u: int, weights=None) -> np.ndarray:
        w = weights or DEFAULT_WEIGHTS
        c = self.components(u)
        tot = sum(w.values()) or 1
        rel = sum(w[k] * c[k] for k in ("content", "cluster_popularity", "similar_learners")) / tot
        return rel * (0.75 + 0.25 * self.rating_n)                # rating-weighted relevance

    def recommend(self, user_id, n=10, levels=None, categories=None, weights=None,
                  exclude_owned=True) -> pd.DataFrame:
        u = self.uid_idx[user_id]
        w = weights or DEFAULT_WEIGHTS
        comp = self.components(u)
        tot = sum(w.values()) or 1
        rel = sum(w[k] * comp[k] for k in ("content", "cluster_popularity", "similar_learners")) / tot
        final = rel * (0.75 + 0.25 * self.rating_n)
        ok = np.ones(len(final), bool)
        if exclude_owned:
            ok &= ~comp["owned"]
        if levels:
            ok &= self.courses.CourseLevel.isin(levels).values
        if categories:
            ok &= self.courses.CourseCategory.isin(categories).values
        order = [i for i in np.argsort(-final) if ok[i]][:n]
        seg = self.seg_names[self.labels[u]]
        pref = self.feat.iloc[u].preferred_category
        rows = []
        for i in order:
            drivers = {"content": comp["content"][i] * w["content"],
                       "cluster_popularity": comp["cluster_popularity"][i] * w["cluster_popularity"],
                       "similar_learners": comp["similar_learners"][i] * w["similar_learners"]}
            top = max(drivers, key=drivers.get)
            reason = {"content": f"Matches your interest in {self.courses.CourseCategory[i]}"
                                 + (" (your top category)" if self.courses.CourseCategory[i] == pref else ""),
                      "cluster_popularity": f"Popular among {seg}",
                      "similar_learners": "Taken by learners with a similar profile"}[top]
            rows.append({"CourseID": self.courses.CourseID[i], "Course": self.courses.CourseName[i],
                         "Category": self.courses.CourseCategory[i], "Level": self.courses.CourseLevel[i],
                         "Type": self.courses.CourseType[i], "Rating": self.courses.CourseRating[i],
                         "Score": round(float(final[i]), 4), "Content": round(float(comp["content"][i]), 3),
                         "ClusterPop": round(float(comp["cluster_popularity"][i]), 3),
                         "SimilarLearners": round(float(comp["similar_learners"][i]), 3),
                         "Why": reason})
        return pd.DataFrame(rows)


def learning_path(recs: pd.DataFrame, steps: int = 5) -> pd.DataFrame:
    """Order the top recommendations into a beginner -> advanced journey."""
    if recs.empty:
        return recs
    r = recs.head(max(steps * 2, steps)).copy()
    r["_lv"] = r.Level.map(level_score)
    r = r.sort_values(["_lv", "Score"], ascending=[True, False]).head(steps).drop(columns="_lv")
    r.insert(0, "Step", range(1, len(r) + 1))
    return r


# --------------------------------------------------------------------------- #
# 6. FULL PIPELINE
# --------------------------------------------------------------------------- #
def run_pipeline(users, courses, tx, k=None, min_enroll=2, ks=range(2, 9)) -> dict:
    feat = build_features(users, courses, tx)
    X, active = prepare(feat, min_enroll)
    sweep = sweep_k(X[active], ks)
    k = int(k) if k else choose_k(sweep)
    km, labels = fit_kmeans(X, active, k)
    names = name_segments(feat, labels)
    feat = feat.copy()
    feat["Cluster"] = labels
    feat["Segment"] = feat.Cluster.map(names)
    feat["low_activity"] = ~active
    hier = hierarchical_validation(X[active], labels[active], k)
    sil = silhouette_score(X[active], labels[active], sample_size=min(3000, int(active.sum())),
                           random_state=RANDOM_STATE)
    intra, overall = intra_cluster_similarity(X[active], labels[active])
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    pc = pca.transform(X)
    feat["PC1"], feat["PC2"] = pc[:, 0], pc[:, 1]
    rec = Recommender(courses, tx, feat, X, labels, names)
    return {"feat": feat, "X": X, "active": active, "k": k, "sweep": sweep, "km": km, "names": names,
            "silhouette": float(sil), "hier": hier, "intra": intra.rename(index=names),
            "intra_overall": overall, "pca_var": pca.explained_variance_ratio_, "recommender": rec}


# --------------------------------------------------------------------------- #
# 7. EVALUATION (leave-last-out hold-out)
# --------------------------------------------------------------------------- #
def evaluate(users, courses, tx, k, min_enroll=2, min_hist=3, Ks=(5, 10), max_users=1500) -> pd.DataFrame:
    """Hide each eligible learner's latest enrolment, fit everything on the rest,
    and test whether it appears in the top-K. Precision@K = hit/K (one held-out item)."""
    t = tx.sort_values("TransactionDate")
    cnt = t.groupby("UserID").CourseID.transform("size")
    elig = t[cnt >= min_hist]
    held = elig.groupby("UserID").tail(1)
    if len(held) > max_users:
        held = held.sample(max_users, random_state=RANDOM_STATE)
    train = t.drop(index=held.index)
    feat = build_features(users, courses, train)
    X, active = prepare(feat, min_enroll)
    km, labels = fit_kmeans(X, active, k)
    names = name_segments(feat, labels)
    rec = Recommender(courses, train, feat, X, labels, names)
    ranks = {"Random": [], "Global popularity": [], "Cluster popularity": [], "Hybrid (cluster-aware)": []}
    n_unowned = []
    rng = np.random.default_rng(RANDOM_STATE)
    for uid, cid in zip(held.UserID, held.CourseID):
        u, h = rec.uid_idx[uid], rec.cid_idx[cid]
        comp = rec.components(u)
        free = ~comp["owned"]
        n_unowned.append(free.sum())
        for name, sc in [("Global popularity", rec.global_pop + 1e-6 * rng.random(len(rec.global_pop))),
                         ("Cluster popularity", comp["cluster_popularity"] + 1e-6 * rng.random(len(rec.global_pop))),
                         ("Hybrid (cluster-aware)", rec.score(u))]:
            s = sc[free]
            ranks[name].append(int((s >= sc[h]).sum()))
    out = []
    n_un = np.array(n_unowned)
    for name in ranks:
        row = {"Model": name}
        for K in Ks:
            if name == "Random":
                hit = float(np.mean(np.minimum(K, n_un) / n_un))
            else:
                hit = float(np.mean(np.array(ranks[name]) <= K))
            row[f"HitRate@{K}"] = hit
            row[f"Precision@{K}"] = hit / K
        out.append(row)
    res = pd.DataFrame(out)
    base = res.loc[res.Model == "Global popularity"]
    for K in Ks:
        b = float(base[f"HitRate@{K}"].iloc[0])
        res[f"EngagementLift@{K} (%)"] = (res[f"HitRate@{K}"] / b - 1) * 100 if b > 0 else np.nan
    res.attrs["n_eval_users"] = len(held)
    return res
