# ==== Asset Recommender (Jupyter) ====
# Alias mining = sama dengan aturan 2025 (support/lift/bigram/substring + filter generic)
# Scoring rekomendasi = alias/canonical (word-boundary + nospace>=5) + token overlap + TF-IDF cosine

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# =========================
# KONFIGURASI
# =========================
DATA_XLSX   = "train data 10 sept 2025 - after cleaning.xlsx"   # ganti sesuai file kamu
TRAIN_YEARS = {2016,2017,2018,2019,2020,2021,2022,2023,2024,2025}
# TRAIN_YEARS = {2023,2024}
TOP_K       = 5

# Bobot skor gabungan (rekomendasi)
W_ALIAS     = 1.00   # alias / canonical substring (kuat)
W_TOKEN_JAC = 0.50   # token Jaccard (overlap per kata bermakna)
W_COSINE    = 0.60   # TF-IDF cosine (char 3–5)




# Stopword & token rules
STOP = {
    "sprint","project","phase","module","system","service","services",
    "apps","app","application","data","report","mobile","api","dashboard",
    "update","enhancement","support","feature","features","tools","tool",
    "test","revamp","change","bug","fix","task","portal","menu","screen",
    "form","forms","page","pages",
    "dan","yang","untuk","pada","di","ke","dari","dengan","atas",
    "sebagai","agar","oleh","dll","dalam",
    "penambahan","perubahan","setup","status","scope","all","order","orders",
    "invoice","invoices","field","fields","push","new","existing","pdf","file",
    "upload","download","export","import","laporan","edit","hapus","tambah","validasi"
}
KEEP_SHORT = {"crm","hcm","rv","fgc","b2b","sso","mfa","apex","imove","idm","ufi","fims","fms","visum"}

# Alias default (seed)
DEFAULT_ALIASES = {
    "strategyone": {"strategy one","strategyone"},
    "avocado": {"b2b","port-f","port f"},
    "port f": {"b2b","port-f","portf"},
    "fifgroup card 2.0": {"fgc","fif card","fifgroupcard"},
    "web fifgroup": {"corporate website","corporate website sprint","website fifgroup"},
    "fims (fiduciary management system)": {"fims","fiduciary management system"},
    "web eform": {"eform"},
    "imove": {"imove","i-move","imove mobile"},
    "action": {"action","actionrest","actionreport","actionmrest"},
    "fmsales": {"fm sales","fmsales","visum","ufi"},
    "microfinancing microapps - fincor": {"microfinancing"},
    "e-Recruitment":{"RISE"},
    "digital leads":{"DBLM"},
    "dealer information system":{"DIS"}
}

# ====== PARAM MINING (DISAMAKAN DENGAN ATURAN 2025) ======
MIN_LEN_TOKEN        = 3
MIN_SUPPORT_ASSET    = 2
MIN_LIFT             = 2.5
MAX_ASSETS_PER_CAND  = 2
MAKE_BIGRAMS         = True
MAKE_TRIGRAMS        = False
ENABLE_SUBSTRING     = True

DROP_PATTERNS = [
    r"\bpenambahan\s+field\b", r"\ball\s+status\b", r"\ball\s+push\b",
    r"\btracking\s+order\b", r"\bpenambahan\s+\w+\b", r"\bupdate\s+\w+\b"
]
DROP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DROP_PATTERNS]

# =========================
# Helpers teks
# =========================
def clean_soft(s: str) -> str:
    if pd.isna(s): return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

def nospace(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())

# token utk SKOR (rekomendasi)
def tokens(text: str) -> list:
    toks = [t for t in clean_soft(text).split() if t]
    out = []
    for t in toks:
        if (len(t) >= 4 and t not in STOP) or (t in KEEP_SHORT):
            out.append(t)
    return out

#tambahan kode baru
def load_asset_register(path: str, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    cols = list(df.columns)

    # kolom pertama = nama aset
    name_col = cols[0]

    if len(cols) >= 4:
        desc_col = cols[3]   # kolom D (index 3)
        df = df[[name_col, desc_col]].rename(
            columns={name_col: "ASSET_NAME_REG", desc_col: "ASSET_DESC_REG"}
        )
    else:
        df = df[[name_col]].rename(columns={name_col: "ASSET_NAME_REG"})
        df["ASSET_DESC_REG"] = ""   # fallback kosong

    df["ASSET_KEY_REG"] = df["ASSET_NAME_REG"].apply(clean_soft)
    df["ASSET_DESC_CLEAN"] = df["ASSET_DESC_REG"].fillna("").apply(clean_soft)
    return df


asset_register = load_asset_register("SRHS - Asset Register.xlsx")

# buat vektor TF-IDF deskripsi aset resmi
vect_reg = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)
M_reg = vect_reg.fit_transform(asset_register["ASSET_DESC_CLEAN"])

# token utk MINING (ikut aturan 2025)
def tokens_mine(text: str) -> list:
    toks = [t for t in clean_soft(text).split() if t]
    out = []
    for t in toks:
        if (len(t) >= MIN_LEN_TOKEN and t not in STOP) or (t in KEEP_SHORT):
            out.append(t)
    return out

def tokens_raw(text: str) -> list:
    return [t for t in clean_soft(text).split() if t]

def ngrams(tokens_list: list, n: int) -> list:
    return [" ".join(tokens_list[i:i+n]) for i in range(len(tokens_list)-n+1)] if len(tokens_list) >= n else []

def is_generic_phrase(s: str) -> bool:
    if not s: return True
    st = clean_soft(s)
    for rx in DROP_PATTERNS:
        if rx.search(st): return True
    toks = tokens_raw(st)
    if toks and all((t in STOP) for t in toks): return True
    if len(toks) == 1 and toks[0] in STOP: return True
    return False

def norm_key_asset(s: str) -> str:
    return clean_soft(s)

# =========================
# Load & normalisasi workbook
# =========================
def read_workbook(path: str) -> dict:
    xls = pd.read_excel(path, sheet_name=None)
    out = {}
    for name, df in xls.items():
        ren = {}
        for c in df.columns:
            u = c.strip().upper()
            if u.startswith("TICKET_TIT"): ren[c] = "TICKET_TITLE"
            elif u.startswith("TICKET_DE"): ren[c] = "TICKET_DESCRIPTION"
            elif u in {"ASSET_NA","ASSET_NAME","ASSET"}: ren[c] = "ASSET_NAME"
            elif u == "YEAR": ren[c] = "YEAR"
            elif u == "RESULT": ren[c] = "RESULT"
            elif u == "TICKET_ID": ren[c] = "TICKET_ID"
            elif u in {"CANONICAL_ASSET","ALIAS"}: ren[c] = u
        if ren: df = df.rename(columns=ren)
        out[name] = df
    return out

def combine_main_sheets(xls: dict) -> pd.DataFrame:
    frames = []
    for name, df in xls.items():
        up = set(df.columns.map(str).str.upper())
        if {"CANONICAL_ASSET","ALIAS"}.issubset(up):   # skip sheet alias
            continue
        need = [c for c in ["TICKET_ID","TICKET_TITLE","TICKET_DESCRIPTION","ASSET_NAME","YEAR","RESULT"] if c in df.columns]
        if not need: continue
        d = df[need].copy()
        for c in ["TICKET_TITLE","TICKET_DESCRIPTION","ASSET_NAME"]:
            if c not in d.columns: d[c] = ""
        frames.append(d)
    if not frames:
        raise RuntimeError("Tidak ada sheet data valid di file.")
    data = pd.concat(frames, ignore_index=True)
    data["YEAR_NUM"] = pd.to_numeric(data["YEAR"], errors="coerce")
    return data

def load_alias_sheet(xls: dict) -> dict:
    alias_map = {}
    A = xls.get("ALIASES")
    if A is not None:
        up = set(A.columns.str.upper())
        if {"CANONICAL_ASSET","ALIAS"}.issubset(up):
            for _, r in A.dropna(subset=["CANONICAL_ASSET","ALIAS"]).iterrows():
                alias_map.setdefault(clean_soft(r["CANONICAL_ASSET"]), set()).add(str(r["ALIAS"]))
    return alias_map

# =========================
# BUILD ALIAS MAP (versi 2025)
# =========================
def build_alias_map(xls: dict, train_df: pd.DataFrame) -> dict:
    """
    Alias map = ALIASES sheet (jika ada) + DEFAULT_ALIASES + mining berbobot
    (support/lift/bigram/substring) dari histori TRAIN_YEARS.
    """
    # 1) alias dari sheet
    alias_map = load_alias_sheet(xls)

    # 2) default aliases
    for k, v in DEFAULT_ALIASES.items():
        alias_map.setdefault(clean_soft(k), set()).update(v)

    # 3) mining 2025-like
    df = train_df[train_df["ASSET_NAME"].astype(str).str.strip() != ""].copy()
    df["ASSET_KEY"] = df["ASSET_NAME"].apply(norm_key_asset)

    # display name terpopuler per key
    disp_counter = defaultdict(lambda: defaultdict(int))
    for _, r in df.iterrows():
        disp_counter[r["ASSET_KEY"]][str(r["ASSET_NAME"]).strip()] += 1

    def display_name_for(key: str) -> str:
        items = sorted(disp_counter[key].items(), key=lambda kv: (-kv[1], kv[0].lower()))
        return items[0][0] if items else key

    asset_ticket_counts = defaultdict(int)
    cand_any_count     = defaultdict(int)
    cand_assets        = defaultdict(set)
    cand_count_by_pair = defaultdict(int)
    cand_type          = {}

    for _, row in df.iterrows():
        title = str(row.get("TICKET_TITLE", ""))
        desc  = str(row.get("TICKET_DESCRIPTION", ""))
        asset = str(row.get("ASSET_NAME", ""))
        akey  = row["ASSET_KEY"]

        asset_ticket_counts[akey] += 1
        raw = f"{title} {desc}"

        # kandidat dari token & n-gram
        tks   = tokens_mine(raw)
        cands = set()
        for t in tks:
            cands.add(("token", t))
        if MAKE_BIGRAMS:
            for g in ngrams(tks, 2): cands.add(("bigram", g))
        if MAKE_TRIGRAMS:
            for g in ngrams(tks, 3): cands.add(("trigram", g))

        # kandidat substring dengan nama aset
        if ENABLE_SUBSTRING:
            asset_ns = nospace(asset)
            if len(asset_ns) >= 4:
                for t in set(tokens_raw(raw)):
                    tns = nospace(t)
                    if len(tns) >= 4 and (tns in asset_ns or asset_ns in tns):
                        if not is_generic_phrase(t):
                            cands.add(("substr", t))

        seen = set()
        for ctype, cand in cands:
            if (ctype, cand) in seen: 
                continue
            seen.add((ctype, cand))
            cand_any_count[cand]      += 1
            cand_assets[cand].add(akey)
            cand_count_by_pair[(akey, cand)] += 1
            if (akey, cand) not in cand_type:
                cand_type[(akey, cand)] = ctype

    # scoring & filter sehingga alias spesifik ke aset
    rows = []
    N = len(df)
    for (akey, cand), supp_asset in cand_count_by_pair.items():
        if is_generic_phrase(cand):
            continue
        n_asset  = asset_ticket_counts[akey]
        supp_all = cand_any_count[cand]
        assets_for_cand = len(cand_assets[cand])

        p_asset  = supp_asset / max(1, n_asset)
        p_global = supp_all  / max(1, N)
        lift     = p_asset / max(p_global, 1e-9)

        disp     = display_name_for(akey)
        asset_ns = nospace(disp)
        cand_ns  = nospace(cand)
        is_sub   = ENABLE_SUBSTRING and len(cand_ns) >= 4 and (cand_ns in asset_ns or asset_ns in cand_ns)

        # buang "asset + kata generik"
        cand_tokens_raw  = set(tokens_raw(cand))
        asset_tokens_raw = set(tokens_raw(disp))
        extra = [t for t in cand_tokens_raw if t not in asset_tokens_raw]
        if is_sub and extra and all((t in STOP) for t in extra):
            continue

        keep = True
        if assets_for_cand > MAX_ASSETS_PER_CAND: keep = False
        if supp_asset < MIN_SUPPORT_ASSET and cand not in KEEP_SHORT: keep = False
        if lift < MIN_LIFT and not (is_sub and supp_asset >= 1): keep = False
        if not keep: 
            continue

        rows.append((akey, cand))

    # merge hasil mining ke alias_map
    for akey, cand in rows:
        alias_map.setdefault(akey, set()).add(cand)

    return alias_map

# =========================
# BUILD ASSET INDEX (untuk rekomendasi)
# =========================
def build_asset_index(train_df: pd.DataFrame, alias_map: dict):
    tmp = train_df.dropna(subset=["ASSET_NAME"]).copy()
    tmp["K"] = tmp["ASSET_NAME"].apply(clean_soft)

    # display name terpopuler per key
    disp = {}
    for k, g in tmp.groupby("K"):
        disp_name = (g["ASSET_NAME"].astype(str).value_counts().idxmax()
                     if not g["ASSET_NAME"].empty else k)
        disp[k] = disp_name

    asset_docs, asset_keys, asset_tokens = [], [], {}
    for k, g in tmp.groupby("K"):
        pieces = [disp.get(k, k)]
        pieces += list(alias_map.get(k, set()))
        # contoh historis (dibatasi agar ringan)
        sample = " ".join(
            (g["TICKET_TITLE"].fillna("") + " " + g["TICKET_DESCRIPTION"].fillna("")).astype(str).tolist()[:300]
        )
        pieces.append(sample)
        doc = clean_soft(" ".join(pieces))
        asset_docs.append(doc)
        asset_keys.append(k)
        asset_tokens[k] = set(tokens(" ".join(pieces)))

    vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
    M = vect.fit_transform(asset_docs)

    display_by_key = {k: disp.get(k, k) for k in asset_keys}
    return {
        "vectorizer": vect,
        "matrix": M,
        "asset_keys": asset_keys,
        "asset_tokens": asset_tokens,
        "display_by_key": display_by_key,
        "alias_map": alias_map
    }

# =========================
# LOAD SEMUA & BUILD SEKALI
# =========================
def load_all_data(path: str):
    xls = read_workbook(path)
    data = combine_main_sheets(xls)

    data["ASSET_KEY"] = data["ASSET_NAME"].apply(clean_soft)

    train_df = data[data["YEAR_NUM"].isin(TRAIN_YEARS)].copy()
    alias_map = build_alias_map(xls, train_df)
    index = build_asset_index(train_df, alias_map)
    return data, index

@st.cache_resource
def get_index():
    data, index = load_all_data(DATA_XLSX)
    return data, index

data, index = get_index()
# print(f"Index siap • train rows: {len(data[data['YEAR_NUM'].isin(TRAIN_YEARS)])} • aset unik: {len(index['asset_keys'])}")

# =========================
# REKOMENDASI (scoring)
# =========================
def _word_in(text_soft: str, phrase_soft: str) -> bool:
    """Match frasa pada batas kata (word boundary)."""
    if not phrase_soft:
        return False
    pat = r"\b" + re.escape(phrase_soft) + r"\b"
    return re.search(pat, text_soft) is not None

def recommend(title: str, desc: str, top_k:int=TOP_K) -> pd.DataFrame:
    text = f"{title} {desc}"
    text_soft = clean_soft(text)
    text_ns   = nospace(text)
    t_set     = set(tokens(text))

    vect = index["vectorizer"]
    M    = index["matrix"]
    keys = index["asset_keys"]
    a_tok= index["asset_tokens"]
    disp = index["display_by_key"]
    amap = index["alias_map"]

    alias_score, alias_reason = {}, {}

    for k in keys:
        sc = 0.0
        rs = []

        # canonical nospace (pakai syarat panjang agar tidak nabrak)
        can_ns = nospace(disp[k])
        if len(can_ns) >= 5 and can_ns in text_ns:
            sc = max(sc, W_ALIAS); rs.append("canonical_substring")

        # alias (word-boundary + nospace>=5)
        for ali in amap.get(k, set()):
            ali_soft = clean_soft(ali)
            ali_ns   = nospace(ali)
            if ali_soft and _word_in(text_soft, ali_soft):
                sc = max(sc, W_ALIAS); rs.append(f"alias_wb:{ali}"); break
            if len(ali_ns) >= 5 and ali_ns in text_ns:
                sc = max(sc, W_ALIAS); rs.append(f"alias_ns:{ali}"); break

        # token overlap (Jaccard)
        inter = t_set & a_tok.get(k, set())
        if inter:
            jacc = len(inter) / max(1, len(t_set | a_tok[k]))
            sc = max(sc, W_TOKEN_JAC * jacc)
            if jacc >= 0.05:
                rs.append(f"token_overlap({len(inter)})")

        # synergy bonus kecil jika ≥2 sinyal
        if ("canonical_substring" in rs and any(r.startswith("alias_") for r in rs)) or \
           (any(r.startswith("alias_") for r in rs) and any(r.startswith("token_overlap") for r in rs)):
            sc += 0.05

        alias_score[k] = round(sc, 6)
        alias_reason[k] = "; ".join(rs)

    # cosine tf-idf
    q = vect.transform([clean_soft(text)])
    cos = cosine_similarity(q, M).ravel()

    # gabungkan + tie-break
    rows = []
    for i, k in enumerate(keys):
        s_alias  = alias_score[k]
        s_cosine = W_COSINE * float(cos[i])
        total    = max(s_alias, s_cosine)
        rows.append({
            "ASSET_SUGGEST": disp[k],
            "SCORE_TOTAL": round(total, 4),
            "SCORE_ALIAS": round(s_alias, 4),
            "SCORE_COSINE": round(s_cosine, 4),
            "WHY": alias_reason[k]
        })
    rows.sort(key=lambda r: (r["SCORE_TOTAL"], r["SCORE_ALIAS"], r["SCORE_COSINE"]), reverse=True)
    return pd.DataFrame(rows[:top_k])

# def recommend_with_history(title: str, desc: str, top_k:int=TOP_K, hist_k:int=3):
#     layer1 = recommend(title, desc, top_k)
#     results = {}

#     for aset in layer1["ASSET_SUGGEST"]:
#         aset_key = clean_soft(aset)

#         # cari alias juga
#         possible_keys = {aset_key}
#         possible_keys |= index["alias_map"].get(aset_key, set())

#         mask = False
#         for key in possible_keys:
#             key_clean = str(key).lower().strip()
#             mask = mask | data["ASSET_NAME"].str.lower().str.contains(key_clean, na=False)
#             # mask = data["ASSET_NAME"].str.lower().str.strip() == aset_key


#         hist = data[mask].copy()
#         if hist.empty:
#             continue

#         hist = hist[["TICKET_TITLE", "TICKET_DESCRIPTION", "ASSET_NAME", "RESULT"]].head(hist_k)
#         results[aset] = hist.reset_index(drop=True)

#     return {"layer1": layer1, "layer2": results}

# def recommend_with_history(title: str, desc: str, top_k:int=TOP_K, hist_k:int=3):
#     layer1 = recommend(title, desc, top_k)
#     results = {}

#     # pastikan kolom kunci ada
#     if "ASSET_KEY" not in data.columns:
#         data["ASSET_KEY"] = data["ASSET_NAME"].apply(clean_soft)

#     for aset in layer1["ASSET_SUGGEST"]:
#         k = clean_soft(aset)

#         # STRICT: hanya baris yang benar-benar milik aset ini
#         cols = ["TICKET_TITLE","TICKET_DESCRIPTION","ASSET_NAME","RESULT"]
#         hist = data.loc[data["ASSET_KEY"] == k, cols].head(hist_k)

#         if not hist.empty:
#             results[aset] = hist.reset_index(drop=True)

#     return {"layer1": layer1, "layer2": results}

# from rapidfuzz import fuzz, process

from fuzzywuzzy import fuzz

def recommend_with_register_and_fuzzy(title: str, desc: str, top_k:int=TOP_K, hist_k:int=5):
    layer1 = recommend(title, desc, top_k)
    results = {}

    if "ASSET_KEY" not in data.columns:
        data["ASSET_KEY"] = data["ASSET_NAME"].apply(clean_soft)

    text = clean_soft(f"{title} {desc}")

    for aset in layer1["ASSET_SUGGEST"]:
        k = clean_soft(aset)
        aset_hist = data.loc[data["ASSET_KEY"] == k, ["TICKET_TITLE","TICKET_DESCRIPTION","ASSET_NAME","RESULT"]]

        if aset_hist.empty:
            continue

        # fuzzy: cosine antara input dan setiap tiket historis
        vect_tmp = TfidfVectorizer().fit(aset_hist["TICKET_DESCRIPTION"].fillna("").astype(str))
        mat_hist = vect_tmp.transform(aset_hist["TICKET_DESCRIPTION"].fillna("").astype(str))
        q_vec = vect_tmp.transform([text])
        cos_scores = cosine_similarity(q_vec, mat_hist).ravel()

        aset_hist = aset_hist.copy()
        aset_hist["FUZZY_SCORE"] = cos_scores

        # historis relevan = skor fuzzy > threshold
        relevan = aset_hist[aset_hist["FUZZY_SCORE"] > 0.1].sort_values("FUZZY_SCORE", ascending=False).head(hist_k)
        lain = aset_hist.drop(relevan.index).head(hist_k)

        results[aset] = {"fuzzy": relevan.reset_index(drop=True),
                         "all": lain.reset_index(drop=True)}

    return {"layer1": layer1, "layer2": results}

# =========================
# Helper pemakaian
# =========================
def suggest(title: str, desc: str, top_k:int=TOP_K):
    """Kembalikan top-N saran aset (DataFrame)"""
    return recommend(title, desc, top_k)

def ask(top_k:int=TOP_K):
    """Mode interaktif di notebook"""
    print("Ketik judul & deskripsi. Tekan ENTER pada judul untuk selesai.\n")
    while True:
        title = input("Title : ").strip()
        if title == "":
            print("Selesai."); break
        desc  = input("Desc  : ").strip()
        out = recommend(title, desc, top_k)
        display(out)
        print("-"*60)

# Contoh pakai:
# ask()
# suggest("SSO FM Verification (I-Move Mobile)",
#         "Adanya kebutuhan implementasi SSO - MFA pada aplikasi Imove Mobile",
#         top_k=5)

# =========================
# Streamlit UI
# =========================


st.title("🔎 Asset Recommender (IT P&G)")

title = st.text_input("Masukkan Judul Ticket/Contract")
desc = st.text_area("Masukkan Deskripsi Ticket/Contract")
top_k = st.slider("Jumlah rekomendasi aset", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if title.strip() == "" and desc.strip() == "":
        st.warning("Mohon masukkan judul atau deskripsi.")
    else:
        out = recommend_with_register_and_fuzzy(title, desc, top_k, hist_k=5)

        st.write("### 🔮 Layer 1: Rekomendasi Aset")
        st.dataframe(out["layer1"])

        st.write("### 📜 Layer 2: Historis Tiap Aset")
        for aset, hist in out["layer2"].items():
            with st.expander(f"Detail historis untuk aset: {aset}"):
                st.write("**Historis Relevan (Fuzzy Match):**")
                st.dataframe(hist["fuzzy"])

                st.write("**Historis Lain Aset Ini:**")
                st.dataframe(hist["all"])
