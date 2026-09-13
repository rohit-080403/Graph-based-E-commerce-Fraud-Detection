"""
Phase 6 (fixed): 
1. Correctly maps each transaction to ONLY the specific card/device it used
   (previous version incorrectly connected every neighbor to every entity).
2. Caps neighbor count for high-degree ("supernode") cards/devices, since
   a card shared by 1000+ transactions provides diluted, low-value signal
   for a single prediction — mirrors the Address/Email exclusion from Phase 5.
"""
import numpy as np
import torch
import joblib
import pandas as pd
from torch_geometric.data import HeteroData

from src.graph.connection import get_driver

SAMPLE_PATH = "data/processed_sample.parquet"
EXCLUDE_COLS = ["TransactionID", "isFraud", "card_id", "address_id", "device_id", "email_domain"]
MAX_NEIGHBORS_PER_ENTITY = 50  # cap — entities with more neighbors than this are treated as supernodes

txn_scaler = joblib.load("src/api/txn_scaler.pkl")
card_scaler = joblib.load("src/api/card_scaler.pkl")
device_scaler = joblib.load("src/api/device_scaler.pkl")
label_encoders = joblib.load("src/api/label_encoders.pkl")

_df_cache = None


def get_transaction_features(transaction_id):
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet(SAMPLE_PATH).set_index("TransactionID")

    if transaction_id not in _df_cache.index:
        raise ValueError(f"Transaction {transaction_id} not found in processed sample")

    row = _df_cache.loc[transaction_id]
    feature_cols = [c for c in _df_cache.columns if c not in EXCLUDE_COLS]
    values = row[feature_cols].copy()

    for col, le in label_encoders.items():
        if col in values.index:
            raw_val = str(values[col]) if pd.notna(values[col]) else "missing"
            if raw_val in le.classes_:
                values[col] = le.transform([raw_val])[0]
            else:
                values[col] = 0  # unseen category fallback

    values = values.fillna(0).values.astype(float).reshape(1, -1)
    scaled = txn_scaler.transform(values)
    return torch.tensor(scaled, dtype=torch.float)


def fetch_live_subgraph(transaction_id):
    """
    Returns this transaction's card_id/device_id plus, for EACH separately,
    up to MAX_NEIGHBORS_PER_ENTITY other transactions sharing that entity.
    Capping here (in Cypher) avoids pulling 1000+ rows over the network
    just to discard most of them in Python.
    """
    driver = get_driver()
    query = """
    MATCH (t:Transaction {transaction_id: $txn_id})
    OPTIONAL MATCH (t)-[:USED_CARD]->(c:Card)
    OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)
    OPTIONAL MATCH (t)-[:FROM_ADDRESS]->(a:Address)
    OPTIONAL MATCH (t)-[:USED_EMAIL_DOMAIN]->(e:EmailDomain)
    WITH t, c, d, a, e
    OPTIONAL MATCH (c)<-[:USED_CARD]-(t2:Transaction)
    WHERE c IS NOT NULL
    WITH t, c, d, a, e, collect(DISTINCT t2.transaction_id)[0..$max_n] AS card_neighbors
    OPTIONAL MATCH (d)<-[:USED_DEVICE]-(t3:Transaction)
    WHERE d IS NOT NULL
    WITH t, c, d, a, e, card_neighbors, collect(DISTINCT t3.transaction_id)[0..$max_n] AS device_neighbors
    RETURN t.transaction_id AS txn_id,
           c.card_id AS card_id,
           d.device_id AS device_id,
           a.address_id AS address_id,
           e.email_domain AS email_domain,
           card_neighbors,
           device_neighbors
    
    """
    with driver.session() as session:
        result = session.run(query, txn_id=transaction_id, max_n=MAX_NEIGHBORS_PER_ENTITY)
        record = result.single()
    driver.close()

    if record is None:
        raise ValueError(f"Transaction {transaction_id} not found in Neo4j")
    return record


def build_inference_subgraph(transaction_id):
    record = fetch_live_subgraph(transaction_id)

    card_id = record["card_id"]
    device_id = record["device_id"]
    card_neighbors = record["card_neighbors"] or []
    device_neighbors = record["device_neighbors"] or []

    # Every transaction in this subgraph: target + card-neighbors + device-neighbors
    all_txn_ids = sorted(set([transaction_id] + card_neighbors + device_neighbors))
    txn_remap = {tid: i for i, tid in enumerate(all_txn_ids)}

    sub = HeteroData()
    sub["transaction"].x = torch.cat([get_transaction_features(tid) for tid in all_txn_ids], dim=0)

    # --- Card ---
    if card_id:
        card_deg = np.log1p([len(card_neighbors)])
        card_x = card_scaler.transform(card_deg.reshape(-1, 1))
        sub["card"].x = torch.tensor(np.hstack([card_x, [[0]]]), dtype=torch.float)

        # ONLY connect transactions that actually use this card (target + card_neighbors)
        card_src = [txn_remap[tid] for tid in [transaction_id] + card_neighbors if tid in txn_remap]
        card_tgt = [0] * len(card_src)  # single card node, index 0
    else:
        sub["card"].x = torch.zeros((0, 2))
        card_src, card_tgt = [], []

    # --- Device ---
    if device_id:
        device_deg = np.log1p([len(device_neighbors)])
        device_x = device_scaler.transform(device_deg.reshape(-1, 1))
        sub["device"].x = torch.tensor(np.hstack([device_x, [[0]]]), dtype=torch.float)

        device_src = [txn_remap[tid] for tid in [transaction_id] + device_neighbors if tid in txn_remap]
        device_tgt = [0] * len(device_src)
    else:
        sub["device"].x = torch.zeros((0, 2))
        device_src, device_tgt = [], []

    sub["transaction", "used_card", "card"].edge_index = torch.tensor(
        [card_src, card_tgt], dtype=torch.long) if card_src else torch.zeros((2, 0), dtype=torch.long)
    sub["card", "rev_used_card", "transaction"].edge_index = torch.tensor(
        [card_tgt, card_src], dtype=torch.long) if card_src else torch.zeros((2, 0), dtype=torch.long)
    sub["transaction", "used_device", "device"].edge_index = torch.tensor(
        [device_src, device_tgt], dtype=torch.long) if device_src else torch.zeros((2, 0), dtype=torch.long)
    sub["device", "rev_used_device", "transaction"].edge_index = torch.tensor(
        [device_tgt, device_src], dtype=torch.long) if device_src else torch.zeros((2, 0), dtype=torch.long)

    target_index = txn_remap[transaction_id]
    return sub, target_index

    address_id = record["address_id"]
    email_domain = record["email_domain"]

    # Address/email: single unexpanded node, connected ONLY to the target
    # transaction — no neighbor expansion (avoids the supernode problem),
    # but preserves the input shape the model was trained on.
    if address_id:
        sub["address"].x = torch.zeros((1, 2))  # degree feature unknown at serve time; neutral placeholder
        addr_src = [txn_remap[transaction_id]]
        addr_tgt = [0]
    else:
        sub["address"].x = torch.zeros((0, 2))
        addr_src, addr_tgt = [], []

    if email_domain:
        sub["email"].x = torch.zeros((1, 2))
        email_src = [txn_remap[transaction_id]]
        email_tgt = [0]
    else:
        sub["email"].x = torch.zeros((0, 2))
        email_src, email_tgt = [], []

    sub["transaction", "from_address", "address"].edge_index = torch.tensor(
        [addr_src, addr_tgt], dtype=torch.long) if addr_src else torch.zeros((2, 0), dtype=torch.long)
    sub["address", "rev_from_address", "transaction"].edge_index = torch.tensor(
        [addr_tgt, addr_src], dtype=torch.long) if addr_src else torch.zeros((2, 0), dtype=torch.long)
    sub["transaction", "used_email", "email"].edge_index = torch.tensor(
        [email_src, email_tgt], dtype=torch.long) if email_src else torch.zeros((2, 0), dtype=torch.long)
    sub["email", "rev_used_email", "transaction"].edge_index = torch.tensor(
        [email_tgt, email_src], dtype=torch.long) if email_src else torch.zeros((2, 0), dtype=torch.long)

