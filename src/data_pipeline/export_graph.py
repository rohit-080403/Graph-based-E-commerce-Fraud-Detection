import pickle
from src.graph.connection import get_driver

OUT_PATH = "data/graph_export.pkl"


def fetch_nodes(driver, label, id_field, extra_fields=None):
    extra = ", " + ", ".join(f"n.{f} AS {f}" for f in extra_fields) if extra_fields else ""
    query = f"MATCH (n:{label}) RETURN n.{id_field} AS node_id{extra}"
    with driver.session() as session:
        result = session.run(query)
        return [dict(r) for r in result]


def fetch_edges(driver, src_label, src_id, rel_type, tgt_label, tgt_id):
    query = f"""
    MATCH (s:{src_label})-[:{rel_type}]->(t:{tgt_label})
    RETURN s.{src_id} AS src, t.{tgt_id} AS tgt
    """
    with driver.session() as session:
        result = session.run(query)
        return [(r["src"], r["tgt"]) for r in result]


def build_index_map(node_ids):
    """Maps each entity's real ID (string or int) -> a 0-based integer position."""
    return {node_id: idx for idx, node_id in enumerate(node_ids)}


def export_graph():
    driver = get_driver()

    print("Fetching nodes...")
    transactions = fetch_nodes(driver, "Transaction", "transaction_id",
                                extra_fields=["amount", "is_fraud", "dt", "ring_id"])
    cards = fetch_nodes(driver, "Card", "card_id")
    addresses = fetch_nodes(driver, "Address", "address_id")
    devices = fetch_nodes(driver, "Device", "device_id", extra_fields=["is_synthetic", "ring_id"])
    emails = fetch_nodes(driver, "EmailDomain", "email_domain")

    print(f"  Transactions: {len(transactions)}")
    print(f"  Cards:        {len(cards)}")
    print(f"  Addresses:    {len(addresses)}")
    print(f"  Devices:      {len(devices)}")
    print(f"  EmailDomains: {len(emails)}")

    
    txn_idx = build_index_map([t["node_id"] for t in transactions])
    card_idx = build_index_map([c["node_id"] for c in cards])
    addr_idx = build_index_map([a["node_id"] for a in addresses])
    device_idx = build_index_map([d["node_id"] for d in devices])
    email_idx = build_index_map([e["node_id"] for e in emails])

    print("\nFetching edges...")
    txn_card_edges = fetch_edges(driver, "Transaction", "transaction_id", "USED_CARD", "Card", "card_id")
    txn_addr_edges = fetch_edges(driver, "Transaction", "transaction_id", "FROM_ADDRESS", "Address", "address_id")
    txn_device_edges = fetch_edges(driver, "Transaction", "transaction_id", "USED_DEVICE", "Device", "device_id")
    txn_email_edges = fetch_edges(driver, "Transaction", "transaction_id", "USED_EMAIL_DOMAIN", "EmailDomain", "email_domain")

    print(f"  Transaction-Card edges:  {len(txn_card_edges)}")
    print(f"  Transaction-Address edges: {len(txn_addr_edges)}")
    print(f"  Transaction-Device edges:  {len(txn_device_edges)}")
    print(f"  Transaction-Email edges:   {len(txn_email_edges)}")

    driver.close()

    export = {
        "nodes": {
            "transaction": transactions,
            "card": cards,
            "address": addresses,
            "device": devices,
            "email": emails,
        },
        "index_maps": {
            "transaction": txn_idx,
            "card": card_idx,
            "address": addr_idx,
            "device": device_idx,
            "email": email_idx,
        },
        "edges": {
            "txn_card": txn_card_edges,
            "txn_address": txn_addr_edges,
            "txn_device": txn_device_edges,
            "txn_email": txn_email_edges,
        },
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(export, f)

    print(f"\nExport saved to {OUT_PATH}")
    return export


if __name__ == "__main__":
    export_graph()