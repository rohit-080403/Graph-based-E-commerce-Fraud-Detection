import random
from src.graph.connection import get_driver

MIN_RING_SIZE = 4
MAX_RING_SIZE = 10
ISOLATION_THRESHOLD = 2  
RANDOM_SEED = 42


def find_isolated_fraud_transactions(driver):
    query = """
    MATCH (t:Transaction {is_fraud: 1})-[:USED_DEVICE]->(d:Device)
    WITH d, collect(t.transaction_id) AS txn_ids, count(t) AS device_txn_count
    WHERE device_txn_count <= $threshold
    RETURN d.device_id AS device_id, txn_ids
    """
    with driver.session() as session:
        result = session.run(query, threshold=ISOLATION_THRESHOLD)
        return [(r["device_id"], r["txn_ids"]) for r in result]


def build_rings(isolated, min_size=MIN_RING_SIZE, max_size=MAX_RING_SIZE, seed=RANDOM_SEED):
    random.seed(seed)
    all_txn_ids = [txn_id for _, txns in isolated for txn_id in txns]
    random.shuffle(all_txn_ids)

    rings = []
    i = 0
    ring_num = 0
    while i < len(all_txn_ids):
        size = random.randint(min_size, max_size)
        ring_txns = all_txn_ids[i:i + size]
        if len(ring_txns) < min_size:
            break  
        rings.append({"ring_id": f"synthetic_ring_{ring_num:04d}", "txn_ids": ring_txns})
        i += size
        ring_num += 1

    return rings


def inject_rings(driver, rings):
    query = """
    UNWIND $rings AS ring
    MERGE (sd:Device {device_id: 'synth_' + ring.ring_id})
      SET sd.is_synthetic = true, sd.ring_id = ring.ring_id

    WITH ring, sd
    UNWIND ring.txn_ids AS txn_id
    MATCH (t:Transaction {transaction_id: toInteger(txn_id)})
    // Remove old (isolated) device edge, add new shared synthetic one
    OPTIONAL MATCH (t)-[old_rel:USED_DEVICE]->(:Device)
    DELETE old_rel
    MERGE (t)-[:USED_DEVICE]->(sd)
    SET t.ring_id = ring.ring_id
    """
    with driver.session() as session:
        session.run(query, rings=rings)


if __name__ == "__main__":
    driver = get_driver()

    isolated = find_isolated_fraud_transactions(driver)
    total_isolated_txns = sum(len(txns) for _, txns in isolated)
    print(f"Found {len(isolated)} near-isolated devices covering {total_isolated_txns} fraud transactions")

    rings = build_rings(isolated)
    print(f"Built {len(rings)} synthetic rings, sizes {MIN_RING_SIZE}-{MAX_RING_SIZE}")
    print(f"Total transactions re-wired: {sum(len(r['txn_ids']) for r in rings)}")

    inject_rings(driver, rings)
    driver.close()
    print("Injection complete.")