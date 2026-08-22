import pandas as pd
from src.graph.connection import get_driver

DATA_PATH = "data/processed_sample.parquet"
BATCH_SIZE = 1000
MAX_RELATIONSHIPS = 400_000
RELATIONSHIPS_PER_TRANSACTION = 4


def estimate_node_count(df):
    n_txn = len(df)
    n_card = df["card_id"].nunique()
    n_addr = df["address_id"].nunique()
    n_device = df["device_id"].nunique()
    n_email = df["email_domain"].nunique()
    total = n_txn + n_card + n_addr + n_device + n_email

    print(f"Transaction nodes: {n_txn}")
    print(f"Card nodes:        {n_card}")
    print(f"Address nodes:     {n_addr}")
    print(f"Device nodes:      {n_device}")
    print(f"EmailDomain nodes: {n_email}")
    print(f"TOTAL:             {total} (Aura free tier cap: ~200,000)")

    if total > 200_000:
        raise ValueError(
            f"Estimated {total} nodes exceeds Aura free tier cap. "
            "Reduce SAMPLE_SIZE in load_and_sample.py and re-run."
        )
    return total


def create_constraints(driver):
    constraints = [
        "CREATE CONSTRAINT txn_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
        "CREATE CONSTRAINT card_id IF NOT EXISTS FOR (c:Card) REQUIRE c.card_id IS UNIQUE",
        "CREATE CONSTRAINT addr_id IF NOT EXISTS FOR (a:Address) REQUIRE a.address_id IS UNIQUE",
        "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
        "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:EmailDomain) REQUIRE e.email_domain IS UNIQUE",
    ]
    with driver.session() as session:
        for c in constraints:
            session.run(c)
    print("Constraints created.")


def ingest_batch(session, batch):
    query = """
    UNWIND $rows AS row
    MERGE (t:Transaction {transaction_id: row.TransactionID})
      SET t.amount = row.TransactionAmt,
          t.is_fraud = row.isFraud,
          t.dt = row.TransactionDT

    MERGE (c:Card {card_id: row.card_id})
    MERGE (t)-[:USED_CARD]->(c)

    MERGE (a:Address {address_id: row.address_id})
    MERGE (t)-[:FROM_ADDRESS]->(a)

    MERGE (d:Device {device_id: row.device_id})
    MERGE (t)-[:USED_DEVICE]->(d)

    MERGE (e:EmailDomain {email_domain: row.email_domain})
    MERGE (t)-[:USED_EMAIL_DOMAIN]->(e)
    """
    session.run(query, rows=batch)


def ingest(df, driver):
    cols = ["TransactionID", "TransactionAmt", "isFraud", "TransactionDT",
            "card_id", "address_id", "device_id", "email_domain"]
    records = df[cols].to_dict(orient="records")

    with driver.session() as session:
        relationship_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        ).single()["count"]
        required_relationships = len(records) * RELATIONSHIPS_PER_TRANSACTION
        if relationship_count + required_relationships > MAX_RELATIONSHIPS:
            available = max(MAX_RELATIONSHIPS - relationship_count, 0)
            max_transactions = available // RELATIONSHIPS_PER_TRANSACTION
            raise ValueError(
                f"Neo4j relationship limit reached: {relationship_count:,}/{MAX_RELATIONSHIPS:,} used. "
                f"This ingest needs up to {required_relationships:,} more. "
                f"Clear the existing graph or use a database with more capacity; "
                f"at most {max_transactions:,} new transactions fit."
            )

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            ingest_batch(session, batch)
            print(f"Ingested {min(i + BATCH_SIZE, len(records))}/{len(records)} transactions")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    estimate_node_count(df)

    driver = get_driver()
    create_constraints(driver)
    ingest(df, driver)
    driver.close()
    print("Ingestion complete.")