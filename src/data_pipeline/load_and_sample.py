import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SAMPLE_SIZE = 80_000


def load_data():
    train_txn = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
    train_id = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    train = train_txn.merge(train_id, on="TransactionID", how="left")
    print(f"Loaded {train.shape[0]} transactions, {train.shape[1]} columns")
    print(train["isFraud"].value_counts(normalize=True))
    return train


def stratified_sample(train, sample_size=SAMPLE_SIZE):
    sample, _ = train_test_split(
        train,
        train_size=sample_size,
        stratify=train["isFraud"],
        random_state=42,
    )
    print(f"Sampled {sample.shape[0]} transactions")
    print(sample["isFraud"].value_counts(normalize=True))
    return sample


def build_entity_keys(df):
    df = df.copy()

    df["card_id"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].fillna(-1).astype(str) + "_" +
        df["card3"].fillna(-1).astype(str) + "_" +
        df["card5"].fillna(-1).astype(str)
    )

    df["address_id"] = (
        df["addr1"].fillna(-1).astype(str) + "_" +
        df["addr2"].fillna(-1).astype(str)
    )

    df["device_id"] = df["DeviceInfo"].fillna("unknown_device")
    df["email_domain"] = df["P_emaildomain"].fillna("unknown_email")

    print("Unique cards:", df["card_id"].nunique(), "vs", len(df), "transactions")
    print("Unique devices:", df["device_id"].nunique())
    print("Unique addresses:", df["address_id"].nunique())
    print("Unique email domains:", df["email_domain"].nunique())

    return df


if __name__ == "__main__":
    train = load_data()
    sample = stratified_sample(train)
    df = build_entity_keys(sample)
    df.to_parquet(f"{DATA_DIR}/processed_sample.parquet", index=False)
    print(f"Saved processed sample to {DATA_DIR}/processed_sample.parquet")