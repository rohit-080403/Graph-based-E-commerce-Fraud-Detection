from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F

from src.models.gcn_model import FraudGCN
from src.api.subgraph_builder import build_inference_subgraph

app = FastAPI(title="Graph-Based Fraud Prevention Engine")

model = FraudGCN()
model.load_state_dict(torch.load("src/models/fraud_gcn.pt", weights_only=True))
model.eval()


class TransactionRequest(BaseModel):
    transaction_id: int


class FraudResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    prediction: str
    subgraph_size: int


@app.post("/validate-transaction", response_model=FraudResponse)
def validate_transaction(req: TransactionRequest):
    try:
        sub, target_index = build_inference_subgraph(req.transaction_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with torch.no_grad():
        out = model(sub.x_dict, sub.edge_index_dict)
        probs = F.softmax(out, dim=1)
        fraud_prob = probs[target_index, 1].item()

    return FraudResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        prediction="FRAUD" if fraud_prob > 0.5 else "CLEAN",
        subgraph_size=sub["transaction"].x.shape[0],
    )


@app.get("/health")
def health():
    return {"status": "ok"}