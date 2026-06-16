import asyncio
import json
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LatentManifold(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.coords = nn.Embedding(num_nodes, 2)
        self.coords.weight.data.normal_(0.5, 0.05)
        
    def forward(self, indices):
        return self.coords(indices)

async def train_loop(websocket: WebSocket, model: LatentManifold, optimizer: optim.Optimizer, target_tensor: torch.Tensor):
    num_nodes = target_tensor.shape[0]
    indices = torch.arange(num_nodes)
    
    try:
        while True:
            optimizer.zero_grad()
            preds = model(indices)
            loss = nn.MSELoss(reduction='sum')(preds, target_tensor)
            loss.backward()
            optimizer.step()
            
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
                    
            mean_loss = loss.item() / num_nodes
            data = {
                "loss": mean_loss,
                "grad_norm": grad_norm,
                "preds": preds.detach().cpu().tolist()
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.066)
    except Exception as e:
        pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    training_task = None
    model = None
    
    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            
            if payload.get("type") == "init":
                targets = payload.get("targets", [])
                if targets:
                    if training_task:
                        training_task.cancel()
                        
                    target_tensor = torch.tensor(targets, dtype=torch.float32)
                    num_nodes = target_tensor.shape[0]
                    
                    model = LatentManifold(num_nodes)
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
                    
                    training_task = asyncio.create_task(train_loop(websocket, model, optimizer, target_tensor))
                    
            elif payload.get("type") == "disrupt" and model is not None:
                data = payload.get("data", [])
                for d in data:
                    idx = int(d[0])
                    x = float(d[1])
                    y = float(d[2])
                    model.coords.weight.data[idx] = torch.tensor([x, y], dtype=torch.float32)
                    
    except WebSocketDisconnect:
        pass
    finally:
        if training_task:
            training_task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
