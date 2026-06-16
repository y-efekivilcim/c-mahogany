# Mahogany

Gradient descent as a physical system. Watch a neural network organise itself into text.

A PyTorch backend trains node embeddings toward pixel targets — the word "mahogany" in a cursive script. Each node's coordinates are learnable parameters; the optimiser is SGD. A FastAPI server streams the training state via WebSocket to a canvas physics engine, which interpolates node positions toward the optimiser's current output. Mouse interactions corrupt the embedding weights directly, forcing recovery in real-time.

When the loss converges, the canvas settles. "NETWORK CONVERGED."

**[→ kivilcimlab.org/mahogany](https://kivilcimlab.org/mahogany)**

---

## The physics equivalence

Gradient descent on a spring-mass network is mathematically equivalent to a damped physical system: the gradient is the restoring force, the learning rate is the spring constant, momentum damps oscillation. Mahogany makes that equivalence visible. The network isn't being *animated* to look like physics — it *is* physics.

## Dual stability condition

The system only settles when two conditions are met simultaneously:
1. **Mathematical convergence** — loss below threshold
2. **Physical equilibrium** — kinetic energy below threshold

This prevents the canvas from freezing while nodes are still moving, and prevents the physics from running after the loss has plateaued.

## Performance

Spatial hashing limits connection rendering to nearby node pairs only. Without it, rendering O(n²) edges at 1238 nodes and 3694 topology edges would drop the framerate significantly.

## Architecture

```
PyTorch training loop (backend)
        ↓ WebSocket (streaming weight updates)
Canvas physics engine (frontend)
        ↑ Mouse drag events → weight corruption → backend
```

## Stack

- Python — PyTorch, FastAPI, WebSocket
- JavaScript — Canvas physics engine, spatial hashing
- React (frontend shell)
