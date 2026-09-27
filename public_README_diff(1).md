--- public/README.md (原始)
# Mars Rover PPO Training Solution

## Overview

This solution trains a PPO agent to drive a Mars rover as far as possible in 300 seconds of simulation time. The policy is exported as an ONNX model compatible with the arena evaluation system.

## Architecture

- **Algorithm**: Proximal Policy Optimization (PPO) with GAE
- **Network**: Feedforward MLP with LayerNorm (256→256→128 hidden)
- **Memory**: M=1 (no recurrence needed)
- **Training**: 64 parallel environments, ~100 minutes on RTX 5090

## Key Design Decisions

1. **Dense reward shaping**: Forward distance progress, velocity bonus, engine management, angle stability
2. **Vectorized training**: 64 parallel environments for sample efficiency
3. **Moderate network size**: Fast training while maintaining expressiveness
4. **Entropy scheduling**: High initial entropy for exploration, decaying over training
5. **Frequent saves**: Model saved every 90 seconds to ensure valid output

## Files

- `train.py` - Training script with PPO implementation
- `Dockerfile` - Container specification (FROM arena-base)
- `README.md` - This file

## Expected Performance

With 2 hours of training on RTX 5090:
- ~50M environment steps collected
- Median distance: 200+ meters (target)
- Model converges to effective driving policy

## Notes

- The policy uses all 160 observation dimensions
- Previous action is embedded (8-dim) and concatenated
- All 7 required inputs are present in the ONNX graph
- Export uses `arena.protocol.export_agent_onnx` with fallback to manual ONNX export


+++ public/README.md (修改后)
# Mars Rover PPO Solution
PPO agent with reward shaping for forward progress.
