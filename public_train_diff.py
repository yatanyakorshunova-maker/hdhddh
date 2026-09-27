--- public/train.py (原始)


+++ public/train.py (修改后)
#!/usr/bin/env python3
"""
Mars Rover PPO Training - Optimized for 2-hour window on RTX 5090
Target: 200+ meters median distance across 48 test tracks.

Key strategies:
1. Dense reward shaping for forward progress
2. Engine/gear management encouragement
3. Efficient vectorized training (64 parallel envs)
4. Proper ONNX export via arena.protocol
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# ─── Constants ───────────────────────────────────────────────────────────────

OBS_DIM = 160
NUM_ACTIONS = 31
MEMORY_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO config
GAMMA = 0.997
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15
ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_ENVS = 64
STEPS_PER_UPDATE = 200
PPO_EPOCHS = 4
MINIBATCH_SIZE = 3200
LR = 4e-4
TOTAL_TRAIN_TIME = 100 * 60  # 100 minutes (leave 20 min buffer)


# ─── Network ─────────────────────────────────────────────────────────────────

class RoverPolicy(nn.Module):
    """
    Feedforward policy: 7 inputs -> logits[31] + next_memory[M].
    Uses LayerNorm for training stability.
    """
    def __init__(self):
        super().__init__()
        self.memory_size = MEMORY_SIZE

        # Previous action embedding
        self.act_emb = nn.Embedding(NUM_ACTIONS, 8)

        # Input: obs(160) + act_emb(8) + scalars(4) = 172
        inp_dim = OBS_DIM + 8 + 4

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(inp_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, NUM_ACTIONS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for output layers
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def _preprocess(self, observation, previous_action, previous_reward,
                    previous_done, trial_progress, trial_start):
        """Combine all inputs into feature vector."""
        act_idx = previous_action.long().clamp(0, NUM_ACTIONS - 1)
        ae = self.act_emb(act_idx)
        scalars = torch.stack([
            previous_reward.float(),
            previous_done.float(),
            trial_progress.float(),
            trial_start.float(),
        ], dim=-1)
        return torch.cat([observation, ae, scalars], dim=-1)

    def forward(self, observation, previous_action, previous_reward,
                previous_done, trial_progress, trial_start, memory):
        """
        Returns: (logits [batch, 31], next_memory [batch, M])
        """
        batch = observation.shape[0]
        x = self._preprocess(observation, previous_action, previous_reward,
                             previous_done, trial_progress, trial_start)
        feat = self.features(x)
        logits = self.policy_head(feat)
        logits = torch.clamp(logits, -50.0, 50.0)
        next_memory = torch.zeros(batch, self.memory_size,
                                  device=observation.device)
        return logits, next_memory

    def get_value(self, observation, previous_action, previous_reward,
                  previous_done, trial_progress, trial_start, memory):
        x = self._preprocess(observation, previous_action, previous_reward,
                             previous_done, trial_progress, trial_start)
        feat = self.features(x)
        return self.value_head(feat).squeeze(-1)

    def evaluate(self, observation, previous_action, previous_reward,
                 previous_done, trial_progress, trial_start, memory, actions):
        """Evaluate specific actions. Returns (log_prob, entropy, value)."""
        x = self._preprocess(observation, previous_action, previous_reward,
                             previous_done, trial_progress, trial_start)
        feat = self.features(x)
        logits = torch.clamp(self.policy_head(feat), -50.0, 50.0)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_head(feat).squeeze(-1)
        return log_prob, entropy, value


# ─── Environment ─────────────────────────────────────────────────────────────

class VecEnv:
    """
    Vectorized Mars Rover environment with reward shaping.
    Handles auto-reset and tracks episode statistics.
    """
    def __init__(self, n):
        self.n = n
        self.envs = [gym.make("mars_rover_env-v0") for _ in range(n)]

        # Current state
        self.obs = None
        self.prev_act = np.zeros(n, dtype=np.int64)
        self.prev_rew = np.zeros(n, dtype=np.float32)
        self.prev_done = np.zeros(n, dtype=np.float32)
        self.progress = np.zeros(n, dtype=np.float32)
        self.start_flag = np.ones(n, dtype=np.float32)

        # Reward shaping state
        self.prev_x = np.zeros(n, dtype=np.float32)
        self.max_x = np.zeros(n, dtype=np.float32)
        self.stuck_count = np.zeros(n, dtype=np.int32)
        self.prev_engine_on = np.zeros(n, dtype=np.float32)
        self.steps = np.zeros(n, dtype=np.int64)

        # Stats
        self.ep_returns = np.zeros(n, dtype=np.float32)
        self.ep_dist = np.zeros(n, dtype=np.float32)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self.completed = []  # list of (return, distance, length)

        # 300 seconds * 60 Hz / 8 steps per action = 2250 action steps max
        self.max_ep_steps = 2250

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        self.obs = np.array(obs_list, dtype=np.float32)
        self.prev_act[:] = 0
        self.prev_rew[:] = 0
        self.prev_done[:] = 0
        self.progress[:] = 0
        self.start_flag[:] = 1
        self.prev_x[:] = 0
        self.max_x[:] = 0
        self.stuck_count[:] = 0
        self.prev_engine_on[:] = 0
        self.steps[:] = 0
        self.ep_returns[:] = 0
        self.ep_dist[:] = 0
        self.ep_len[:] = 0
        return self.obs

    def step(self, actions):
        """
        Step all envs. Returns state before step + rewards/dones.
        """
        # Save pre-step state
        obs_old = self.obs.copy()
        pa_old = self.prev_act.copy()
        pr_old = self.prev_rew.copy()
        pd_old = self.prev_done.copy()
        prog_old = self.progress.copy()
        sf_old = self.start_flag.copy()

        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        new_obs_list = []

        for i in range(self.n):
            obs, raw_rew, terminated, truncated, info = self.envs[i].step(int(actions[i]))
            done = terminated or truncated

            # Extract state from observation
            x = obs[0] * 1000.0        # position (meters)
            vx = obs[2] * 20.0         # horizontal velocity
            vz = obs[3] * 20.0         # vertical velocity
            angle = obs[4]             # body angle (radians)
            ang_vel = obs[5] * 10.0    # angular velocity
            energy = obs[6]            # energy fraction
            engine_stop = obs[109]     # engine stopped flag
            gear = obs[104]            # current gear
            in_air = obs[151]          # in air flag
            contact_front = obs[126]   # front body contact
            contact_bottom = obs[127]  # bottom body contact

            # ── Reward shaping ──────────────────────────────────────
            r = 0.0

            # 1. Forward distance (main objective)
            dx = x - self.prev_x[i]
            r += dx * 10.0  # Strong reward for forward progress

            # 2. New record bonus
            if x > self.max_x[i]:
                improvement = x - self.max_x[i]
                r += improvement * 5.0  # Extra bonus for exploring further
                self.max_x[i] = x

            # 3. Forward velocity bonus
            if vx > 0.1:
                r += min(vx, 8.0) * 0.5
            elif vx < -0.5:
                r -= 1.0  # Penalty for going backward

            # 4. Engine management
            # engine_stop=1 means stopped, =0 means running
            # prev_engine_on stores (1-engine_stop), so 1=was running, 0=was stopped
            # Detect transition: was stopped (prev<0.5) and now running (engine_stop==0)
            engine_just_started = (engine_stop < 0.5 and self.prev_engine_on[i] < 0.5)
            if engine_just_started:
                r += 5.0  # Reward for starting engine

            # 5. Angle stability
            abs_angle = abs(angle)
            if abs_angle > 1.5:
                r -= 20.0  # Severe penalty for flipping
            elif abs_angle > 1.0:
                r -= 5.0
            elif abs_angle > 0.6:
                r -= 1.0

            # 6. Ground contact bonus (stability)
            if not in_air:
                r += 0.02

            # 7. Stuck detection
            if abs(dx) < 0.005 and abs(vx) < 0.1:
                self.stuck_count[i] += 1
                if self.stuck_count[i] > 30:
                    r -= 0.5  # Penalty for being stuck
            else:
                self.stuck_count[i] = 0

            # 8. Landing penalty
            if contact_bottom > 0.5:
                r -= 2.0

            # 9. Angular velocity penalty (uncontrolled spinning)
            r -= abs(ang_vel) * 0.1

            # 10. Raw environment reward
            r += raw_rew * 1.0

            # 11. Episode completion bonus
            if done:
                r += self.max_x[i] * 0.3  # Bonus proportional to distance reached

            rewards[i] = r
            dones[i] = float(done)

            # Update tracking
            self.prev_x[i] = x
            self.prev_engine_on[i] = 1.0 - engine_stop
            self.steps[i] += 1
            self.progress[i] = min(self.steps[i] / self.max_ep_steps, 1.0)

            # Episode stats
            self.ep_returns[i] += raw_rew
            self.ep_dist[i] = max(self.ep_dist[i], x)
            self.ep_len[i] += 1

            # Auto-reset on done
            if done:
                self.completed.append((
                    float(self.ep_returns[i]),
                    float(self.ep_dist[i]),
                    int(self.ep_len[i])
                ))
                if len(self.completed) > 2000:
                    self.completed = self.completed[-1000:]

                obs, _ = self.envs[i].reset()
                self.prev_x[i] = 0
                self.max_x[i] = 0
                self.stuck_count[i] = 0
                self.prev_engine_on[i] = 0
                self.steps[i] = 0
                self.progress[i] = 0
                self.ep_returns[i] = 0
                self.ep_dist[i] = 0
                self.ep_len[i] = 0

            new_obs_list.append(obs)

        # Update state for next step
        self.obs = np.array(new_obs_list, dtype=np.float32)
        self.prev_act = actions.astype(np.int64)
        self.prev_rew = rewards.copy()
        self.prev_done = dones.copy()
        self.start_flag = dones.copy()  # New episode starts after done

        return obs_old, pa_old, pr_old, pd_old, prog_old, sf_old, rewards, dones


# ─── PPO Training ────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self):
        self.policy = RoverPolicy().to(DEVICE)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=LR,
                                      weight_decay=1e-5, eps=1e-5)
        self.env = VecEnv(NUM_ENVS)

        self.total_steps = 0
        self.start_time = time.time()
        self.last_save = time.time()
        self.update_i = 0

        # Entropy coefficient schedule (high -> low)
        self.ent_coef = ENTROPY_COEF_START

    def collect_rollout(self):
        """Collect experience from vectorized envs."""
        T = STEPS_PER_UPDATE
        N = NUM_ENVS
        total = T * N

        # Buffers
        obs_b = np.zeros((total, OBS_DIM), dtype=np.float32)
        pa_b = np.zeros(total, dtype=np.int64)
        pr_b = np.zeros(total, dtype=np.float32)
        pd_b = np.zeros(total, dtype=np.float32)
        tp_b = np.zeros(total, dtype=np.float32)
        ts_b = np.zeros(total, dtype=np.float32)
        act_b = np.zeros(total, dtype=np.int64)
        rew_b = np.zeros(total, dtype=np.float32)
        done_b = np.zeros(total, dtype=np.float32)
        logp_b = np.zeros(total, dtype=np.float32)
        val_b = np.zeros(total, dtype=np.float32)

        pos = 0
        for t in range(T):
            with torch.no_grad():
                o = torch.FloatTensor(self.env.obs).to(DEVICE)
                pa = torch.LongTensor(self.env.prev_act).to(DEVICE)
                pr = torch.FloatTensor(self.env.prev_rew).to(DEVICE)
                pd = torch.FloatTensor(self.env.prev_done).to(DEVICE)
                tp = torch.FloatTensor(self.env.progress).to(DEVICE)
                ts = torch.FloatTensor(self.env.start_flag).to(DEVICE)
                mem = torch.zeros(N, MEMORY_SIZE).to(DEVICE)

                logits, _ = self.policy(o, pa, pr, pd, tp, ts, mem)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.policy.get_value(o, pa, pr, pd, tp, ts, mem)

            act_np = actions.cpu().numpy()

            # Store pre-step data
            e = pos + N
            obs_b[pos:e] = self.env.obs
            pa_b[pos:e] = self.env.prev_act
            pr_b[pos:e] = self.env.prev_rew
            pd_b[pos:e] = self.env.prev_done
            tp_b[pos:e] = self.env.progress
            ts_b[pos:e] = self.env.start_flag
            act_b[pos:e] = act_np
            logp_b[pos:e] = log_probs.cpu().numpy()
            val_b[pos:e] = values.cpu().numpy()

            # Step
            (_, _, _, _, _, _, rewards, dones) = self.env.step(act_np)
            rew_b[pos:e] = rewards
            done_b[pos:e] = dones

            pos = e
            self.total_steps += N

        # Bootstrap value
        with torch.no_grad():
            o = torch.FloatTensor(self.env.obs).to(DEVICE)
            pa = torch.LongTensor(self.env.prev_act).to(DEVICE)
            pr = torch.FloatTensor(self.env.prev_rew).to(DEVICE)
            pd = torch.FloatTensor(self.env.prev_done).to(DEVICE)
            tp = torch.FloatTensor(self.env.progress).to(DEVICE)
            ts = torch.FloatTensor(self.env.start_flag).to(DEVICE)
            mem = torch.zeros(N, MEMORY_SIZE).to(DEVICE)
            last_val = self.policy.get_value(o, pa, pr, pd, tp, ts, mem).cpu().numpy()

        # GAE computation
        adv = np.zeros(total, dtype=np.float32)
        ret = np.zeros(total, dtype=np.float32)

        rew_2d = rew_b.reshape(T, N)
        done_2d = done_b.reshape(T, N)
        val_2d = val_b.reshape(T, N)

        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nv = last_val
                nnd = 1.0 - done_2d[t]
            else:
                nv = val_2d[t + 1]
                nnd = 1.0 - done_2d[t]

            delta = rew_2d[t] + GAMMA * nv * nnd - val_2d[t]
            gae = delta + GAMMA * GAE_LAMBDA * nnd * gae
            adv[t*N:(t+1)*N] = gae

        ret = adv + val_b

        return dict(obs=obs_b, prev_actions=pa_b, prev_rewards=pr_b,
                    prev_dones=pd_b, trial_progress=tp_b, trial_starts=ts_b,
                    actions=act_b, log_probs=logp_b, values=val_b,
                    advantages=adv, returns=ret)

    def ppo_update(self, data):
        """Run PPO update on collected data."""
        n = len(data['returns'])

        # Normalize advantages
        adv = data['advantages'].copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {'pl': 0, 'vl': 0, 'ent': 0, 'n': 0}

        for epoch in range(PPO_EPOCHS):
            perm = np.random.permutation(n)

            for s in range(0, n, MINIBATCH_SIZE):
                e = min(s + MINIBATCH_SIZE, n)
                idx = perm[s:e]
                bs = len(idx)

                # Tensors
                o = torch.FloatTensor(data['obs'][idx]).to(DEVICE)
                pa = torch.LongTensor(data['prev_actions'][idx]).to(DEVICE)
                pr = torch.FloatTensor(data['prev_rewards'][idx]).to(DEVICE)
                pd = torch.FloatTensor(data['prev_dones'][idx]).to(DEVICE)
                tp = torch.FloatTensor(data['trial_progress'][idx]).to(DEVICE)
                ts = torch.FloatTensor(data['trial_starts'][idx]).to(DEVICE)
                mem = torch.zeros(bs, MEMORY_SIZE).to(DEVICE)
                acts = torch.LongTensor(data['actions'][idx]).to(DEVICE)
                old_lp = torch.FloatTensor(data['log_probs'][idx]).to(DEVICE)
                mb_adv = torch.FloatTensor(adv[idx]).to(DEVICE)
                mb_ret = torch.FloatTensor(data['returns'][idx]).to(DEVICE)

                # Forward
                new_lp, ent, val = self.policy.evaluate(o, pa, pr, pd, tp, ts, mem, acts)

                # Policy loss
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                pl = -torch.min(surr1, surr2).mean()

                # Value loss
                vl = F.mse_loss(val, mb_ret)

                # Entropy
                ent_loss = -ent.mean()

                # Total
                loss = pl + VALUE_COEF * vl + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                stats['pl'] += pl.item()
                stats['vl'] += vl.item()
                stats['ent'] += ent.mean().item()
                stats['n'] += 1

        self.update_i += 1

        # Decay entropy coefficient
        progress = self.update_i / 3000  # approximate total updates
        self.ent_coef = max(ENTROPY_COEF_END,
                            ENTROPY_COEF_START * (1 - progress))

        return {k: v/max(stats['n'],1) for k, v in stats.items() if k != 'n'}

    def save(self):
        """Save model to /output/policy.onnx"""
        os.makedirs("/output", exist_ok=True)

        # Try export_onnx first (for custom forward with 7 args)
        try:
            from arena.protocol import export_onnx
            export_onnx("/output/policy.onnx", self.policy,
                       obs_dim=OBS_DIM, memory_size=MEMORY_SIZE)
            print("  [OK] Saved via arena.protocol.export_onnx")
            return True
        except Exception as e1:
            pass

        # Try export_agent_onnx (standard format)
        try:
            from arena.protocol import export_agent_onnx
            export_agent_onnx("/output/policy.onnx", self.policy, None)
            print("  [OK] Saved via arena.protocol.export_agent_onnx")
            return True
        except Exception as e2:
            print(f"  [WARN] arena.protocol failed: {e1}, {e2}")
            return self._fallback_export()

    def _fallback_export(self):
        """Fallback ONNX export."""
        try:
            import onnx

            # Move to CPU for export
            self.policy.eval()
            self.policy.cpu()

            b = 1
            inputs = (
                torch.zeros(b, OBS_DIM),
                torch.zeros(b, dtype=torch.int64),
                torch.zeros(b),
                torch.zeros(b),
                torch.zeros(b),
                torch.zeros(b),
                torch.zeros(b, MEMORY_SIZE),
            )

            torch.onnx.export(
                self.policy, inputs, "/output/policy.onnx",
                input_names=['observation', 'previous_action', 'previous_reward',
                            'previous_done', 'trial_progress', 'trial_start', 'memory'],
                output_names=['logits', 'next_memory'],
                dynamic_axes={k: {0: 'batch'} for k in [
                    'observation', 'previous_action', 'previous_reward',
                    'previous_done', 'trial_progress', 'trial_start',
                    'memory', 'logits', 'next_memory']},
                opset_version=17,
            )

            # Add metadata
            model = onnx.load("/output/policy.onnx")
            mp = model.metadata_props.add()
            mp.key = "rover.format"
            mp.value = "rover-policy-onnx-v1"
            onnx.save(model, "/output/policy.onnx")

            self.policy.to(DEVICE)
            self.policy.train()
            print("  [OK] Fallback export successful")
            return True
        except Exception as e:
            print(f"  [ERR] Export failed: {e}")
            self.policy.to(DEVICE)
            self.policy.train()
            return False

    def train(self):
        """Main training loop."""
        print(f"{'='*60}")
        print(f"Mars Rover PPO Training")
        print(f"Device: {DEVICE} | Envs: {NUM_ENVS} | Steps/update: {STEPS_PER_UPDATE}")
        print(f"{'='*60}")

        self.env.reset()
        self.save()  # Initial save
        self.last_save = time.time()

        while True:
            elapsed = time.time() - self.start_time

            # Time check
            if elapsed > TOTAL_TRAIN_TIME:
                print(f"\n[TIME LIMIT] {elapsed/60:.1f} min elapsed")
                break

            # Collect & update
            data = self.collect_rollout()
            stats = self.ppo_update(data)

            # Logging every 10 updates
            if self.update_i % 10 == 0:
                ep_dists = [d for _, d, _ in self.env.completed[-100:]]
                mean_d = np.mean(ep_dists) if ep_dists else 0
                max_d = max(ep_dists) if ep_dists else 0
                med_d = np.median(ep_dists) if ep_dists else 0
                n_ep = len(self.env.completed)

                print(f"[{elapsed/60:6.1f}m] Step {self.total_steps/1e6:.1f}M | "
                      f"PL:{stats['pl']:.3f} VL:{stats['vl']:.3f} "
                      f"Ent:{stats['ent']:.3f} | "
                      f"Ep:{n_ep} Dist(med/max):{med_d:.0f}/{max_d:.0f}m")

            # Periodic save (every 90 seconds)
            if time.time() - self.last_save > 90:
                self.save()
                self.last_save = time.time()

        # Final save
        self.save()

        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training complete: {total_time/60:.1f} min, {self.total_steps:,} steps")

        if self.env.completed:
            dists = [d for _, d, _ in self.env.completed]
            print(f"Episodes: {len(dists)}")
            print(f"Distance: mean={np.mean(dists):.1f}m, "
                  f"median={np.median(dists):.1f}m, max={np.max(dists):.1f}m")
        print(f"{'='*60}")


def main():
    os.makedirs("/output", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
