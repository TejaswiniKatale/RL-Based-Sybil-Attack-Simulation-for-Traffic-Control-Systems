"""Module 11: RL-based Sybil attack agent for adaptive traffic signal control."""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sybil_env import SybilAttackEnv

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        return Transition(*zip(*random.sample(self.memory, batch_size)))

    def __len__(self) -> int:
        return len(self.memory)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)


class AttackDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, gamma: float = 0.98, tau: float = 1e-3) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 64
        self.q = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer()

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q(x), dim=1).item())

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        b = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array(b.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(b.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(b.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(b.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(b.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.q(next_states), dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())

    def soft_update(self) -> None:
        for target_param, local_param in zip(self.target.parameters(), self.q.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * local_param.data)

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.q.state_dict())


def train(args: argparse.Namespace) -> None:
    env = SybilAttackEnv(gui=args.gui, episode_seconds=args.seconds, target_edge=args.target_edge, max_fake=args.max_fake)
    agent = AttackDQNAgent(env.state_dim, env.action_dim)
    Path("results").mkdir(exist_ok=True)
    eps_start, eps_end, eps_decay = 1.0, 0.05, max(args.episodes * 0.7, 1)

    with open("results/attack_training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "avg_fake", "avg_target_queue", "detected", "epsilon"])
        writer.writeheader()
        for ep in range(1, args.episodes + 1):
            env.seed = args.seed + ep
            state = env.reset()
            done = False
            total_reward = 0.0
            infos: List[dict] = []
            epsilon = eps_end + (eps_start - eps_end) * max(0.0, (eps_decay - ep) / eps_decay)
            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.memory.push(state, action, reward / 100.0, next_state, done)
                agent.learn()
                state = next_state
                total_reward += reward
                infos.append(info)
            row = {
                "episode": ep,
                "reward": round(total_reward, 3),
                "avg_fake": round(float(np.mean([x["fake_count"] for x in infos])), 3) if infos else 0,
                "avg_target_queue": round(float(np.mean([x["target_queue"] for x in infos])), 3) if infos else 0,
                "detected": int(any(x["detected"] for x in infos)),
                "epsilon": round(epsilon, 4),
            }
            writer.writerow(row)
            print(row)
            if ep % args.save_every == 0:
                agent.save(args.model)
    env.close()
    agent.save(args.model)
    print(f"Saved attacker model to {args.model}")


def evaluate(args: argparse.Namespace) -> None:
    env = SybilAttackEnv(gui=args.gui, episode_seconds=args.seconds, target_edge=args.target_edge, max_fake=args.max_fake, seed=args.seed)
    agent = AttackDQNAgent(env.state_dim, env.action_dim)
    agent.load(args.model)
    state = env.reset()
    done = False
    total_reward = 0.0
    infos: List[dict] = []
    while not done:
        action = agent.act(state, epsilon=0.0)
        state, reward, done, info = env.step(action)
        total_reward += reward
        infos.append(info)
    env.close()
    print("Sybil attacker evaluation")
    print({
        "reward": round(total_reward, 3),
        "avg_fake": round(float(np.mean([x["fake_count"] for x in infos])), 3) if infos else 0,
        "avg_target_queue": round(float(np.mean([x["target_queue"] for x in infos])), 3) if infos else 0,
        "detected": bool(any(x["detected"] for x in infos)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fake", type=int, default=10)
    parser.add_argument("--target-edge", default="east_in")
    parser.add_argument("--model", default="sybil_attacker.pth")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
