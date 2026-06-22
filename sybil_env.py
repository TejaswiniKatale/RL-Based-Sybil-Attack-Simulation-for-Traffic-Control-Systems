"""SUMO environment for Module 11: RL Sybil attack against adaptive signal control.

This is a controlled simulation. Fake BSMs are modeled at the traffic-controller
perception layer. They are not inserted as real vehicles into the road network.
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import traci
except ImportError as exc:
    raise ImportError("Install SUMO Python tools: pip install traci sumolib") from exc

IN_EDGES = ["north_in", "south_in", "east_in", "west_in"]
NS_EDGES = ["north_in", "south_in"]
EW_EDGES = ["east_in", "west_in"]
TLS_ID = "center"


class SybilAttackEnv:
    """RL environment where an attacker learns fake-BSM intensity.

    Defender/controller:
        A simple adaptive controller chooses the green direction using perceived
        queue counts. The attacker adds fake BSM counts to a target approach.

    Actions:
        0..max_fake -> number of fake BSM vehicles added to target perception.

    Reward:
        target congestion gain - detection penalty - network-wide penalty.
    """

    def __init__(
        self,
        sumocfg: str = "sumocfg.xml",
        gui: bool = False,
        episode_seconds: int = 1800,
        target_edge: str = "east_in",
        max_fake: int = 10,
        detection_jump_threshold: int = 6,
        seed: int | None = None,
    ) -> None:
        self.sumocfg = sumocfg
        self.gui = gui
        self.episode_seconds = episode_seconds
        self.target_edge = target_edge
        self.max_fake = max_fake
        self.detection_jump_threshold = detection_jump_threshold
        self.seed = seed
        self.started = False
        self.step_count = 0
        self.current_phase = 0
        self.last_fake = 0
        self.detected = False
        self.phase_age = 0

    @property
    def state_dim(self) -> int:
        return 11

    @property
    def action_dim(self) -> int:
        return self.max_fake + 1

    def _sumo_binary(self) -> str:
        return "sumo-gui" if self.gui else "sumo"

    def _ensure_network(self) -> None:
        if not Path("intersection.net.xml").exists():
            subprocess.run(["netconvert", "-n", "nodes.xml", "-e", "edges.xml", "-o", "intersection.net.xml"], check=True)

    def _write_random_routes(self) -> None:
        rng = random.Random(self.seed)
        ns_rate = rng.randint(250, 600)
        ew_rate = rng.randint(250, 600)
        routes = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" guiShape="passenger" carFollowModel="IDM"/>
    <flow id="north_south" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ns_rate}" from="north_in" to="south_out"/>
    <flow id="south_north" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ns_rate}" from="south_in" to="north_out"/>
    <flow id="east_west" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ew_rate}" from="east_in" to="west_out"/>
    <flow id="west_east" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ew_rate}" from="west_in" to="east_out"/>
</routes>
'''
        Path("routes.xml").write_text(routes, encoding="utf-8")

    def reset(self) -> np.ndarray:
        self.close()
        self._ensure_network()
        self._write_random_routes()
        cmd = [self._sumo_binary(), "-c", self.sumocfg, "--no-warnings", "true"]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        traci.start(cmd)
        self.started = True
        self.step_count = 0
        self.current_phase = 0
        self.phase_age = 0
        self.last_fake = 0
        self.detected = False
        self._set_green(0)
        for _ in range(3):
            traci.simulationStep()
        return self._get_state(fake_count=0)

    def close(self) -> None:
        if self.started:
            try:
                traci.close(False)
            except Exception:
                pass
            self.started = False

    def _set_green(self, phase: int) -> None:
        try:
            phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(TLS_ID)[0].phases
            phase_index = 0 if phase == 0 else min(2, len(phases) - 1)
            traci.trafficlight.setPhase(TLS_ID, phase_index)
        except Exception:
            pass
        self.current_phase = phase
        self.phase_age = 0

    def _real_queues(self) -> Dict[str, float]:
        return {edge: float(traci.edge.getLastStepHaltingNumber(edge)) for edge in IN_EDGES}

    def _waiting_times(self) -> Dict[str, float]:
        return {edge: float(traci.edge.getWaitingTime(edge)) for edge in IN_EDGES}

    def _adaptive_defender(self, fake_count: int) -> int:
        q = self._real_queues()
        perceived = dict(q)
        perceived[self.target_edge] += fake_count
        ns_pressure = sum(perceived[e] for e in NS_EDGES)
        ew_pressure = sum(perceived[e] for e in EW_EDGES)
        return 0 if ns_pressure >= ew_pressure else 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        fake_count = int(np.clip(action, 0, self.max_fake))
        jump = abs(fake_count - self.last_fake)
        detected_now = jump > self.detection_jump_threshold
        self.detected = self.detected or detected_now

        chosen_phase = self._adaptive_defender(fake_count)
        if chosen_phase != self.current_phase and self.phase_age >= 8:
            self._set_green(chosen_phase)

        traci.simulationStep()
        self.step_count += 1
        self.phase_age += 1

        q = self._real_queues()
        waits = self._waiting_times()
        target_queue = q[self.target_edge]
        total_queue = sum(q.values())
        total_wait = sum(waits.values())

        detection_penalty = 300.0 if detected_now else 0.0
        stealth_penalty = 10.0 * fake_count
        network_penalty = 0.2 * total_queue
        reward = 20.0 * target_queue - detection_penalty - stealth_penalty - network_penalty

        self.last_fake = fake_count
        done = self.step_count >= self.episode_seconds or traci.simulation.getMinExpectedNumber() <= 0 or self.detected
        info = {
            "fake_count": fake_count,
            "target_queue": target_queue,
            "total_queue": total_queue,
            "total_wait": total_wait,
            "detected": float(self.detected),
        }
        return self._get_state(fake_count), float(reward), done, info

    def _get_state(self, fake_count: int) -> np.ndarray:
        q = self._real_queues()
        waits = self._waiting_times()
        ns_q = sum(q[e] for e in NS_EDGES)
        ew_q = sum(q[e] for e in EW_EDGES)
        values = [
            q["north_in"] / 30.0,
            q["south_in"] / 30.0,
            q["east_in"] / 30.0,
            q["west_in"] / 30.0,
            waits["north_in"] / 300.0,
            waits["south_in"] / 300.0,
            waits["east_in"] / 300.0,
            waits["west_in"] / 300.0,
            fake_count / max(1, self.max_fake),
            1.0 if self.current_phase == 0 else 0.0,
            (ew_q - ns_q) / 60.0,
        ]
        return np.array(values, dtype=np.float32)
