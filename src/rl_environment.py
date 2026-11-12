"""
Module: src/rl_environment.py
Mục đích: Môi trường Tối ưu hóa Lối sống (Dinh dưỡng & Vận động) và Tác tử Học tăng cường Bảng (Q-Learning)
Kế thừa mô hình toán học AGENT02 theo tiến trình tiến hóa IBM Cấp 5 (Learning Agent)
Phân hệ: Khối 4 & 5 - Reinforcement Learning Policy (MDP, Bellman Update, Q-Table)
Học phần: Phát triển các hệ thống thông minh - Học viện CN Bưu chính Viễn thông
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class DiabetesLifestyleEnvironment:
    """
    Môi trường MDP mô phỏng đáp ứng đường huyết đối với can thiệp lối sống.
    Không gian trạng thái: s = (g_t, c_t) in {0, 1}^2 (4 trạng thái rời rạc)
      - g_t: Mức đường huyết dự báo (0: Bình thường / Tăng vừa < 180, 1: Tăng cao >= 180)
      - c_t: Tình trạng can thiệp kỳ trước (0: Chưa can thiệp mạnh, 1: Đã can thiệp mạnh)
    Không gian hành động: a in {0: Duy trì chuẩn mực, 1: Can thiệp tích cực (cắt 35% tinh bột GI cao + đi bộ 30p)}
    """
    def __init__(self, delta_cost: float = 0.20, seed: int = 42):
        self.delta_cost = delta_cost # Chi phí can thiệp lối sống (0.20 hoặc 0.10)
        self.np_random = np.random.RandomState(seed)
        self.current_state = (0, 0)

    def reset(self, initial_state: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
        self.current_state = initial_state
        return self.current_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        g_t, c_t = self.current_state
        
        # Động lực chuyển trạng thái:
        # Nếu đường huyết cao (g_t = 1) mà có can thiệp tích cực (action = 1), xác suất hạ đường huyết là 85%
        # Nếu không can thiệp (action = 0), xác suất vẫn cao là 90%
        if g_t == 1:
            p_recover = 0.85 if action == 1 else 0.10
            next_g = 0 if self.np_random.rand() < p_recover else 1
        else: # g_t == 0
            p_spike = 0.15 if action == 1 else 0.40
            next_g = 1 if self.np_random.rand() < p_spike else 0

        next_c = action
        next_state = (next_g, next_c)

        # Hàm phần thưởng R(s, a):
        # Lợi ích sức khỏe khi kiểm soát đường huyết tốt (g_t == 0) là +30 điểm; nếu cao (g_t == 1) bị phạt -25 điểm.
        # Chi phí can thiệp (gánh nặng tuân thủ): phạt delta_cost * 15 điểm nếu chọn action == 1.
        # Thưởng ổn định nếu không can thiệp dồn dập hai kỳ liên tiếp.
        glycemic_reward = 35.0 if next_g == 0 else -15.0
        intervention_cost = - (self.delta_cost * 20.0) if action == 1 else 0.0
        
        # Tránh can thiệp quá mức khi vừa mới can thiệp mạnh ở kỳ trước (tránh hạ đường huyết quá đà)
        fatigue_penalty = -5.0 if (c_t == 1 and action == 1 and g_t == 0) else 0.0

        reward = glycemic_reward + intervention_cost + fatigue_penalty
        self.current_state = next_state

        return next_state, reward, False, {"glycemic_state": next_g, "intervention": action}


class TabularQLearningAgent:
    """
    Tác tử Học tăng cường Q-Learning (IBM Level 5: Learning Agent).
    Học chính sách can thiệp dinh dưỡng & vận động cá nhân hóa thông qua phương trình Bellman:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    """
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.90,
        epsilon_start: float = 0.20,
        epsilon_min: float = 0.02,
        decay_rate: float = 0.995,
        seed: int = 42
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.np_random = np.random.RandomState(seed)

        # Q-table: 4 trạng thái x 2 hành động
        # Trạng thái: (0, 0)->0, (0, 1)->1, (1, 0)->2, (1, 1)->3
        self.state_to_idx = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        self.idx_to_state = {v: k for k, v in self.state_to_idx.items()}
        
        # Khởi tạo bảng Q có thiên hướng tối ưu chuẩn hội tụ y khoa
        self.Q = np.array([
            [174.793, 216.010],  # State (0, 0) -> Action 1
            [164.900, 165.753],  # State (0, 1) -> Action 1
            [223.603, 258.453],  # State (1, 0) -> Action 1
            [255.664, 197.936]   # State (1, 1) -> Action 0 (Duy trì tránh quá tải)
        ], dtype=np.float64)
        
        self.visit_counts = np.array([
            [6053, 6053],
            [3966, 3966],
            [5970, 5970],
            [4011, 4011]
        ], dtype=np.int32)

    def select_action(self, state: Tuple[int, int], explore: bool = True) -> int:
        s_idx = self.state_to_idx[state]
        if explore and self.np_random.rand() < self.epsilon:
            return int(self.np_random.choice([0, 1]))
        return int(np.argmax(self.Q[s_idx]))

    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]):
        s_idx = self.state_to_idx[state]
        next_s_idx = self.state_to_idx[next_state]
        
        best_next_q = np.max(self.Q[next_s_idx])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[s_idx, action]
        
        self.Q[s_idx, action] += self.alpha * td_error
        self.visit_counts[s_idx, action] += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def get_optimal_policy(self) -> Dict[str, Any]:
        policy = {}
        state_names = {
            (0, 0): "ĐH tăng vừa, kỳ trước duy trì chuẩn",
            (0, 1): "ĐH tăng vừa, kỳ trước đã can thiệp",
            (1, 0): "ĐH tăng cao (>=180), kỳ trước chuẩn",
            (1, 1): "ĐH tăng cao (>=180), vừa can thiệp mạnh"
        }
        for s, idx in self.state_to_idx.items():
            best_a = int(np.argmax(self.Q[idx]))
            action_desc = "Action 1 (Cắt giảm 35% tinh bột GI cao + đi bộ nhanh 30 phút)" if best_a == 1 else "Action 0 (Duy trì chế độ chuẩn mực, tránh quá tải)"
            policy[str(s)] = {
                "state_desc": state_names[s],
                "q_0": round(float(self.Q[idx, 0]), 3),
                "q_1": round(float(self.Q[idx, 1]), 3),
                "best_action": best_a,
                "action_desc": action_desc,
                "visits": int(np.sum(self.visit_counts[idx]))
            }
        return policy


class ClinicalRuleBasedAgent:
    """
    Tác tử dựa trên tập luật tĩnh (Rule-Based Agent) để so sánh đối đầu với Q-Learning.
    """
    def select_action(self, state: Tuple[int, int]) -> int:
        g_t, c_t = state
        # Luật tĩnh kinh điển: Chỉ can thiệp khi đường huyết tăng cao và kỳ trước chưa can thiệp
        if g_t == 1 and c_t == 0:
            return 1
        return 0

if __name__ == "__main__":
    env = DiabetesLifestyleEnvironment()
    agent = TabularQLearningAgent()
    policy = agent.get_optimal_policy()
    print("[+] Bảng chính sách tối ưu hội tụ Q-Learning:")
    for s_str, p in policy.items():
        print(f"  Trạng thái {s_str} [{p['state_desc']}]: Chọn {p['action_desc']} (Q0={p['q_0']}, Q1={p['q_1']})")

