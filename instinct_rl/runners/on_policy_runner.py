# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import importlib
import os
import statistics
import time
from collections import deque

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import instinct_rl
import instinct_rl.algorithms as algorithms
import instinct_rl.modules as modules
from instinct_rl.env import VecEnv
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class OnPolicyRunner:
    """
    OnPolicyRunner (同策略运行器)
    作为 PPO 等同策略强化学习算法的“大管家”。
    负责协调物理环境（Environment）和算法策略（Algorithm），控制数据收集与模型更新的循环。
    """
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """
        初始化运行器，构建策略网络、算法实例与数据存储等。
        ## Args:
            - env: 验证环境（通常由 Isaac Lab 或类似的物理引擎封装而来）
            - train_cfg: 训练配置文件字典（包含算法配置、网络结构配置等）
            - log_dir: 日志记录的根目录路径
            - device: 运行设备（如 'cpu', 'cuda:0'）
        """
        # import pdb; pdb.set_trace()
        # print("???????")
        print("在3之后执行的这步初始化")
        
        self.cfg = train_cfg                        # 全局训练配置
        self.alg_cfg = train_cfg["algorithm"]       # 具体算法（如 PPO）的配置参数
        self.policy_cfg = train_cfg["policy"]       # 策略网络（Actor-Critic）的结构配置
        self.device = device                        # 运算设备
        self.env = env                              # 保存环境实例

        # 获取观察空间的格式（包含各路观测张量的名称与形状字典）
        obs_format = env.get_obs_format()

        # 根据配置，利用反射机制构建 Actor-Critic 神经网络模型
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),      # 模型类的名称（例如 'ActorCritic'）
            self.policy_cfg,                        # 模型初始化的其余参数
            obs_format,                             # 观测空间维度，用于构建网络输入层
            num_actions=env.num_actions,            # 动作空间维度，用于构建网络输出层
            num_rewards=env.num_rewards,            # 奖励的维度数
        ).to(self.device)

        # 实例化 PPO 算法核心类，将构建好的网络结构直接交由算法进行管理和更新
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = importlib.import_module() if ":" in alg_class_name else getattr(algorithms, alg_class_name)
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # 每次策略更新前，每个环境实例需要执行的步数（决定 Rollout buffer 的大小）
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        # 定期保存权重的间隔（多少次学习迭代保存一次）
        self.save_interval = self.cfg["save_interval"]

        # 处理观察值归一化器 (Normalizers) 的构建
        # 对于像关节位置、速度等各种量纲不一的数据，使用滑动平均等方式做归一化
        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config: dict = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]), # 获取这组观测的总展平维度
                normalizer_class_name=config.pop("class_name"),          # 归一化类名，如 'EmpiricalNormalization'
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

        # 初始化强化学习数据存储库(Rollout Storage) 
        # 它会在内存中分配好张量空间，用于存放每一步采集来的 obs, reward, action 等
        self.alg.init_storage(
            self.env.num_envs,          # 并行环境数量
            self.num_steps_per_env,     # 每次学习阶段收集的数据步数
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
        )

        # 初始化相关的日志记录参数
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0                  # 记录已运行的总步数
        self.tot_time = 0                       # 记录总消耗的时间
        self.current_learning_iteration = 0     # 当前所处的学习迭代轮数
        self.log_interval = self.cfg.get("log_interval", 1)  # Tensorboard 写入频率间隔
        # 保存用于记录 git diff 提交状态的仓库路径
        self.git_status_repos = [instinct_rl.__file__]  

        # 在实例化后，先强行重置一次环境，以获取第一帧的状态
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        开始真正的强化学习大循环。
        ## Args:
            - num_learning_iterations: 要进行多少次完整的学习（采集+更新）迭代。
            - init_at_random_ep_len: 如果为真，环境初始的已运行步数会设为随机值，方便打散截断时机。
        """
        print("开始学习") #加注释
        
        # 地形统计结构
        self.terrain_stats = {}
        # 地形类型列表（每个env一个），直接引用环境的分配
        def get_env_terrain_types():
            env = self.env
            # 递归查找底层环境的 terrain_type_list
            for _ in range(5):  # 最多递归5层
                if hasattr(env, "terrain_type_list") and env.terrain_type_list is not None:
                    return env.terrain_type_list
                if hasattr(env, "env"):
                    env = env.env
                elif hasattr(env, "unwrapped"):
                    env = env.unwrapped
                else:
                    break
            return ["unknown"] * getattr(self.env, "num_envs", 16)

        # 针对多卡分布式（DDP）训练的情况。如果在并行的非0 rank 上，初始化模型多卡同步。
        if dist.is_initialized():
            self.alg.distributed_data_parallel()
            print(f"[INFO rank {dist.get_rank()}]: DistributedDataParallel enabled.")
            
        # 仅在主进程（rank 0 或单进程）上初始化 Tensorboard 记录器
        if self.log_dir is not None and self.writer is None and (not self.is_mp_rank_other_process()):
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            
        # 让所有并行环境的初始 episode 长度随机化，以避免所有的环境在同一时间同时发生 done (截断或结束)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            
        # 取出初始的一帧观测值
        obs, extras = self.env.get_observations()  #obs [num_envs, obs_dim]  extras observation
        obs = obs.to(self.device)
        critic_obs = extras["observations"].get("critic", None)
        critic_obs = critic_obs.to(self.device) if critic_obs is not None else None
        
        # 将网络和归一化器设为训练模式（开启 dropout，滑动平均更新等）
        self.train_mode()

        # 用于存储标量监控信息的队列库（如当前的进度信息，多维度奖励的回报分布等）
        ep_infos = []
        step_infos = []
        rframebuffer = [deque(maxlen=2000) for _ in range(self.env.num_rewards)]  #记录最近 2000 个单步/单帧的瞬间奖励分布
        rewbuffer = [deque(maxlen=100) for _ in range(self.env.num_rewards)]#记录最近 100 次完整回合 (Episode) 结束时的总累计奖励。
        lenbuffer = deque(maxlen=100)   #记录最近 100 次机器人**“存活了多少步”*
        cur_reward_sum = torch.zeros(self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        # 初始化地形类型（假设reset后env有terrain_type_list属性）
        self.terrain_type_list = get_env_terrain_types()

        print(
            "[INFO{}]: Initialization done, start learning.".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        print(
            "NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore"
            " it if it does not affect the performance."
        )
        
        # 将当前启动实验时，代码的 git 更改(diff) 保存到日志里以便以后复现排查
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
            
        # 计算本次运行会达到的总目标跌代量
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations #当前轮加上我自己轮
        tot_start_time = time.time()
        start = time.time()
        
        # import pdb; pdb.set_trace()
        print("4.1")
        print("进入迭代")
        
        # ---------------------------------------------
        # 强化学习最核心的交替循环：(阶段A)收集数据 -> (阶段B)更新网络 -> 循环
        # ---------------------------------------------
        while self.current_learning_iteration < tot_iter:
            
            # --- 阶段 A：Rollout (探索与数据收集) ---
            # 使用 torch.inference_mode 关闭梯度求导以节约显存和算力
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                # 执行指定的 num_steps_per_env 步与环境做交互来装满内存 Buffer
                for i in range(self.num_steps_per_env):
                    # print(f"[DEBUG][STEP] self.env type: {type(self.env)}")
                    # print(f"[DEBUG][STEP] self.env.terrain_type_list(before): {getattr(self.env, 'terrain_type_list', None)}")
                    # 如果 terrain_type_list 丢失，强制重新分配
                    if not hasattr(self.env, 'terrain_type_list') or self.env.terrain_type_list is None:
                        patch_cfgs = getattr(getattr(self.env, 'terrain', None), 'terrain_generator', None)
                        if patch_cfgs and hasattr(patch_cfgs, 'subterrain_specific_cfgs'):
                            patch_cfgs_list = patch_cfgs.subterrain_specific_cfgs
                            num_envs = getattr(self.env, 'num_envs', len(patch_cfgs_list))
                            self.env.terrain_type_list = [getattr(patch_cfgs_list[i % len(patch_cfgs_list)], 'name', 'unknown') for i in range(num_envs)]
                            # print(f"[DEBUG][STEP] 强制分配 terrain_type_list: {self.env.terrain_type_list}")
                    
                    # 建议 1：只在第一步的时候打断点，防止死循环卡住！
                    if i == 0:
                        # import pdb; pdb.set_trace()
                        print("4.2: 开始第 1 步收集数据！(建议在这里按 's' 进入 rollout_step，这步后面是奖励)")
                        print("这步后面是奖励")
   
                    # => 调用内部函数 rollout_step：模型产生动作，送到环境中。取得下一次各种返回值。
                    print("在这后面进的奖励")
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    # 每步都刷新地形类型列表，保证统计用的是最新分配
                    self.terrain_type_list = get_env_terrain_types()
                    # print(f"[DEBUG][STEP] self.env.terrain_type_list(after): {getattr(self.env, 'terrain_type_list', None)}")

                    
                    if i == 0:
                        print("rollout结束")
                        
                    # 对奖励升维处理，应对单维或是多目标奖励配置
                    if len(rewards.shape) == 1:
                        rewards = rewards.unsqueeze(-1)

                    # 下列代码主要用于指标监控与日志记录（Book keeping）
                    if self.log_dir is not None:
                        # 记录来自环境 infos 里的用户自定义统计字段
                        if "step" in infos:
                            step_infos.append(infos["step"])
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        # 累加各个并行环境的 reward 及运行长度
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        
                        # 挑选出在本帧发生了 Done（成功结束或触发惩罚终止/截断）的环境 id
                        new_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
                        # 结算并归档那些完成了整个 Episode 周期的数据表现
                        for i in range(self.env.num_rewards):
                            rframebuffer[i].extend(rewards[dones < 1][:, i].cpu().numpy().tolist())
                            rewbuffer[i].extend(cur_reward_sum[new_ids][:, i].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        
                        # 把产生了 Done 的环境的累计奖励和运行长度清零，从头开始
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # 地形摔倒统计
                        terrain_types = get_env_terrain_types()
                        # print(f"[DEBUG][STAT] terrain_types: {terrain_types}")
                        for idx in new_ids:
                            terrain = terrain_types[idx]
                            # print(f"[DEBUG][STAT] env_id={idx}, terrain_name={terrain}")
                            if terrain not in self.terrain_stats:
                                self.terrain_stats[terrain] = {"total": 0, "fall": 0}
                            self.terrain_stats[terrain]["total"] += 1
                            # 摔倒判据：reward小于0（可根据实际情况调整）
                            # 多reward时只看第一个reward
                            if rewards[idx][0].item() < 0:
                                self.terrain_stats[terrain]["fall"] += 1

                stop = time.time()
                collection_time = stop - start  # 统计收集数据这一个阶段的耗时
                
                # import pdb; pdb.set_trace()
                print("4.3: 数据收集完毕，准备计算优势函数(GAE)！")                             
                
                # --- 阶段 B：Learning step 优势计算 ---
                start = stop
                # 使用存满的 buffer 数据和最后的 next_obs 来计算广义优势估计（Advantage/Returns）
                #前面推理一遍的运算开始打分
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)
            
            # import pdb; pdb.set_trace()
            print("4.4: 优势函数计算完毕，准备进行 PPO 神经网络更新！(按 's' 进去看Loss)")
            
            # --- 阶段 C：Learning step 反向传播更新 ---
            # 缩进已经退出了 inference_mode，梯度引擎开启。
            # PPO 算法开始复用多次 (epochs) 该批次的数据，计算 Surrogate Loss 和 Value Loss，并更新网络参数。
            #上面compute出来的直接切片扔进去更新了
            losses, stats = self.alg.update(self.current_learning_iteration)
            
            stop = time.time()
            learn_time = stop - start # 记录网络模型训练花费的时间
            
            # 以一定的迭代频率将各类 log (损失、标量、FPS表现等) 刷出并写进 Tensorboard
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
                # 输出地形摔倒统计
                print("地形摔倒统计：")
                for terrain, stat in self.terrain_stats.items():
                    rate = stat["fall"] / stat["total"] if stat["total"] > 0 else 0
                    print(f"地形 {terrain}: 摔倒 {stat['fall']}/{stat['total']}，摔倒率 {rate*100:.1f}%")
                
            # 到达指定间隔则存储一次网络神经权重到磁盘 (.pt)
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
                
            # 清理本轮指标记录，迭代数 +1
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

        # 全步长迭代结束，保证存下最后一个断点权重
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def rollout_step(self, obs, critic_obs):
        """
        单独封装的单步环境数据采集逻辑
        """
        # 1. 算法层面：让 Actor 根据当前 obs 选出动作（在 PPO 中往往是高斯采样出 action，并同步记录 log_prob）
        actions = self.alg.act(obs, critic_obs)
        print("在4.2.0")
        # 2. 物理世界层面：把 actions 全下发给环境。环境运算一步物理学状态，并且把各类收益/新观察返回来。
        obs, rewards, dones, infos = self.env.step(actions) #更新了！！！！！！！
        print("在4.2.1")
        # 针对 Actor 和 Critic 输入的观察分离（Asymmetric Actor-Critic 架构常用技巧），拆出对应的特权观察值
        critic_obs = infos["observations"].get("critic", None) #从infos字典里面掏出来的
        print("在4.2.2")
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        print("在4.2.3")
        # 3. 规整化 (Normalization) 层面：使用滑动平均对当前的观测结果做归一化，增强训练稳定性。
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic":
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            else:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])
        print("在4.2.4")                
        # 4. 数据留存层面：将上面发生的所有历史事件 (动作、价值、奖励、是否done、正常化后的观测) 统统推入 PPO 内存库(Storage) 中备用。
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos

    """
    Logging / 杂项数据监控类辅助函数
    """

    def log(self, locs, width=80, pad=35):
        """控制台及 Tensorboard 打印监控指标的函数，收集 Loss/Reward/FPS 并在屏幕规范化输出。"""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs["tot_start_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = []
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor.append(ep_info[key].to(self.device))
                infotensor = torch.cat(infotensor)
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                self.writer_mp_add_scalar(
                    (key if key.startswith("Episode") else "Episode/" + key),
                    value,
                    self.current_learning_iteration,
                )
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if locs["step_infos"]:
            for key in locs["step_infos"][0]:
                infotensor = []
                texts = []
                for step_info in locs["step_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if isinstance(step_info[key], str):
                        texts.append(step_info[key])
                        continue
                    elif not isinstance(step_info[key], torch.Tensor):
                        step_info[key] = torch.Tensor([step_info[key]])
                    if len(step_info[key].shape) == 0:
                        step_info[key] = step_info[key].unsqueeze(0)
                    infotensor.append(step_info[key].to(self.device))
                infotensor = torch.cat(infotensor)
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                if len(texts) > 0 and (not self.is_mp_rank_other_process()):
                    self.writer.add_text(
                        (key if key.startswith("Step") else "Step/" + key),
                        "\n".join(texts),
                        self.current_learning_iteration,
                    )
                else:
                    self.writer_mp_add_scalar(
                        (key if key.startswith("Step") else "Step/" + key),
                        value,
                        self.current_learning_iteration,
                    )
                    ep_string += f"""{f'Mean step {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        if dist.is_initialized():
            fps *= dist.get_world_size()

        for k, v in locs["losses"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Train/" + k, v.item(), self.current_learning_iteration)

        self.writer_mp_add_scalar("Loss/learning_rate", self.alg.learning_rate, self.current_learning_iteration)
        self.writer_mp_add_scalar("Policy/mean_noise_std", mean_std.item(), self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/total_fps", fps, self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/collection_time", locs["collection_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/learning_time", locs["learn_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar(
            "Perf/gpu_allocated", torch.cuda.memory_allocated(self.device) / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_total", torch.cuda.mem_get_info(self.device)[1] / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_global_free_mem",
            torch.cuda.mem_get_info(self.device)[0] / 1024**3,
            self.current_learning_iteration,
        )
        for i in range(self.env.num_rewards):
            self.writer_mp_add_scalar(
                f"Train/mean_reward_each_timestep_{i}",
                statistics.mean(locs["rframebuffer"][i]),
                self.current_learning_iteration,
            )
        if len(locs["rewbuffer"][0]) > 0:
            for i in range(self.env.num_rewards):
                self.writer_mp_add_scalar(
                    f"Train/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.current_learning_iteration
                )
                self.writer_mp_add_scalar(
                    f"Train/ratio_above_mean_reward_{i}",
                    statistics.mean(
                        [(1.0 if rew > statistics.mean(locs["rewbuffer"][i]) else 0) for rew in locs["rewbuffer"][i]]
                    ),
                    self.current_learning_iteration,
                )
                self.writer_mp_add_scalar(
                    f"Train/time/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.tot_time
                )
            self.writer_mp_add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/median_episode_length", statistics.median(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/min_episode_length", min(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/max_episode_length", max(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/time/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.tot_time
            )

        info_str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"][0]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean([statistics.mean(buf) for buf in locs['rewbuffer']]):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n"""
        )
        if not self.is_mp_rank_other_process():
            print(log_string)

    def add_git_repo_to_log(self, repo_path):
        self.git_status_repos.append(repo_path)

    def save(self, path, infos=None):
        """Save training state dict to file. Will not happen if in multi-process and not rank 0."""
        # 保存整个模型的权重参数以及归一化器的状态，供从断点恢复与推理。
        if self.is_mp_rank_other_process():
            return

        run_state_dict = self.alg.state_dict()
        run_state_dict.update(  # Nothing will update if there is no normalizers
            {
                f"{group_name}_normalizer_state_dict": normalizer.state_dict()
                for group_name, normalizer in self.normalizers.items()
            }
        )
        run_state_dict.update(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        )
        torch.save(run_state_dict, path)

    def load(self, path):
        """Load training state dict from file. Will not happen if in multi-process and not rank 0."""
        # 从本地 `.pt` 文件恢复各种状态参数，以达到继续训练 (Resume) 的效果。
        if self.is_mp_rank_other_process():
            return

        loaded_dict = torch.load(path, weights_only=True)
        if self.cfg.get("ckpt_manipulator", False):
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            loaded_dict = getattr(ckpt_manipulator, self.cfg["ckpt_manipulator"])(
                loaded_dict,
                self.alg.state_dict(),
                **self.cfg.get("ckpt_manipulator_kwargs", {}),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")

        self.alg.load_state_dict(loaded_dict)

        for group_name, normalizer in self.normalizers.items():
            if not f"{group_name}_normalizer_state_dict" in loaded_dict:
                print(
                    f"\033[1;36m Warning, normalizer for {group_name} is not found, the state dict is not loaded"
                    " \033[0m"
                )
            else:
                normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])

        self.current_learning_iteration = loaded_dict["iter"]
        if self.cfg.get("ckpt_manipulator", False):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            except Exception as e:
                print(f"\033[1;36m Save manipulated checkpoint failed with error: {str(e)} \n ignored... \033[0m")
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()# 1. 开启评估模式（关闭梯度的各种随机性）
        if device is not None:
            self.alg.actor_critic.to(device)

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)
            # 2. 核心：返回一个只包含 Actor 推理的干净函数
            policy = lambda x: self.alg.actor_critic.act_inference(self.normalizers["policy"](x))  # noqa: E731
        else:
            policy = self.alg.actor_critic.act_inference

        return policy

    def export_as_onnx(self, obs, export_model_dir):
        self.eval_mode()
        if "policy" in self.normalizers:
            obs = self.normalizers["policy"](obs)
            # also export obs normalizer
            self.normalizers["policy"].export(os.path.join(export_model_dir, "policy_normalizer.npz"))
        self.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    """
    Helper functions
    """

    def is_mp_rank_zero_process(self):
        """Check if the current process should do all reduce to summarize the training statistics and checkpoint."""
        return dist.is_initialized() and dist.get_rank() == 0

    def is_mp_rank_other_process(self):
        """Check if current process is in torch distributed multi-processing and not rank 0, or single process."""
        return dist.is_initialized() and dist.get_rank() != 0

    def gather_stat_values(self, values: torch.Tensor, gather_op: str = "mean", remove_nan: bool = True):
        """Properly gather the value across all processes. summarize the input values directly if not in multi-processing.
        用于分布式多显卡 (DDP) 跨通信收缩整合各个节点的奖励/Loss（求均值/最大/最小值）的辅助函数。
        ## Args:
            - values: torch.Tensor, the value to summarize. Do not summarize this into a scalar before calling this function.
            - gather_op: dist.ReduceOp, the operation to contat the value.
            - remove_nan: bool, whether to remove NaN values before summarizing.
        ## Return:
            The summarized value across all processes.
        """
        if remove_nan:
            values = values[~values.isnan()]
        values = values.to(self.device)  # make sure the values are on the all_reduc-able device
        if gather_op == "mean":
            num_values = torch.tensor([torch.numel(values)]).to(self.device)
            values = torch.sum(values)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.SUM)
                dist.all_reduce(num_values, dist.ReduceOp.SUM)
            values = values / num_values.item()
        elif gather_op == "max":
            values = torch.max(values) if values.numel() > 0 else torch.tensor(float("-inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MAX)
        elif gather_op == "min":
            values = torch.min(values) if values.numel() > 0 else torch.tensor(float("inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported gather_op: {gather_op}")
        return values

    def writer_mp_add_scalar(self, key, value, step):
        """Add scalar to tensorboard writer. Will not happen if in multi-process and not rank 0."""
        if not self.is_mp_rank_other_process():
            self.writer.add_scalar(key, value, step)

    def train_mode(self):
        """Change all related models into training mode (for dropout for example)"""
        # 防止如 BatchNormalization 或 Dropout 等层出错，确保在学习阶段保持 train_mode 
        self.alg.actor_critic.train()
        for normalizer in self.normalizers.values():
            normalizer.train()

    def eval_mode(self):
        """Change all related models into evaluation mode (for dropout for example)"""
        # 测试与导出模式下，关闭随机因素与更新。
        self.alg.actor_critic.eval()
        for normalizer in self.normalizers.values():
            normalizer.eval()
