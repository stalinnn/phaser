import numpy as np
import time
import os
import glob

class RealEvTTCDataset:
    """
    真实的 EvTTC (FCWD) 数据集加载器。
    从指定的目录中读取 .txt 事件文件。
    格式为: timestamp(s) x y polarity(0 or 1)
    
    为了与流体引擎兼容，将其转换为 (x, y, t_us, p) 的张量格式。
    """
    def __init__(self, data_dir, duration_s=1.0, start_time_offset=0.0):
        self.data_dir = data_dir
        self.duration_s = duration_s
        self.start_time_offset = start_time_offset # 从序列的第几秒开始读取
        
    def load_events(self):
        """
        读取并解析真实的事件 txt 文件。
        由于事件量可能极大，我们根据 duration_s 限制读取的范围。
        """
        # 获取所有 .txt 文件并排序
        txt_files = sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))
        if not txt_files:
            print(f"[!] 在 {self.data_dir} 未找到 .txt 事件文件。")
            return np.empty((0, 4))
            
        print(f"[*] 找到 {len(txt_files)} 个事件切片文件，开始读取...")
        events_list = []
        
        # 为了高效，我们尝试从文件中找出属于我们所求时间段的事件
        # 假设文件名与时间戳有一定的单调关系
        current_time_s = None
        base_time_s = None
        
        for file_path in txt_files:
            try:
                # 简单高效的读取：每行格式为 t, x, y, p
                # 使用 np.loadtxt，指定只读所需列，或者先读入再筛选
                data = np.loadtxt(file_path, dtype=np.float32)
                if data.size == 0:
                    continue
                    
                # 如果是多行，data 的 shape 为 (N, 4)
                # 列的顺序是: 0:time, 1:x, 2:y, 3:polarity
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                    
                if base_time_s is None:
                    base_time_s = data[0, 0]
                    
                # 根据我们的偏移量进行筛选
                relative_time_s = data[:, 0] - base_time_s
                
                # 筛选属于 [start_time_offset, start_time_offset + duration_s] 的事件
                valid_mask = (relative_time_s >= self.start_time_offset) & (relative_time_s < self.start_time_offset + self.duration_s)
                valid_data = data[valid_mask]
                
                if valid_data.size > 0:
                    # 重新组织列顺序以适配我们之前的引擎格式: x, y, t_us, p
                    # t 需要转成从 0 开始的微秒
                    t_us = (valid_data[:, 0] - base_time_s - self.start_time_offset) * 1e6
                    x = valid_data[:, 1]
                    y = valid_data[:, 2]
                    p = valid_data[:, 3]
                    
                    # 在 FCWD 中极性可能是 0 和 1，转换为 -1 和 1 更有利于能量扰动的对称性
                    p = np.where(p == 0, -1, 1)
                    
                    formatted_events = np.column_stack((x, y, t_us, p))
                    events_list.append(formatted_events)
                    
                # 如果已经读取超过我们想要的时长，可以提前终止
                if relative_time_s[-1] >= self.start_time_offset + self.duration_s:
                    break
                    
            except Exception as e:
                print(f"[!] 读取文件 {file_path} 时出错: {e}")
                continue
                
        if events_list:
            all_events = np.vstack(events_list)
            # 确保按时间排序
            all_events = all_events[all_events[:, 2].argsort()]
            return all_events
        else:
            return np.empty((0, 4))

class MockEvTTCDataset:
    """
    用于调试全双工流体控制引擎的模拟事件相机数据集。
    生成包含 (x, y, t, p) 的事件流张量。
    - x, y: 像素坐标 (默认 128x128)
    - t: 时间戳 (微秒 us)
    - p: 极性 (1 或 -1，代表变亮或变暗)
    """
    def __init__(self, width=128, height=128, duration_ms=1000):
        self.width = width
        self.height = height
        self.duration_us = duration_ms * 1000
        
    def generate_looming_collision(self, collision_time_ms=800):
        """
        生成一个“迫近碰撞 (Looming)”事件流。
        前半段：平稳行驶，只有少量随机背景噪声事件。
        后半段（接近 collision_time）：视野中心出现一个迅速放大的物体，产生大量边缘扩张事件。
        """
        events = []
        collision_time_us = collision_time_ms * 1000
        
        # 1. 注入背景白噪声事件 (平稳期)
        # 假设整个过程中有恒定的随机事件
        num_bg_events = self.duration_us // 100  # 每 100us 产生一个噪声
        t_bg = np.random.uniform(0, self.duration_us, num_bg_events)
        x_bg = np.random.randint(0, self.width, num_bg_events)
        y_bg = np.random.randint(0, self.height, num_bg_events)
        p_bg = np.random.choice([-1, 1], num_bg_events)
        
        for i in range(num_bg_events):
            events.append([x_bg[i], y_bg[i], t_bg[i], p_bg[i]])
            
        # 2. 注入迫近碰撞事件 (高熵突变期)
        # 假设障碍物从 t=500ms 开始出现，逐渐变大，直到 collision_time 占满屏幕
        start_looming_us = 500 * 1000
        if collision_time_us > start_looming_us:
            # 随着时间推移，事件密度呈指数增加 (模拟 1/TTC 膨胀率)
            t_looming = np.linspace(start_looming_us, collision_time_us, 5000)
            
            center_x, center_y = self.width // 2, self.height // 2
            
            for t in t_looming:
                # 剩余时间 TTC
                ttc = (collision_time_us - t + 1e-3) / 1e6
                # 障碍物半径反比于 TTC
                radius = min(self.width // 2, int(5.0 / ttc)) 
                
                # 在圆环边缘产生事件 (变亮或变暗)
                angle = np.random.uniform(0, 2 * np.pi)
                ex = int(center_x + radius * np.cos(angle))
                ey = int(center_y + radius * np.sin(angle))
                
                if 0 <= ex < self.width and 0 <= ey < self.height:
                    ep = 1  # 假设障碍物比背景亮
                    events.append([ex, ey, t, ep])
                    
        # 转换为 Numpy 数组并按时间戳排序
        events = np.array(events)
        events = events[events[:, 2].argsort()]
        return events

class EventStreamer:
    """
    事件流式加载器 (Streaming DataLoader)。
    将排序好的事件数组按照指定的时间窗口 (dt) 切片，模拟连续时间步滴入。
    """
    def __init__(self, events, dt_ms=1.0):
        self.events = events
        self.dt_us = dt_ms * 1000
        self.current_idx = 0
        self.total_events = len(events)
        self.current_time_us = 0.0
        self.max_time_us = events[-1, 2] if self.total_events > 0 else 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_time_us > self.max_time_us and self.current_idx >= self.total_events:
            raise StopIteration
            
        next_time_us = self.current_time_us + self.dt_us
        
        # 寻找落在这个时间窗口内的所有事件
        start_idx = self.current_idx
        while self.current_idx < self.total_events and self.events[self.current_idx, 2] < next_time_us:
            self.current_idx += 1
            
        end_idx = self.current_idx
        
        # 提取当前微步 (micro-step) 的事件切片
        micro_batch = self.events[start_idx:end_idx]
        
        timestamp_ms = self.current_time_us / 1000.0
        self.current_time_us = next_time_us
        
        return timestamp_ms, micro_batch

if __name__ == "__main__":
    print("================================================================")
    print(" 测试 Event Data Pipeline (EvTTC 流式数据加载器)")
    print("================================================================")
    
    # 测试加载真实的 FCWD 数据
    real_data_dir = "/phaser/EvTTC_Fluid_Control/data/FCWD/FCWD/20240302/2024-03-02-10-35-27/Event/right"
    if os.path.exists(real_data_dir):
        print("\n[*] 检测到真实的 FCWD 数据集，正在加载...")
        # 为了演示，我们只读取前 0.5 秒的真实数据
        real_dataset = RealEvTTCDataset(data_dir=real_data_dir, duration_s=0.5)
        events = real_dataset.load_events()
    else:
        print("\n[*] 未检测到真实的 FCWD 数据集，退回使用 Mock 发生器...")
        dataset = MockEvTTCDataset(width=128, height=128, duration_ms=1000)
        events = dataset.generate_looming_collision(collision_time_ms=800)
        
    print(f"[*] 共生成/读取事件数量: {len(events)}")
    if len(events) > 0:
        print(f"[*] 事件格式示例 [x, y, t(us), p]: {events[0]}")
    print(f"[*] 数据维度 -> {events.shape}\n")
    
    # 2. 模拟流式滴入过程 (dt = 10ms，即帧率 100Hz 的观察窗，但网络内部可以更细)
    dt_ms = 10.0
    streamer = EventStreamer(events, dt_ms=dt_ms)
    
    print(f"[*] 开始以 {dt_ms}ms 微步长模拟事件滴入过程 (只打印变化显著的时间段):")
    print("----------------------------------------------------------------")
    
    for t_ms, micro_batch in streamer:
        num_events = len(micro_batch)
        
        # 为了输出清爽，只打印事件密集的时刻或初始阶段
        if t_ms < 50 or (num_events > 500) or (790 < t_ms < 810):
            if num_events > 500:
                print(f"[T={t_ms:6.1f} ms] 滴入事件数: {num_events:5d} 💥 [真实高熵/突变特征涌现!]")
            else:
                print(f"[T={t_ms:6.1f} ms] 滴入事件数: {num_events:5d}  (平稳期)")
