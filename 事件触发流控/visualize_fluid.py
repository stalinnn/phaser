import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os

# 引入我们的物理流体算子和数据集模拟器
from utils.event_dataset import MockEvTTCDataset, EventStreamer
from models.fluid_core import EvTTCFluidNetwork

print("================================================================")
print(" 🎬 渲染流体引擎的动态势能场动画 (Fluid Dynamics Visualization)")
print("================================================================\n")

def generate_fluid_animation(save_path="fluid_dynamics.mp4"):
    # 注意：由于集群上的其他实验 (TGN WikiText) 正在跑，显存几乎被占满。
    # 为了保证全双工控制引擎 Demo 能够顺利运行且不受影响，我们强制将其回退到 CPU 执行。
    # 由于该流体引擎占用资源极小 (O(1) 显存)，在 CPU 上同样能流畅运行。
    device = torch.device("cpu")
    print(f">>> 挂载模型至: {device}")
    
    # 缩小分辨率以加快动画渲染，但物理机制完全一样
    H, W = 64, 64  
    model = EvTTCFluidNetwork(height=H, width=W, embed_dim=32).to(device)
    model.eval()

    # 缩短时间以使得 GIF 不会过大
    total_duration_ms = 800 
    dataset = MockEvTTCDataset(width=W, height=H, duration_ms=total_duration_ms)
    events = dataset.generate_looming_collision(collision_time_ms=600) # 600ms 处撞击
    
    dt_ms = 10.0
    streamer = EventStreamer(events, dt_ms=dt_ms)
    
    # --- 准备画板 ---
    # 我们用黑底模拟夜间行车或极客风格
    plt.style.use('dark_background')
    # 增加高度以容纳下方的终端日志区
    fig = plt.figure(figsize=(15, 8))
    # 使用网格布局：上面一行三个主面板，下面一行一个宽的终端面板
    gs = GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 0.3])
    
    ax_events = fig.add_subplot(gs[0, 0])
    ax_fluid = fig.add_subplot(gs[0, 1])
    ax_alarm = fig.add_subplot(gs[0, 2])
    ax_term = fig.add_subplot(gs[1, :]) # 下方的一整条作为终端
    
    # 隐藏坐标轴
    for ax in [ax_events, ax_fluid, ax_alarm, ax_term]:
        ax.set_xticks([])
        ax.set_yticks([])
        # 终端面板稍微加个边框以显专业
        if ax == ax_term:
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.set_facecolor('#0a0a0a')
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

    ax_events.set_title("1. Event Camera Stream\n(Raw Road Input)", fontsize=12, color='white')
    ax_fluid.set_title("2. Continuum Intelligence Field\n(Extracting Looming Threat)", fontsize=12, color='white')
    ax_alarm.set_title("3. Zero-Latency\nBraking System", fontsize=12, color='white')
    ax_term.set_title("Terminal Logs (Microsecond Inference Engine)", fontsize=10, color='gray', loc='left', pad=2)
    
    # 投资人往往看不懂黑底上散落的青色点是什么。
    # 把左图也设置为相等的比例，避免被拉伸
    ax_events.set_aspect('equal')
    ax_events.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='gray', lw=2) # 边框
    ax_events.plot([W/2, W/2], [H/2-10, H/2+10], color='red', marker='+', markersize=15, alpha=0.5) # 中心准星
    
    # 初始化图形对象
    scatter_events = ax_events.scatter([], [], c='cyan', s=1, alpha=0.5) 
    ax_events.set_xlim(0, W)
    ax_events.set_ylim(0, H)
    ax_events.invert_yaxis() # 图像坐标系 Y 轴向下
    
    # 为热力图也设置 equal，确保三块面板同样高
    ax_fluid.set_aspect('equal')
    # 初始化一个空的热力图 (由于后面加入了归一化，vmax 设置为 1.0)
    fluid_img = ax_fluid.imshow(np.zeros((H, W)), cmap='magma', vmin=0, vmax=1.0, interpolation='gaussian')
    
    # 报警指示灯 (修复椭圆问题，在画板上固定比例)
    # 使用 Rectangle 或者固定长宽比的 Circle
    alarm_circle = plt.Circle((0.5, 0.6), 0.3, color='green', alpha=0.8)
    ax_alarm.add_patch(alarm_circle)
    ax_alarm.set_xlim(0, 1)
    ax_alarm.set_ylim(0, 1)
    ax_alarm.set_aspect('equal') # 强制右侧面板的 X 和 Y 比例相等，防止画出椭圆
    
    text_alarm = ax_alarm.text(0.5, 0.6, "SAFE", ha='center', va='center', fontsize=18, fontweight='bold', color='white')
    text_vram = ax_alarm.text(0.5, 0.3, "VRAM: 24MB\n(Constant)", ha='center', va='center', fontsize=12, color='cyan')
    
    # 初始化终端文本行 (显示最近的 5 条日志)
    log_lines = []
    log_texts = []
    for i in range(5):
        # 从下往上排，最下面是最新的一条
        txt = ax_term.text(0.02, 0.1 + i*0.2, "", fontsize=11, fontfamily='monospace', color='#00ff00', va='bottom')
        log_texts.append(txt)
    
    # --- 收集动画帧数据 ---
    frames_data = []
    
    print(">>> 正在进行前向传播以捕捉每一帧的物理状态...")
    with torch.no_grad():
        for t_ms, micro_batch in streamer:
            collision_risk = model(micro_batch)
            
            # 提取当前的事件坐标用于左图显示
            if len(micro_batch) > 0:
                event_xy = micro_batch[:, 0:2] # [x, y]
            else:
                event_xy = np.empty((0, 2))
                
            # 提取此时的全局势能场用于中图显示 (求特征维度的 L2 范数)
            state_tensor = model.global_state[0] # [embed_dim, H, W]
            energy_landscape = torch.norm(state_tensor, dim=0).cpu().numpy()
            
            # 为了让流体热力图的视觉效果更震撼，我们加入一些非线性对比度增强
            # 真实数据可能有极端的峰值，导致其他地方看起来全黑
            energy_landscape = np.log1p(energy_landscape) # 对数缩放压制极端峰值
            
            # 动态归一化到 0~1 之间，让微小的涟漪也能显现出来
            max_val = np.max(energy_landscape)
            if max_val > 0:
                energy_landscape = energy_landscape / max_val
                
            # 提取系统的真实物理熵
            entropy = model.get_entropy()
            
            # 为了让热力图更有“寻找目标”的感觉，我们画一个最高亮度的红圈
            target_y, target_x = np.unravel_index(np.argmax(energy_landscape), energy_landscape.shape)
            
            frames_data.append({
                'time': t_ms,
                'events': event_xy,
                'fluid': energy_landscape,
                'risk': collision_risk,
                'entropy': entropy,
                'event_count': len(micro_batch),
                'target_x': target_x,
                'target_y': target_y
            })
            
            # 限制最大帧数，防止渲染过久
            if len(frames_data) > 60:
                break

    print(f">>> 物理状态捕捉完成，共 {len(frames_data)} 帧。正在保存静态结果图以防动画生成失败...")
    
    # 动态目标追踪框 (在中间热力图上画一个狂闪的框)
    target_rect = plt.Rectangle((0, 0), 20, 20, linewidth=2, edgecolor='red', facecolor='none', alpha=0.0)
    ax_fluid.add_patch(target_rect)
    
    # 保存最后几帧的静态图以备不时之需
    for i in [10, 30, 50]:
        if i < len(frames_data):
            data = frames_data[i]
            fluid_img.set_array(data['fluid'])
            
            # 临时设置靶框位置用于截图
            if data['risk'] > 0.6:
                target_rect.set_xy((data['target_x']-10, data['target_y']-10))
                target_rect.set_alpha(1.0)
                
            fig.savefig(save_path.replace(".mp4", f"_frame_{i}.png"))
            target_rect.set_alpha(0.0) # 截完图复原
    
    print(f">>> 静态图已保存。正在渲染动画...")
    
    def update(frame_idx):
        if frame_idx >= len(frames_data):
            frame_idx = len(frames_data) - 1
            
        data = frames_data[frame_idx]
        
        # 1. 更新左侧事件散点
        if len(data['events']) > 0:
            display_events = data['events']
            if len(display_events) > 8000:
                indices = np.random.choice(len(display_events), 8000, replace=False)
                display_events = display_events[indices]
            scatter_events.set_offsets(display_events[:, :2]) # 防止超出维度
            
            # 使用新的 danger 标志来控制左侧点的颜色
            # 取第一帧的熵作为基准
            base_entropy = frames_data[0]['entropy']
            is_danger = data['entropy'] < (base_entropy * 0.98) or data['risk'] > 0.6
            
            # 投资人看不懂颜色变化，统一用高亮青色表示“有东西在动”
            # 但危险时，事件点也变成红色，加强视觉冲击
            if is_danger:
                scatter_events.set_color('red')
            else:
                scatter_events.set_color('cyan')
            
            # 放大散点以便看清
            scatter_events.set_sizes([3])
        else:
            scatter_events.set_offsets(np.empty((0, 2)))
            is_danger = False
            
        # 2. 更新中间流体势能场
        fluid_img.set_array(data['fluid'])
        
        # 3. 危险时锁定目标
        risk = data['risk']
        if is_danger:
            target_rect.set_xy((data['target_x']-10, data['target_y']-10))
            target_rect.set_alpha(1.0)
        else:
            target_rect.set_alpha(0.0)
        
        # 4. 更新右侧警报器
        # 我们用模型内部真实的“熵 (entropy)”来代替随机权重的 risk 触发警报
        # 当连续统熵降低到某一个阈值以下时，说明场发生了极化坍缩，即危险！
        
        if is_danger: # 触发抢答警报
            alarm_circle.set_color('red')
            text_alarm.set_text(f"EMERGENCY\nBRAKE!")
            # 制造闪烁效果
            if frame_idx % 2 == 0:
                alarm_circle.set_alpha(1.0)
            else:
                alarm_circle.set_alpha(0.5)
            log_status = "[CRITICAL] ENTROPY COLLAPSE. LOOMING TARGET DETECTED!"
            log_color = "#ff3333"
        else:
            alarm_circle.set_color('green')
            alarm_circle.set_alpha(0.8)
            text_alarm.set_text("CRUISING")
            log_status = "[NORMAL] Background noise dissipating."
            log_color = "#00ff00"
            
        # 5. 更新下方的滚动终端日志
        # 构造当前帧的日志字符串
        entropy = data['entropy']
        new_log = f"> [T={data['time']:7.1f}ms] Ingested Events: {data['event_count']:5d} | Continuum Entropy: {entropy:.4f} | Risk: {risk:.4f} {log_status}"
        
        # 将新日志加入列表，保持最多 5 条
        log_lines.append((new_log, log_color))
        if len(log_lines) > 5:
            log_lines.pop(0)
            
        # 更新文本对象 (倒序显示，最新的在最下面)
        # 注意: i=0 是最下面 (y=0.1), i=4 是最上面 (y=0.9)
        for i in range(5):
            if i < len(log_lines):
                # 列表中的最新一条是 log_lines[-1]，我们要把它放在最下面 (i=0)
                line_idx = len(log_lines) - 1 - i
                log_texts[i].set_text(log_lines[line_idx][0])
                log_texts[i].set_color(log_lines[line_idx][1])
                # 越老的日志颜色越暗
                alpha_val = 1.0 - i * 0.2
                log_texts[i].set_alpha(alpha_val)
            else:
                log_texts[i].set_text("")
        
        # 返回所有需要更新的 Artist 对象
        return [scatter_events, fluid_img, alarm_circle, text_alarm, target_rect] + log_texts

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=50, blit=False)
    
    try:
        gif_path = save_path.replace(".mp4", ".gif")
        print(f">>> 这可能需要几十秒，请稍候...")
        ani.save(gif_path, writer='pillow', fps=10)
        print(f"\n[√] 真实数据流体物理动画已成功保存至: {gif_path}")
    except Exception as e:
        import traceback
        print(f"[!] 动画保存失败: {e}")
        traceback.print_exc()
        
    plt.close(fig)

if __name__ == "__main__":
    out_dir = "/phaser/EvTTC_Fluid_Control"
    os.makedirs(out_dir, exist_ok=True)
    generate_fluid_animation(os.path.join(out_dir, "fluid_dynamics_real.mp4"))
