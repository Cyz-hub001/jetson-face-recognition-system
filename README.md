# Jetson 人脸识别门禁系统

基于 NVIDIA Jetson 的人脸识别门禁控制系统。通过摄像头采集视频，使用 InsightFace 进行人脸检测与识别，当授权人员被成功识别后，向门锁控制器发送串口 "OPEN" 指令以开门。

## 功能特性

- 基于 InsightFace `buffalo_l` ONNX 模型的实时人脸检测与识别
- 多帧连续确认机制：需要连续多次成功识别后才允许开门
- 冷却机制：成功开门后进入冷却期，防止重复触发
- 陌生人录像：检测到陌生人时自动录制视频
- Web 控制面板：实时监控系统状态并支持手动控制
- JSON 格式结构化日志，支持日志轮转
- 优雅降级：未安装串口硬件时仍可正常运行
- 跨平台摄像头支持：CSI/GStreamer（Jetson）和 DirectShow/MSMF（Windows）

## 项目结构

```
face_door_system/
  main.py                # 系统入口与协调器
  recognition_engine.py  # 摄像头采集与人脸识别
  state_machine.py       # 门禁状态机
  serial_comm.py         # 串口通信（控制门锁）
  web_api.py             # FastAPI Web 服务器与 REST API
  services.py            # 业务逻辑层
  access_log.py          # 结构化门禁事件日志
  logger_setup.py        # 日志配置
  video_recorder.py      # 陌生人视频录制
  static/
    index.html           # Web 控制面板页面
  config.json            # 运行时配置文件
```

## 安装部署

### 环境要求

- Python 3.8+
- NVIDIA Jetson 设备（推荐）或带摄像头的 Windows 电脑
- 串口连接的门锁控制器（可选）

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone <repository-url>
   cd weekend_project
   ```

2. 创建并激活虚拟环境：

   ```bash
   conda create -n face_door python=3.10
   conda activate face_door
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. （可选）如需串口硬件控制：

   ```bash
   pip install pyserial
   ```

5. 将已知人脸图片放入 `人脸图片保存/` 目录。支持格式：`.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp`。
   - 图片在子目录中时，以父文件夹名称作为人名
   - 图片在根目录时，以文件名作为人名

6. （可选）将 InsightFace `buffalo_l` 模型权重下载到 `models/insightface/models/buffalo_l/`。未下载时，InsightFace 会在首次运行时自动下载。

## 使用方法

### 命令行演示模式

在 `face_door_system/` 目录下运行：

```bash
cd face_door_system
python main.py
```

该模式使用模拟识别结果进行状态机和串口通信测试。

### Web API 模式

```bash
cd face_door_system
python -m uvicorn web_api:app --host 0.0.0.0 --port 8000
```

控制面板访问地址：`http://localhost:8000/`。人脸识别默认不会自动启动，需在配置中设置 `web.auto_start_recognition` 为 `true`。

### 运行测试

```bash
cd face_door_system
python state_machine_regression_test.py
```

包含 16 项回归测试，覆盖状态机、串口故障、冷却行为、手动开门、服务层和日志轮转。

## 配置说明

所有配置项位于 `face_door_system/config.json`：

### 系统设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `system.owner_name` | `"pyy02"` | 系统所有者名称 |
| `system.window_name` | `"Face Door System"` | 窗口标题 |

### 识别设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `recognition.threshold` | `0.42` | 最小余弦相似度阈值 |
| `recognition.open_score` | `5` | 连续匹配次数达到此值后确认开门 |
| `recognition.cooldown_seconds` | `3.0` | 开门后的冷却时间（秒） |
| `recognition.det_size` | `[320, 320]` | 人脸检测模型输入分辨率 |
| `recognition.loop_interval` | `0.2` | 识别循环间隔（秒） |

### 路径设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `paths.known_faces_dir` | `"../人脸图片保存"` | 已知人脸图片目录 |
| `paths.insightface_model_root` | `"../models/insightface"` | InsightFace 模型存储目录 |
| `paths.trt_cache_dir` | `"~/.cache/insightface_trt"` | TensorRT 缓存目录（Jetson） |

### 摄像头设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `camera.sensor_id` | `0` | 摄像头设备 ID |
| `camera.width` / `height` | `640` / `480` | 目标采集分辨率 |
| `camera.capture_width` / `capture_height` | `1280` / `720` | CSI 摄像头原生分辨率 |
| `camera.fps` | `30` | 帧率 |
| `camera.backend` | `"csi"` | 摄像头后端：`"csi"`、`"auto"` 或指定后端名称 |

### 串口设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `serial.enabled` | `true` | 是否启用串口通信 |
| `serial.port` | `"/dev/ttyUSB0"` | 串口设备路径 |
| `serial.baudrate` | `9600` | 串口波特率 |
| `serial.open_command` | `"OPEN\n"` | 开门指令字符串 |

### Web 设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `web.allow_manual_open` | `true` | 是否允许通过 API 手动开门 |
| `web.manual_open_local_only` | `true` | 手动开门仅限本地访问 |
| `web.auto_start_recognition` | `false` | 服务器启动时自动开始识别 |

### 录像设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `recording.enabled` | `true` | 是否启用陌生人视频录制 |
| `recording.save_dir` | `"recordings"` | 录像保存目录 |
| `recording.duration` | `30` | 最大录制时长（秒） |
| `recording.fps` | `20` | 录制帧率 |
| `recording.codec` | `"mp4v"` | 视频编码格式 |

### 日志设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `logging.log_dir` | `"logs"` | 日志文件目录 |
| `logging.level` | `"INFO"` | 最低日志级别 |
| `logging.max_bytes` | `2097152` | 单个日志文件最大大小（字节） |
| `logging.backup_count` | `5` | 保留的日志备份数量 |

## 状态机流程

系统采用 5 状态有限状态机控制门禁流程：

```
等待 --> 匹配中 --> 已确认 --> 开门 --> 冷却中 --> 等待
  ^         |                                    |
  |         v                                    |
  +----- (相似度低于阈值) ----------------------+
  +----- (人员切换，计数重置) -------------------+
  +----- (串口发送失败) -------------------------+
```

| 状态 | 说明 |
|------|------|
| **等待 (WAITING)** | 未检测到高于阈值的人脸，持续监控 |
| **匹配中 (MATCHING)** | 检测到高于阈值的人脸，正在累积连续匹配次数 |
| **已确认 (CONFIRMED)** | 连续匹配次数达到 `open_score`，确认允许开门 |
| **开门 (OPEN)** | 已向串口发送 "OPEN" 指令，正在开门 |
| **冷却中 (COOLDOWN)** | 开门后冷却期，屏蔽识别事件 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 控制面板 |
| GET | `/api/health` | 系统健康检查 |
| GET | `/api/status` | 完整系统状态 |
| GET | `/api/logs/access?limit=N` | 门禁日志（1-500 条） |
| GET | `/api/logs/system?limit=N` | 系统日志（1-500 条） |
| GET | `/api/config` | 当前配置 |
| POST | `/api/door/open` | 手动开门 |
| POST | `/api/recognition/start` | 启动人脸识别循环 |
| POST | `/api/recognition/stop` | 停止人脸识别循环 |
| GET | `/api/recording/status` | 视频录制状态 |
| GET | `/api/recording/list` | 列出已录制视频 |

## 硬件要求

- **NVIDIA Jetson**（Nano、Xavier、Orin）+ CSI 摄像头，或带 USB 摄像头的 Windows 电脑
- **串口门锁控制器**（如 Arduino），通过 USB 连接用于控制物理门锁
- **InsightFace buffalo_l 模型**（约 285 MB），用于人脸检测和识别

## 许可证

详见 [LICENSE](LICENSE)。
