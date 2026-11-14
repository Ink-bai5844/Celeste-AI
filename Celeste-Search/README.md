# Celeste 辅助机

当前入口为根目录的 start_celeste_assist.command（macOS）或 start_celeste_assist.ps1（Windows），独立小地图选格并确认后才搜索和接管。旧入口也已转到辅助机。参见 [项目说明](../README.md)。以下是旧版开发记录，其启动参数不再适用于当前入口。

# Celeste 离线路线搜索

不训练模型，不调用大模型。程序自动循环：**进入房间 → 后台离线搜索 → 路线可用时开始执行 → 成功继续，死亡后在当前复活点自然复活再搜**。游戏和画面持续运行；搜索任务运行在独立进程中，主循环不等待结果，也不暂停游戏。无需由聊天助手逐个检索或手工指定路线。

主入口是工作区根目录的 **`start_celeste_search.ps1`**，原来的 `Celeste-NEAT` 保留。

## 快速开始

Python 3.10+，Python 部分只使用标准库，不需要 `pip install`。本地已有游戏补丁和 C# 构建依赖。在工作区目录执行：

```powershell
# 启动游戏与持续控制器：4进程后台计算，难点自动增加预算
.\start_celeste_search.ps1

# 更多后台计算时间，游戏仍然正常运行
.\start_celeste_search.ps1 -Workers 4 -PlanSeconds 5 -Retries 10

# 复用搜索失败记录；不会重放旧章节或恢复旧坐标
.\start_celeste_search.ps1 -Resume .\Celeste-Search\outputs\autoplay.json

# 搜索器更新后原地连接仍在运行的游戏；保留当前场景与已收草莓
.\start_celeste_search.ps1 -Attach -Resume .\Celeste-Search\outputs\continuous-live.json
```

默认从第一章开始。`-Map` 可选择其他本地章节文件。`-MaxSeconds 0`、`-MaxRollbacks 0` 默认持续运行，直到章节结束或按 **F12** 交还控制；Ctrl+C 也可退出控制器。退出后保留游戏和当前场景，可以人工接着玩。连接意外中断时会原地重连；明确按 F12 退出后，需要 F11 才重新允许接入。实际房间、草莓、死亡和搜索记录自动写入 `outputs/autoplay.json`。当前持续运行写入 `outputs/continuous-live.json`，每段路线和逐帧观测在其 `trace_directory` 指向的目录。

**死亡与回退**：死亡使用游戏原生逻辑，在当前房间的复活点重生，保持同一连接和会话。路线偏离预测时先释放按键，从角色实际位置重新后台计算。只有在当前房间反复找不到路线时，才请求游戏原生的房间内重试；不会重开章节、跳转出生房间或重放整段章节输入。失败候选加入黑名单，后续搜索改变动作时长和优先级。原生金莓死亡的特殊惩罚不由本程序改写，金莓目前也不作为支持目标。

搜索优先在角色自然落到安全地面、速度和输入缓冲稳定后启动。计算期间只发送松键输入，物理、死亡、复活和画面都继续更新。结果返回时再次检查位置、出生编号、冲刺数及草莓状态；过期结果丢弃，不能把旧路线直接套到已经变化的角色身上。等待计算时角色可能站着不动，这不是暂停游戏。

如果角色在弹簧上循环弹跳，程序会在后台预测未来的输入帧，从那一帧计算脱离路线，并在实际帧、位置、速度、私有计时器及亚像素余量一致时接入。搜索结果来晚或状态不符就重新计算。不会为了等待落地而永远卡在弹簧上。

探索记录区分实际走过的区域、站稳的房间、当前目标和没有进展的重复穿门。跨房间连续动作由同一条已计算路线执行，保留切版跳与必要的回穿，不设置禁止回到上一房间的规则。飞翼草莓拾取之前禁用冲刺；已经逃走的草莓需要当前房间原生恢复。可破坏墙是明确的探索目标，只有游戏实体移除或 `DoNotLoad` 确认后才记录撞开；假墙、遮挡墙作为隐藏区域探索处理。

原先会阻塞游戏或重开章节的控制器已从主程序移除。旧版 `celeste-autoplay-checkpoint-v1` 检查点不会载入；新格式是 `celeste-live-search-v1`。

以下入口是**单独离线计算/检查工具**，无需启动游戏：

```powershell
# 第一章，4 个进程分别搜索 120 秒，保存最佳连续路线和候选路线
.\run_offline_search.ps1 -Seconds 120 -Workers 4

# 汇总所有地图、房间、草莓种类与不支持的机关
.\run_offline_search.ps1 -Mode catalog

# 各章独立后台批量计算；不会启动游戏
.\run_offline_search.ps1 -Mode batch -Seconds 120 -Workers 4
```

默认结果位于 `Celeste-Search/outputs/route.json`，同名 `.html` 是可缩放的地图/轨迹报告。HTML 不需要联网。候选 `.candidate-N.json` 是各进程独立找到的完整路线，程序不会把不相容的中间状态拼在一起。

```powershell
# 从章节起点搜索到房间 2
python .\Celeste-Search\search.py plan --target room:2 --seconds 30

# 明确的草莓目标（房间名:地图实体 ID）
python .\Celeste-Search\search.py plan --target berry:2:11 --seconds 120

# 继续计算已保存路线；先在模拟器中重放原有按键，再向前搜索
python .\Celeste-Search\search.py plan --resume .\Celeste-Search\outputs\route.json --seconds 300

# 单房间练习搜索；这不算从章节起点走到该房间
python .\Celeste-Search\search.py plan --room 2 --target berry:2:11 --seconds 60

# 纯离线复算，校验地图哈希、存活及最终状态
python .\Celeste-Search\search.py verify .\Celeste-Search\outputs\route.json
```

`--seconds` 是每个候选进程的墙钟预算；`--expansions` 是每个目标最多展开的状态数；`--max-goals` 控制搜索阶段数。默认目标依次偏向当前房间草莓、相邻未到达房间、局部地面探索点。未达到目标保留为 pending；超时不等于证明不可达。`--allow-unsupported` 仅供研究未建模机关，所得路线禁止自动回放。

## 搜索与物理

这是状态图搜索，而不是学习策略。节点包含位置、速度、冲刺次数、体力、输入边沿、跳跃/冲刺/收集计时器、已收集和跟随草莓、破坏块与活动平台状态。边是若干帧的方向、跳跃、冲刺或攀爬组合。加权优先队列使用地形距离提示扩展候选，草莓附近额外搜索松键落地动作，避免把触碰草莓误当完成。

模型处理 8 像素地砖、亚像素位移、站立/蹲下碰撞、加速、跳跃宽限和缓冲、可变跳高、八向冲刺、墙跳与攀爬、尖刺、单向平台、弹簧、补充晶体、普通/飞翼草莓，以及第一章常见的冲刺破坏块、坍塌平台、绳索平台和下落方块。

**模拟器不是原游戏引擎的逐位等价副本。** 拐角修正、移动方块携带与推挤、转场、机关更新顺序、飞翼草莓动画等仍有近似。风、水、梦境块、钥匙门、脚本事件和特殊收集机制尚未完整实现。`capability_report()` 列出不支持的内容；默认不通过这些房间。金莓、种子莓和月莓不会被虚报成已收集。

地图目录中的房间总数只是清单。HTML 报告的房间/草莓是**离线预测**；实时日志 `autoplay.json` 的 `confirmed_berries` 来自游戏确认。触碰后仍跟随的莓不算确认收集。

## 单条路线回放工具

除了上述持续控制器，也可手工回放导出的单条路线。已有的 `GAME/Celeste-1` 是带 instrumentation 的 XNA 游戏副本；回放需要 .NET SDK、.NET Framework 4.5 目标框架及 XNA 引用。此电脑上已有这些依赖。

```powershell
.\play_search_route.ps1 -Route .\Celeste-Search\outputs\route.json
```

脚本编译新的观察/输入桥，备份选定游戏目录原有 DLL，启动该副本播放**一次**。配置临时设为正常速度，退出时恢复；游戏保持打开并交还键盘控制。不会对普通存档进行合并，延用原接口隔离的 666 号存档会话。回放通过正常按键移动，不写坐标、不授予草莓。

桥在游戏读取虚拟输入之前接受每帧指令，原游戏的冲刺冻结与转场期间不消耗新指令。后台搜索不阻塞这条通道；桥有响应超时保护，控制器失联会交还人工控制。单条回放工具遇到偏差停止；持续控制器则从实际状态继续后台搜索，死亡交给原生房间复活。两者都不训练模型。

也可连接手动启动、端口已配置好的游戏：

```powershell
python .\Celeste-Search\search.py play .\Celeste-Search\outputs\route.json --port 25101
```

桥协议说明位于 `Celeste-Instrumentation/CelesteInstrumentation/PLAYBACK_PROTOCOL.md`。旧 137 浮点训练接口仍可使用，只有新客户端设置 `SearchMode=true` 时采用新协议。

## 验证

2026-11-12 持续运行版已实际走到多个后续房间和支路，确认收集 `2:11`、`3:9`，并实际撞开 `5:43`。这些进度来自同一游戏进程中的正常操作；Python 更新通过原地重连接入，没有重开章节。最新实时进度以 `outputs/continuous-live.json` 为准。早期仅到第二房间的40秒记录 `outputs/live-run.json` 已被后续验证超越，不能作为当前进度。仍未完成全章、全地图或全收集验证。

离线回归覆盖数值精度、攀爬动作顺序、跨房间输入、飞翼草莓禁冲刺、破墙确认、目标持续性和弹簧未来帧接入。已保存实跑输入可在后台复算，不用游戏反复试候选。状态复制优化在同一房间快照上提高约1.8倍模拟步吞吐，详见 `../artifacts/planner-profile/state-copy-benchmark.json`。

```powershell
Push-Location .\Celeste-Search
python -m unittest discover -s tests -v
Pop-Location
dotnet build .\Celeste-Instrumentation\CelesteInstrumentation\CelesteInstrumentation\CelesteInstrumentation.csproj -c Release
dotnet run --project .\Celeste-Instrumentation\CelesteInstrumentation\tests\SocketProtocolTests\SocketProtocolTests.csproj -c Release
```

测试覆盖地图二进制格式、实际离线搜索、碰撞和机关、草莓确认、逐房间规划、后台任务不阻塞、过期结果拒绝、原生复活状态机、TCP 分包/断连、按键边沿和报告转义。通过这些测试不代表完整游戏物理已经一致，也不代表所有章节已经通过。

## 源码依据

- [官方公开的 Player.cs](https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs)：移动常量及角色逻辑。
- [Celeste 反编译源码仓库](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled)：地图格式、关卡、机关与会话。
- [BinaryPacker.cs](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/BinaryPacker.cs)、[LevelData.cs](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/LevelData.cs)：地图解析与坐标。
- [Strawberry.cs](https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Strawberry.cs)：跟随与安全地面确认收集的区别。

本目录独立实现读取器、模拟器和搜索程序，不包含下载的完整反编译代码或游戏资源。游戏资源从现有本地安装读取。

