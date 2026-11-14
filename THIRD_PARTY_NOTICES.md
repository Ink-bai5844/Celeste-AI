# 来源与第三方组件

本仓库包含 Celeste 辅助机的 Python 搜索器、游戏桥接源码，以及保留的 NEAT 实验代码。

- `Celeste-Instrumentation/` 基于 [hdrien0/Celeste-Instrumentation](https://github.com/hdrien0/Celeste-Instrumentation)，保留原项目说明与署名。
- `Celeste-NEAT/` 基于 [hdrien0/Celeste-NEAT](https://github.com/hdrien0/Celeste-NEAT)。其中 `FileReporter.py`、`visualize.py` 保留来自 [neat-python](https://github.com/CodeReclaimers/neat-python) 的来源注释。
- 桥接构建使用 Harmony、Mono.Cecil 等依赖；依赖引用见相应 `.csproj`，各组件适用其自身许可。

Celeste 游戏程序、地图、图像、音频、个人存档和构建依赖二进制不随此源码仓库发布。使用者需要自行安装游戏，并从本机游戏生成运行副本。

本次公开不为整套代码新增统一许可证，也不替代上游组件的许可或署名要求。
