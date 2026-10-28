<div align="center">

# Can Astra Guide a VLA to Solve RoboMME?

### 面向记忆依赖操控的自演进控制框架

Xingping Chen<sup>1</sup> · Zihao Mo<sup>1</sup> · Yimin Chen<sup>2</sup> · Yu Lu<sup>2</sup> · Zhanghui Kuang<sup>2</sup> · Guanbin Li<sup>1,*</sup>

<sup>1</sup> Sun Yat-sen University &nbsp; <sup>2</sup> TengenX &nbsp; <sup>*</sup> 通讯作者

<p align="center">
  <img src="assets/sysu-logo.svg" alt="Sun Yat-sen University logo" height="60">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/tengenx-logo.png" alt="TengenX logo" height="60">
</p>

<!-- 论文：arXiv 链接发布后补充。 -->
[![项目网站](https://img.shields.io/badge/Website-Project-217C80)](https://legend717.github.io/astra-guide-vla-website/)
[![许可证](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

[English](README.md) | **简体中文**

</div>

<p align="center">
  <img src="assets/overview.svg" alt="Astra–VLA 框架概览及 RoboMME 实验结果" width="100%">
</p>

Astra 负责规划与检查，冻结权重的官方 GroundSG VLA 负责执行；经验库为后续任务规划、
执行步数分配和失败恢复提供参考。默认的 `step-budget` 模式不依赖 JEV 模型。
仓库同时保留 `astra-only`、`jev` 和 `rules-only` 模式，供对比与恢复使用。

技术报告记录了 RoboMME 全部 **16 个任务、800 个测试回合中的 777 次成功（97.1%）**。
评测协议与完整结果见[结果说明](docs/results.md)，运行方法见[复现指南](docs/reproduction.md)。

## 快速开始

1. 按照[安装指南](docs/installation.md)在 Linux 上准备环境。
2. 获取官方 GroundSG 检查点，并在本地配置 API 凭据。
3. 按照[复现指南](docs/reproduction.md)运行单个回合或完整评测。
4. 按需按照[安装指南](docs/installation.md#offline-tests)运行离线测试。

默认分支包含控制器代码和使用文档；技术报告、项目网站及四段演示视频保存在独立的
[网站仓库](https://github.com/Legend717/astra-guide-vla-website)。
VLA 模型权重和模拟器资源需要按安装指南单独获取。详细安装与运行文档目前为英文。

克隆代码仓库：

```sh
git clone https://github.com/Legend717/astra-guide-vla.git
cd astra-guide-vla
```

## 两种经验预设

两种预设均已包含在仓库中，每种都提供全局经验及全部 16 个任务的经验文件。

| 预设 | 内容 | 默认更新方式 |
| --- | --- | --- |
| [`initial`](assets/initial-experience/) | 共享的全局指导，以及从训练演示中提取的任务时长先验 | 第 0–4 回合无论成功或失败均可更新；第 5–49 回合仅失败后更新 |
| [`final`](assets/final-experience/) | 完整评测结束后的经验快照 | 冻结，不更新 |

通过 `--experience-preset initial` 或 `--experience-preset final` 选择预设。
评测脚本会将预设复制到新的运行目录。`initial` 从训练数据提取的时长先验开始；
`final` 包含报告中的测试评测所积累的经验。使用两种预设的结果应分别报告。

## JEV 预实验

我们曾在预实验中探索 JEV，但效果未达到预期，相关代码和实验结果尚未完成全面整理。
仓库保留了这一可选接口供参考。
如果你对这些实验感兴趣，或需要了解配置细节，欢迎[提交 issue](https://github.com/Legend717/astra-guide-vla/issues)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `scripts/robomme_agent/` | 推理驱动、工具、提示词和执行调度 |
| `scripts/codex_supervisor/`、`scripts/codex_rgb_eval.py`、`scripts/codex_symbolic_eval.py` | 运行桥接代码 |
| `src/`、`packages/openpi-client/` | 适配后的 OpenPi/MME-VLA 推理依赖 |
| `assets/initial-experience/`、`assets/final-experience/` | 初始经验与最终经验快照 |
| `configs/` | 报告所用的模型配置 |
| `third_party/robomme_benchmark/` | 固定版本的官方模拟器子模块 |
| `docs/` | 安装、复现和结果说明 |

## 致谢与许可

本项目基于 OpenPi 和 RoboMME policy learning，保留了上游包名以保证兼容性。
第三方声明及适用许可见 [NOTICE](NOTICE.md)、[LICENSE](LICENSE)
和 [Gemma 许可](LICENSE_GEMMA.txt)。
