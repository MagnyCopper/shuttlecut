# ShuttleCut v2 重设计:零训练 P/R ≥0.90 + 精彩度排序

> 日期:2026-09-09 | 状态:待用户终审
> 前置调研:librarian 带出处核验(2026-09-09,见文末精选参考)
> 上下文:`docs/history/2026-09-08-windows-handoff.md` + `docs/eval-history.md` 全部 12 节
> 修订(2026-09-09 晚):用户裁决删除全部旧权重(含 joint2)。预标注器与 E0 对照改由首个 v2 联合模型承担;历史基线 0.818/0.964 以 eval-history 记录为准(曲线可自 git 历史 0ccc29a 找回)。

## 1. 背景与目标

**现状**(joint2,B1+B2 联合训练):
- in-domain:B1 P=0.957/R=0.918;B2 P=0.984/R=0.968
- 零训练跨域(0037):P=0.818/R=0.964/MAE=0.02s——R 已过线,**P 差 0.082**
- 已证伪:异质数据天真混训(0031 两实验 0.818→0.414/0-0)、单帧/逐帧特征/球检测/音频修剪投票/视觉否决/滞回归调参

**总目标**:新视频**零训练**(不训练不标注)段级 P/R 双 ≥0.90(唯一口径:`src/shuttlecut/eval/evaluate.py`,tol 2.5s/重叠≥0.5×较长段),并输出**精彩度排序集锦**。

**需求画像**(用户 2026-09-09 确认):
1. 推理期严格零训练零人工;训练期投入不设限,欢迎重新设计训练方案
2. 训练素材走**混合风格**(多场馆/机位/光线)——泛化与域异质性是同一枚硬币
3. 算力:RTX 2070 8GB(Windows/训练)+ M4 Mac mini 16GB(MPS/推理与辅助),无云算力
4. 范围:回合剪辑 + 精彩度排序;不做比分统计

## 2. 设计原则

1. **官方评测是唯一裁判**:所有实验(含失败)如实追加 `docs/eval-history.md`
2. **新数据入训必须先体检**:零训练曲线段内/外对比度 + 边界陡度(0031 纪律)
3. **根因驱动迭代**:本设计是第一次尝试的框架。遇到问题/瓶颈时,先做根因分析(曲线解剖/误差分层/可视化审计),证据指向框架性缺陷时**果断切换框架或修复迭代**,不恋战、不绕过,直到完成目标
4. **解耦**:切分(是否回合)与排序(是否精彩)分离;特征提取与训练分离(缓存复用)
5. **三轨互为对照与备件**:单轨失败不阻塞其余两轨

## 3. 总体架构

```
                    ┌─ 轨 A:差分输入 + X3D-S + MixStyle + 滞回切分(工程保底,E1-E2)
素材三池 ──→ 共享底座 ─┼─ 轨 B:DINOv2 冻结特征 + Transformer 时序头(泛化赌注,E4)
(帧/特征缓存)      └─ 轨 C:TriDet 检测头(边界升级件,条件触发,E3)
                              ↓ 胜出模型
                    精彩度 ranking head(E6)→ 排序集锦出片
```

共享底座:15fps 帧缓存(`temp/work/<video>/frames15/`,现有)、特征缓存(新 `temp/features/<video>/`)、素材登记表(`data/videos.jsonl`)、自主标注管线(`tools/autolabel/`)、统一评测。

## 4. 数据战略(三池)

| 池 | 内容 | 入训方式 |
|---|---|---|
| ① 核心标注池 | B1/B2 + 用户新拍混合素材 | 自主标注(模型预标→9帧视觉+音频分诊→用户只复核疑点段),精标 GT |
| ② 弱标扩展池 | **B站自录羽毛球全场合集 + GitHub 开源数据集(BFMD 19场20.3h/1687回合,机位待验证;FineBadminton 预切片仅辅助)。不用 YouTube** | joint2 零训练预标 + 曲线体检 + 人眼抽查(10%)后入训 |
| ③ 无标注特征池 | 大量公开素材不求标注 | 只提特征缓存;自有域 SSL 预训练列为 E7 最后手段 |

- **素材登记表** `data/videos.jsonl`:每段登记 id/来源/机位/场馆/许可/域标签,驱动分组采样与留一验证
- **vlog 素材许可纪律**:仅本地研究训练用,不入库不分发
- **小样验证流程**(用户已批准先小样):B站拉 3-5 段自录全场 + BFMD 下载验证 → joint2 零训练曲线体检 → 判据:段内/外均值对比度不低于现有水平 + 人眼抽检吻合 → 通过后批量
- 采集工具落工程 venv 内(B站下载器如 you-get/yt-dlp 仅用于 B站)

## 5. 轨 A:渐进域扩展(E1-E2,工程保底)

- 骨干:R3D-18 → **X3D-S**(3.8M 参数,Kinetics-400 预训练,pytorchvideo checkpoint 现成)
- 输入不变:补偿帧差 64 帧窗 @15fps stride 2(与全部历史结果可比)
- **MixStyle**(Zhou et al., ICLR'21):插在中后两个 block,训练期 p=0.5 随机激活,混合不同视频的特征统计——直接针对负迁移根因(域统计被迫共享)
- 训练:`tools/cuda/train_heavy.py` 扩展 `--backbone x3d_s --mixstyle`;场馆分组采样;`--device cuda`(2070)
- 预期:最快产出 held-out 数字;显存富余可同步上 X3D-M 对照

## 6. 轨 B:基础表征时序头(E4,泛化赌注)

- **DINOv2-S 冻结**(facebookresearch/dinov2,torch.hub 权重落 `models/dinov2/`),每 2 帧取 1 帧(≈7.5fps),CLS+空间池化 384 维帧向量,缓存 npy
- 时序头:2 层 Transformer encoder(d=256,4 heads,窗口 64 帧)→ 逐帧 actionness logits
- 依据:DINOv2 帧特征跨视角稳健(ECCV'24 跨视角时序分割)+ 13 种融合对比中 self-attention 最优(arXiv:2407.15605)
- 只训 head:2070 分钟级/epoch,M4 可并行实验
- 切分先沿用滞回(可比),轨 C 可无缝接管边界
- 风险预案:若帧特征丢快速挥拍动态,降级为轨 A 的辅助特征(拼接差分特征)

## 7. 轨 C:TriDet 检测头(E3,条件触发)

- **触发条件**:A/B 曲线质量好但官方口径 miss/extra 集中在边界(MAE>0.5s)
- 接入:OpenTAD(third_party),特征=胜出骨干的缓存输出,TriDet 头(Shi et al., CVPR'23;算力约 ActionFormer 47%)直接回归起止边界
- 特征模式运行,8GB 无忧;GT json→TAD 格式转换脚本随做
- 不作第一优先:TAL 头惯例在大数据集训练,~15 视频有过拟合风险,故仅在边界成为瓶颈时启用

## 8. 精彩度排序(E6)

- 与切分**完全解耦**:对切好的 rally 段做段级评分
- **v1 无监督规则加权**:回合时长、actionness 峰值/方差、运动能量、(可选)击球瞬态密度——只用场地声特征,不 veto、不参与切分
- **v2 学习版**:用户对 top/bottom 各标约 20 段,训轻量 MLP pairwise ranker
- 输出:`highlights.mp4` 按分数排序,每段附分

## 9. 评测协议(铁律)

1. 唯一口径:官方 `evaluate.py`(tol 2.5s/重叠≥0.5×较长段)
2. **held-out 留一**:核心池留 1-2 段 + 弱标池留 2-3 段永不入训,作主指标;vlog 角度单独作为最难域报告
3. 每实验记录:held-out P/R、边界误差分布、miss/extra 归因、rally 长度分层、训练显存峰值、单视频推理耗时
4. 失败也入 `docs/eval-history.md`,注明根因与去向

## 10. 工程落地

**代码结构**:
- `tools/cuda/train_heavy.py`:扩展 --backbone(x3d_s/r3d_18)与 --mixstyle
- 新 `src/shuttlecut/features.py`:骨干特征提取与缓存(X3D/DINOv2 统一接口)
- 新 `src/shuttlecut/rank.py`:精彩度评分与排序集锦
- 新 `tools/data/registry.py`:素材登记表读写与体检
- 新 `tools/data/fetch_bilibili.py` + BFMD 校验脚本
- `third_party/OpenTAD`(触发轨 C 时引入)

**双机分工**:
- RTX 2070(训练机):A/B/C 训练 + 大批量特征提取
- M4 mini(使用机):推理出片 + autolabel 视觉分诊 + 日常零训练处理

**里程碑**:
- M0(1-2 天):素材登记表 + B站/BFMD 小样验证 + 用户新素材导入与体检
- M1(1 周内):轨 A 首个 held-out 结果(vs joint2 基线)
- M2(并行):轨 B 首轮 held-out
- M3:胜出者 + 按需触发轨 C + 精彩度 v1
- M4:全自主端到端 demo(新视频零训练出片+排序集锦),eval_history 定格

**实验阶梯**(survey 推荐顺序):
```
E0 joint2 基线按场馆分组重跑(校准对照)
E1 X3D-S + 原滑窗        E2 E1+MixStyle
E4 DINOv2+时序头          E3 TriDet(条件触发)
E6 精彩度 ranker          E7 自有域 SSL(最后手段)
```

## 11. 风险与止损

| 风险 | 止损 |
|---|---|
| A/B 都卡 0.85-0.90 | 触发轨 C 边界升级;仍不行→E7 SSL 预训练;再不行→按设计原则 3 根因分析后果断换框架 |
| vlog 域差拉崩训练 | 课程学习:核心池先收敛,弱标池低 lr 低权重二阶段加入;体检不过的源剔除 |
| DINOv2 丢动态 | 轨 B 降级为轨 A 辅助特征 |
| 精彩度 v1 不理想 | 直接进 v2 学习版(用户标注成本 40 段) |
| TriDet 小数据过拟合 | 冻结特征 + 强增广 + 早停;或退回滞回+边界校准 |

## 12. 精选参考(全部出处核验)

- MixStyle: Zhou et al., ICLR'21 — arxiv.org/abs/2104.02008;github.com/KaiyangZhou/Dassl.pytorch
- X3D: Feichtenhofer et al., CVPR'20 — arxiv.org/abs/2004.04730;github.com/facebookresearch/pytorchvideo
- TriDet: Shi et al., CVPR'23 — github.com/dingfengshi/TriDet
- OpenTAD: github.com/sming256/OpenTAD
- DINOv2: Oquab et al. — arxiv.org/abs/2304.07193;github.com/facebookresearch/dinov2
- 跨视角 DINOv2 时序分割: ECCV'24 — ecva.net/papers/eccv_2024(09116-supp)
- 时序融合 probing: arxiv.org/abs/2407.15605
- VideoMAE: Tong et al., NeurIPS'22 — github.com/MCG-NJU/VideoMAE
- BFMD(羽毛球数据集): CVPRW'26 — arxiv.org/abs/2603.25533;github.com/Ning-D/BFMD
- STDN(视频域泛化): arxiv.org/abs/2310.17942

**证据薄弱声明**:无公开证据表明任何方法能在 5-15 个混合域羽毛球视频上直接达成 P/R≥0.90;本设计即以实验阶梯自行验证,失败即记录并迭代。
