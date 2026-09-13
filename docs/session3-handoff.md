# Session 3 终局交接(2026-09-13 深夜)

## 本 session 完成与判负(全部已入台账 docs/eval-history.md)
- 用户 8 段实拍全流程 GT(网格 222 判 + 谷 49 判 + 条带 66 判),GT v4 回合级定稿并提交
- E11(粗GT)/E12(细GT) 各 3 种子:0034/0036 停在 0.36/0.44(集成);u0014/u0010 远未及线
- 纯视觉 R3D 天花板 ~0.45 实证;音频共分割 v1 判负(瞬态噪声 19-53%,聚簇失效)
- 差分谷拆分判死(停顿与静默差分分布重合)

## 下一步(按杠杆排序)
1. **击球声分类器**:audio_transients 现把聊天/脚步全算 hit(回合外 19-53%)。用频谱特征(拍击=宽带 2-8kHz 陡衰减)训 SVM/小 CNN;正样本=GT 回合内瞬态,负样本=回合外。分类后重跑 coseg.py(门控+聚簇参数已就位)→ 若 P/R>0.7 则产品化进 CLI。
2. **边界回归头**:R3D 曲线 + 双向峰值搜索(边界=局部概率拐点而非 0.5 阈值),或 W24 精细窗。
3. B站 4 场馆 GT(帧库就绪,曲线链 PID 19508 自动跑完)——扩大训练池前先让范式在 DJI 域闭环。
4. 验收口径悬而未决:tol2.5s 与 look_at GT 采样分辨率(±1.5-3s)的张力,若范式换新后仍卡 0.6-0.7,需与用户对齐 GT 精度方案(音频锚定细化到 1s)。

## 运行中
- pipe_biliresume2.ps1(PID 19508):B站 8 stem 曲线+分诊(等 e12.done 已满足,自动执行中)
- E11/E12 模型与曲线已落盘;全部 GT 已提交(HEAD 6259e30)

## 关键命令
- 评测: .venv\Scripts\python.exe tools\cuda\segment_eval.py --gt data\ground_truth\<stem>.json --tag <tag>
- 共分割: temp\coseg.py(参数 GAP/MINH/THR 在头部)
- 条带渲染/终版: temp\render_strips.py / temp\finalize_gt4*.py
