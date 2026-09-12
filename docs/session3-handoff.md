# Session 3 交接文档(2026-09-12,上下文压缩前落盘)

> 目标(用户裁决方向 A):CLI 一条命令输出回合片段+精彩集锦;跨场馆零训练 P/R 双 ≥0.90(官方口径 evaluate.py,tol 2.5s);look_at 替代人工;B站可补素材;目标完成前不退出。

## 一、素材全景(temp/)

| 组 | 文件 | 状态 |
|---|---|---|
| 原始 6 段 | 0015(B1,15.1m)/0025(B2,18.6m)/0030/0033/0034/0036 | 帧+GT(v4)+VMAE 特征全就绪 |
| **用户新增 8 段(9/12 实拍,优先级最高)** | 0007(7.4m)/0008(7m)/0009(0.9m)/0010(8.1m)/0011(8.7m)/0013(4.5m)/0014(17.9m)/0015(5.6m) | **未抽帧、未做任何处理** |
| B站场馆 D | bili\baoganghui_p1/p2(各~70m) | 帧+GT(v4,音频+24%视觉核验,视觉重建未做) |
| B站场馆 E/F/G/H | bili\yygq(139k帧)/jiguang(108k帧)/linzhou(25k帧)/yangjiang_p1-p6 | 帧已抽;GT 未做;预处理链(pipe_s3prep.ps1,PID 48608)自动跑曲线+候选+音频分诊中 |

## 二、管线与工具(全部已验证可用)

- **训练**: `tools/cuda/train_heavy.py --frames a,b --gt ga,gb --out X.pt --win 64 --epochs 6 --batch 8 --device cuda --seed N`(差分缓存 temp/work/<stem>/diffs_cache.npy 位级无损;--backbone x3d_s 已判死勿用)
- **曲线**: `tools/cuda/curves.py --frames <dir> --ckpt <pt> --tag T`(缓存复用;产物 artifacts/curves/prob_*_T.npy)
- **评测**: `tools/cuda/segment_eval.py --gt <json> --tag T [--grid]`;官方同语义
- **GT 构建(每新视频)**: ①`tools/autolabel/auto.py prepare --frames D --curves T --stem S --out temp/autolabel/S` ②`temp/audio_triage.py <mp4> <autolabel目录>` ③`temp/bg_combo.py <autolabel目录>`出3合1组合图 ④**look_at 逐组合裁定**(playing/not,多球场问近场/远场) ⑤`temp/gt_unify.py`式音频锚定边界(span内first_hit-1.2→last_hit+1.8,<3s并段)
- **VMAE 特征**: `temp/extract_vmae.py <stem>`(transformers 5.x 布局 (B,T,C,H,W);产物 vmae_feat.npy;RGB头v1弱,备用)
- **集成**: `temp/ensemble.py t1,t2,t3 outtag`(曲线插值均值)
- **融合**: `temp/fusion.py`(audio_density+energy_z+hit 簇;temp/fuse_diag.py 解剖)
- **B站下载**: `temp/bili_cookie.txt` 匿名cookie + yt-dlp(add-headers Cookie+UA+referer);分离脚本模式 `temp/pipe_*.ps1`

## 三、实验结论矩阵(铁证,勿重复)

1. **训练方差 ±0.1-0.25**(cudnn+AMP 非确定):一切对比必须 ≥3 种子;单次数字无效。E0-data 0.60/0.61 是幸运抽样,同配方均值 ~0.42/0.36
2. **3种子集成**(B1/B2/0030/0033 视觉GT训):0034 0.49/0.44、0036 0.32/0.36(tol2.5);tol8 时 0.54-0.60/0.71-0.74 → **误差~50%是边界漂移**(FP抽样3/3为真回合、边界±3-6s)
3. **GT 风格一致性>一切**:纯视觉核验GT迁移最佳;特征不可见正样本毒害(E3 0.064);音频宽边界GT毒害(LOO 0.2-0.35)
4. X3D-S+MixStyle 跨域判死;RGB冻结特征v1判弱(0.19-0.29天花板)
5. E5(全8视频)同场馆 in-sample 0.92-0.98 ✓(用户实拍同场馆场景已达标)
6. 音频看得见差分看不见的回合(0034 51/54音频可见);融合召回0.49→0.76但精度卡0.53
7. CLI 端到端已验证(修复pos[id(r)]键型bug):36片段+rallies.json+highlights.mp4+highlights_top.mp4(按分排序)+音频击球密度评分;**精彩排序双验证通过**(定量top=长+强+高击球密度;look_at判rank1对抗姿态强)

## 四、下一步计划(优先级序)

1. **用户新增 8 段处理**(最高优先,真实目标域):抽帧(fps=15 scale=960)→probe 验帧数→用 models/r3d_all6_w64.pt 出曲线→prepare+audio_triage+组合图→**look_at 全量核验**(真实用户域 GT 质量直接决定成败)→GT v4 锚定。其中选 2 段(建议 0014 最长+0010)作**新 held-out**,其余 6 段入训练池
2. 检查 pipe_s3prep.ps1 产物(8 个 B站 stem 的曲线/候选/分诊/组合图),look_at 批量核验 → GT v4
3. **E11 大训练**:B站 4 场馆(或其子集)+原 6 段+用户新 6 段 × 3 种子;LOO(0034/0036/0014/0010)多种子终审
4. 若 0.90 未达:W24 精细边界模型(边界占误差50%)+音频融合产品化(CLI 集成 fusion)
5. 终局:CLI 全链 + 精彩排序 look_at 验收 + README/台账更新
6. 遗留可选:宝岗汇 GT 视觉重建(~132组合,session2 已渲染 vcombo_*.jpg 于 temp/autolabel/baoganghui_p{1,2}/)

## 五、运行中的分离进程(压缩后检查)

- pipe_s3prep.ps1(PID 48608):B站 8 stem 预处理(曲线→prepare→triage→组合图);标记 temp/logs/session3_prep.done
- 各 pipe_*.ps1 均分离运行,终端关闭不影响;**勿用模式杀进程,只按精确 PID**(历史误杀教训×2)

## 六、关键路径速查

- 模型: models/r3d_all6_w64.pt(生产候选)/r3d_e0_s{13,42,7}.pt(集成基线)/rgb_head.pt(弱)
- GT: data/ground_truth/*.json(v4 音频锚定版,git 已提交)
- 台账: docs/eval-history.md(全部实验含失败,如实追加)
- 测试: 42 全绿(pytest -q);规约: AGENTS.md(temp/产物、venv内、全局装先问、子任务阻塞、结构化提问)
