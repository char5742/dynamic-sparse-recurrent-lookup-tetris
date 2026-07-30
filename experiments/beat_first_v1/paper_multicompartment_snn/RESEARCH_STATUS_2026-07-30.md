# HD-SWSNN-TwinProp 研究状況

> **2026-07-30 本線変更:** この文書は今後TwinProp/Digital Twin研究branchの
> 履歴を記録し、Tetrisモデル開発の正本ではない。CPU知能の本線は
> `../reduced_hay_direct_tetris/README.md`、すなわちTetris教師損失から
> 直接最適化するReduced Hay cellである。詳細Hay、Digital Twin、frozen
> 11-state cellは削除せず、oracle/controlとして維持する。

最終更新: 2026-07-30 16:40 JST
整理時の基準revision: `797ac31ccf5b5d5d4f9b10a447ec7fbc568582ac`

## 結論

現在の研究成果は、次の三段階に分けて扱う。

1. **Point Serial Workspace SNN**
   - 4,608点LIFノードの対照モデル。
   - CPU向けbarrierless・allocation-free学習で100kを完走した。
   - 教師順位は明確に学習したが、training-only固定panelではPreActとDSRLNを
     上回っていない。
2. **Reduced high-dimensional dendritic SNN**
   - 4 basal branch、plateau、apical、soma、adaptationからなる11状態cell。
   - graph通信のspike planeと、細胞内のcontinuous-state planeを分離した。
   - Tetrisのfull-scratch 10kは完走したが、これは詳細Hayモデルから蒸留・凍結した
     TwinProp cellではない。held-out精度優位も未確認である。
3. **HD-SWSNN-TwinProp**
   - `詳細Hay L5PC -> voltage/spike/NMDA digital twin -> XOR/parity ->
     11-state distillation -> internal freeze -> Tetris` の本線。
   - fail-closedの実装基盤は下流まで存在するが、digital twinの固定品質gateが
     未達なので、論文結果再現、11状態cellの正式凍結、Tetris 10kへの昇格は
     まだ行っていない。

したがって現時点で主張できるのは、CPU実行基盤、高次元cellの機構成立、
詳細教師生成と固定品質gate付き統合経路までである。TwinProp論文のXOR/parity
再現や、HD-SWSNN-TwinPropのTetris精度優位はまだ研究結果ではない。

## 1. Point-SNN対照の実測

対照モデルは96 block x 48点LIFノード、合計4,608ノード、110,592候補edge、
workspace top-8、4 cycle、453,815 parameterである。

100,000更新 / 800,000 teacher stateの実測:

| 項目 | 値 |
|---|---:|
| states/s | 77.812378 |
| updates/s | 9.726547 |
| 20 core平均CPU | 87.618880% |
| hot allocation | 0 byte |
| hot GC | 0.0秒 |
| loss | 5.258287 -> 2.720394 |
| top-1 | 0.023438 -> 0.523438 |
| NDCG | 0.825348 -> 0.978654 |
| pairwise | 0.500014 -> 0.843108 |

評価はtraining-only固定128状態であり、汎化評価ではない。同panelの記録値は
PreAct top-1 `0.789063`、DSRLN `0.578125`、SWSNN `0.523438`だった。
teacher state予算も揃っていないため、これは現checkpointの位置づけであり、
モデル系列の公平な最終比較ではない。

正本:

- `../serial_workspace_snn/BARRIERLESS_ARENA_PERFORMANCE_2026-07-27.md`
- `../serial_workspace_snn/README.md`

## 2. Reduced 11-state dendritic SNN

`../dendritic_workspace_snn/` は、点LIFを次のcellへ置換した独立した対照実装である。

```text
branch voltage x 4
plateau state x 4
apical/context
soma voltage
adaptation
```

96 block x 8 cell x 6 analog exportsによって、従来と同じ48次元block interfaceを
維持する。soma spikeは疎graph配送に使用し、branch/apical/somaの連続状態は
1 bitへ圧縮しない。

full-scratch 10k / 80,000 teacher state:

| 項目 | 値 |
|---|---:|
| initial single-batch loss | 6.644819 |
| final single-batch loss | 4.060909 |
| wall time | 692.668秒 |
| updates/s | 14.436928 |
| hot allocation / GC | 0 byte / 0秒 |
| firing rate | 0.040612 |
| plateau mean | 0.244637 |
| routing entropy | 0.619508 |

全compartment、graph、routing、head parameter群は変化し、4種類のlocal predictor
lossも短期窓で低下した。一方、routing entropyは設定floor `0.70`を下回り、
held-out/sealed Tetris比較は未実施である。この10kから「高次元ニューロンが
Point-SNNより高精度」とは結論しない。

この実装は能動樹状突起のCPU縮約モデルであり、詳細モデルからのdigital-twin
蒸留物ではない。HD-SWSNN-TwinProp本線とは分離して対照として残す。

## 3. HD-SWSNN-TwinPropの実装境界

### 3.1 詳細教師

正本 `neuron_hay_teacher_final.py` は、ModelDB 139653のHay L5PC morphology、
active channel、AMPA、電位依存NMDA、GABA_Aを使い、soma voltage、spike、
4領域NMDA電流、選択dendriteの電位/NMDA/Ca診断を記録する。

production contract:

```text
50,000 train + 2,000 independent held-out trials
10,000 ms / trial
dt 0.025 ms, record stride 1 ms
400 axons x exactly 20 contacts = 8,000 contacts
4,000 excitatory + 4,000 inhibitory contacts
```

公開論文はStep-1 random-driveの正確なaxon/contact規模、rate level分布、
著者のgenerator codeを公開していない。400 x 20は公開記述の曖昧さに対する
明示的な再構成解釈であり、manifestは
`fully_paper_scale_claim=false`を記録する。従ってmechanism-faithfulではあるが、
著者実装とのbyte-identical reproductionとは呼ばない。

### 3.2 development-scale digital twin

最初のM1000 digital twinは、1,024 base trial + 3,072 augmentation trialの
4,096 fit trialで学習した。held-outは開いておらず、base validation 8 trialのみを
選択に使用した。

最良checkpoint:

```text
runs/paper_elm_official_fit4096/
  staged_u1024_to_u4480_20260730T0433JST/
  stage_u2560/checkpoint_update_2560.pt

SHA-256
2614b7cd43f9e9a93c9b899414a32969e398586a84e63f02a4fb960073cc91b2
```

| gate項目 | 実測 | 必須条件 | 判定 |
|---|---:|---:|---|
| exact spike AUROC | 0.889319 | >= 0.985 | fail |
| clipped voltage RMSE | 2.565583 mV | <= 1.0 mV | fail |
| balanced spike BCE | 0.400387 | 記録のみ | - |
| NMDA normalized RMSE region 1 | 0.005013 | <= 1.0 | pass |
| region 2 | 0.439993 | <= 1.0 | pass |
| region 3 | 0.443924 | <= 1.0 | pass |
| region 4 | 0.319730 | <= 1.0 | pass |

u2816以降の追加stageはu2560を超えなかった。従ってdevelopment-scaleでは
NMDA出力はgate内だが、spikeとvoltage表現は明確に未達である。

### 3.3 下流のfail-closed chain

下流実装は存在するが、上記gateを通過したsealed production twinなしには
昇格しない。

```text
verified M1000 checkpoint
  -> immutable contract-fixed handoff
  -> detailed-NEURON XOR / 4-bit parity
  -> official-1278 distillation dataset
  -> frozen 11-state cell
  -> Pinned-V5 HD-SWSNN-TwinProp scratch 64 / 1k / 10k
```

直近の開発fixture検証記録:

| 対象 | 結果 |
|---|---:|
| final barrierless executor unit | 31 / 31 |
| real-teacher executor integration | 14 / 14 |
| contract-fixed parity bridge | 15 / 15 |
| official-1278 -> 11-state core | 13 / 13 |
| 11-state CLI | 6 / 6 |
| pinned checkpoint-lineage plumbing | 13 / 13、10 / 10、10 / 10 |

final executorではserial/parallel output一致、parameter最大差約`2e-7`、
hot allocation `1,952 byte`、hot GC `0秒`を確認した。これらは実装・配線の
検証であり、production twin品質や論文task精度の検証ではない。

## 4. 正本entrypoint

版番号付きの中間fileは、監査履歴、negative control、immutable overlayとして
保持する。新規実行は次だけを正本入口とする。

| 段階 | 正本 |
|---|---|
| Hay/NEURON教師 | `neuron_hay_teacher_final.py` |
| Windows生成wrapper | `generate_neuron_teacher_final.ps1` |
| Torch M1000 training core | `train_paper_elm_torch_cpu_variable.py` |
| fit4096 staged runner | `run_paper_elm_torch_fit4096_staged_to_u4480.py` |
| M1000 evaluator | `evaluate_paper_elm_torch_fit4096_base_val8_m1000.py` |
| immutable M1000 handoff | `TorchM1000ContractFixedHandoffV1.jl` |
| XOR/parity | `run_twinprop_parity_sealed_v2_contract_fixed_v2.jl` |
| distillation bridge | `prepare_distillation_dataset_official1278_sealed_contract_fixed_v2_final.jl` |
| 11-state distill/freeze | `run_official1278_to_eleven_state_contract_fixed_v2_final.jl` |
| final CPU executor loader | `LoadPaperArenaExecutorFinal.jl` |
| Tetris scratch lineage | `train_hd_swsnn_pinned_v5_checkpoint_lineage_final.jl` |
| CLI contract | `PinnedV5ScratchRunnerCLI.jl` |

`runs/`, datasets、NEURON shards、PyTorch/JLD2 checkpointはGit対象外である。
manifest、hash、checkpoint lineageによって外部artifactを結び付ける。

## 5. 停止したproduction教師生成

本線変更後の指示により、2026-07-30 16:40 JSTまでにproduction generator、
monitor、watchdogを停止した。生成済みshardとlogは保持している。以下は履歴
snapshotであり、現在実行中という意味ではない。

2026-07-30 11:34:09 JSTのsnapshot:

| 項目 | 値 |
|---|---:|
| status | stopped; historical snapshot |
| watchdog attempt | 1 |
| completed | 1,018 / 52,000 trials |
| progress | 1.957692% |
| observed rate | 0.112231 trial/s |
| estimated remaining | 約126.2時間 |
| verified shard pairs | 509 |
| generated bytes | 851,076,678 |
| workers | 18 |

dataset root:

```text
D:\tetris-paper-plus\data\
  hd_swsnn_twinprop_neuron_teacher_production_50k10s_fit50k_a
```

run control:

```text
runs/paper_elm_official_fit50k/
  teacher_production_generation_20260730T0854JST/
```

このsnapshotは当時進行中だった観測値であり、完了実績ではない。再開を明示
された場合はverified `.done.json` shardから再開できる。

## 6. TwinProp研究branchを再開する場合の旧critical path

以下は停止中であり、Reduced Hay direct-Tetris本線の依存関係ではない。
論文再現branchの再開を明示された場合にだけ実行する。

1. 52,000試行のproduction teacherを完了し、manifest/shard hashを確定する。
2. M1000 digital twinをproduction train splitでscratch学習する。
3. spike AUROC `>= 0.985`、voltage RMSE `<= 1 mV`、4 NMDA RMSE `<= 1`を判定する。
4. gateを通ればsealed artifactを作り、詳細NEURONでXORと4-bit parityを測る。
5. 論文taskを通過したtwinだけを11状態cellへ蒸留し、内部parameterを凍結する。
6. frozen cellをPinned-V5へ統合し、64 -> 1k -> 10k scratchを実行する。
7. 同一panel・同一予算でPoint-SNNとの精度と速度を比較する。

production dataを増やしてもspike/voltage gateが改善しない場合は、蒸留やTetrisへ
進まず、digital twinの状態表現・routing・lossを詳細モデルへ近づける。

## 7. 主張してよいこと／いけないこと

主張してよい:

- Point-SNNはallocation-freeなCPU 100k学習を完走し、教師順位を学習した。
- reduced 11-state cellはbranch-local plateau、持続電位、analog/event planeを
  実装し、Tetris 10kを学習できた。
- 詳細Hay教師からsealed twin、parity、蒸留、凍結、Tetrisへ至るfail-closed
  integration pathを構築した。
- development M1000ではNMDA gateは通過したが、spike/voltage gateは未達だった。

まだ主張してはいけない:

- TwinProp原論文のXOR/parityを再現した。
- 11状態cellが詳細digital twinを十分に蒸留できた。
- HD-SWSNN-TwinPropがPoint-SNN、DSRLN、PreActより高精度または高効率である。
- 進行中の52,000試行teacher generationが完了した。
