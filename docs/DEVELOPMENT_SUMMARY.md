# FLD 解码与生成项目 — 开发总结

本文档总结本仓库的目标、架构、关键技术决策，以及 **ex4_e（cellular_phone）** 案例中从 SDAT 生成 FLD 并被 scPOST 读入的完整攻关过程。

---

## 1. 项目目标

### 1.1 核心目标

1. **解析** Software Cradle scFLOW 的 **FLD**（`CRDL-FLD`）二进制场文件。
2. **转换** 为 CGNS/HDF5，布局对齐官方 `FLDUTIL` 导出（如 `ex1_e_100_orig.cgns`）。
3. **生成** 初始 FLD：仅依赖 **SDAT (.s)** + **EMT (.xemt)**，**无需** 求解结果文件 `.r` 或同网格模板 FLD（模板可选，用于提高 scPOST 兼容性）。

### 1.2 验收标准

**ex4_e（手机散热）**

| 指标 | 官方 `ex4_e_63.fld` | 本仓库生成（目标） |
|------|---------------------|-------------------|
| 单元数 | 1,470,392 | 一致 |
| 材料 bincount | 7 种材料分布 | 一致 |
| 体段名 | `PARTS1…PARTS32` + 32 部件名（64 个体段） | 一致 |
| 顶点数 | 1,619,610 | 约 1,618,403（界面节点策略差 ~1,207） |
| 表面四边形 | ~956,447 | ~530,737（网格/界面策略差异） |
| scPOST | 正常打开、显示 32 部件树 | 可读入并完成 Relocating volumes |

**ex3_e（室内空调）**

| 指标 | 官方 `ex3_e_151.fld` | 本仓库 `ex3_e_from_sxemt.fld` |
|------|----------------------|-------------------------------|
| 单元数 | 101,548 | 122,206（SDAT 网格与 step-151 不同，预期） |
| 体段名 | 10 个 256B 槽位 | 与官方一致（自模板读取） |
| `LS_VolumeGeometryArray` block1 | 5 桶 `[main,p2,p3,p5,p6]` | 同结构，main = n_cells − tail |
| scPOST | 正常 | **已验证可读入**（2026-06） |

官方 FLD 含求解步结果；`.s` + `.xemt` 生成文件场量为初值/默认常数，可接受。

---

## 2. 系统架构

```
.s (SDAT)          .xemt (XML)
    │                    │
    ▼                    ▼
 s_model.py          xemt_model.py
    │                    │
    └────────┬───────────┘
             ▼
      mesh_builder.py  ←── surface_builder.py
             │
             ▼
      fld_writer.py ──→ .fld
             │
             ▼
      fld_model.py ──→ fld2cgns.py ──→ .cgns
```

### 2.1 模块职责

| 模块 | 职责 |
|------|------|
| `s_model.py` | 解析 SDAT：CXYZ 间距、`PARTS` 盒区域、初边值、环境参数 |
| `xemt_model.py` | 解析 `.xemt`：材料/部件名、分组；生成 `volume_names` 列表 |
| `mesh_builder.py` | 结构化六面体网格：材料分配、MPI I 向分裂、界面节点复制 |
| `surface_builder.py` | 边界面分类（ENT/PARTS/外边界），生成 `surface_cats` |
| `fld_model.py` | FLD 二进制解析、面列表与 BC 重建 |
| `fld_writer.py` | FLD 写出：场量节 vendor 布局、几何节、SDAT 嵌入 |
| `fld2cgns.py` | FLD → CGNS（单 Zone、NGON 面、FlowSolution） |
| `sxemt2fldcgns.py` | 端到端：`.s` + `.xemt` → FLD + CGNS |
| `s2fld.py` | SDAT → FLD（支持模板补丁模式 `compose_fld`） |
| `fld_parser.py` | 命令行查看节布局 |

### 2.2 两种 FLD 写出策略

**A. 模板补丁（`compose_fld`）**

- 复制已有 FLD，仅替换场量数组与 `LS_SFile` 内 SDAT 文本。
- 要求网格与模板 **完全一致**（节大小不变）。
- 适合：同网格换初值/时间步。

**B. 自网格写出（`write_fld_from_mesh`）**

- 从 `vertices / cell_conn / material` 完整构造所有节。
- 可选 `template_fld`：复制文件头、`LS_STREAM*`、几何前导区、Volume/Surface 部分二进制块。
- ex4 默认自动选用 `tests/ex4_e_63.fld` 作模板（当 `n_cells == 1470392`）。
- ex3 默认自动选用 `tests/ex3_e_151.fld`（`resolve_geometry_layout_fld` 按 stem 匹配）；用于体/面几何节布局与 vol-flag 模板，**不要求**单元数一致。

---

## 3. 网格构建关键技术

### 3.1 CXYZ 结构化网格

SDAT 中 `CXYZ` 给出 I/J/K 方向节点间距累加，生成物理坐标 `x[], y[], z[]`。单元为结构化六面体，`ni×nj×nk` 个 cell。

### 3.2 PARTS 盒区域

每个 SDAT `PARTS` 可含多个盒 `(i1,i2,j1,j2,k1,k2)`，支持 `/` 分隔多区域。单元中心八顶点均在盒内则划入该部件，赋予 `part_id` 与 `material_id`。

### 3.3 多材料界面节点复制

vendor 做法：在材料交界面上，**同一结构化节点可为不同材料各保留一个顶点 ID**（最多 7 份，对应材料 1–7）。

实现（`mesh_builder._assign_node_ids`）：

1. 对每个结构化节点，收集相邻单元材料集合。
2. 为每种材料分配独立全局顶点 ID。
3. **I 向 MPI 分裂**（`I_SPLIT=30`）：与官方 ex1/ex4 分区布局一致，在 `i=30` 处复制节点。

此策略导致顶点数略少于官方（官方可能在部分界面采用不同复制规则），但单元数与材料分布一致。

### 3.4 体区域命名

`volume_names_from_parts`：

```text
PARTS1, PARTS2, …, PARTS{n}, 部件名1, 部件名2, …
```

ex4：`n=32`，共 **64** 个字符串写入 `LS_VolumeGeometryArray` block0（16384 字节 ASCII）。

**ex3 特例**（10 个体段，非 PARTS1…PARTS{n}+部件名列表）：

- block0：**10 × 256 字节** ASCII 槽（共 2560 B），名称自模板 `ex3_e_151.fld` 读取。
- 名称示例：`PARTS1, PARTS2, PARTS3, PARTS5, PARTS6, Interior, Table, Table2, Wall, Window`（无 PARTS4/PARTS7 槽）。
- block1：5 桶 `[main, part2, part3, part5, part6]`，main = `n_cells − (p2+p3+p5+p6)`；part4 计入 main，不单独占 tail 桶。

### 3.5 表面分类

`build_vendor_surfaces` 扫描：

- 域外边界 → `Xmin/Xmax/…`
- 材料/部件变化 → `ENT*`、`PARTS`、`SURFACE`

`vendor_bc_plan_from_categories` 生成与 `fld_model._build_face_list_and_bcs` 对称的 BC 计划，供 CGNS `ZoneBC` 使用。

---

## 4. FLD 写出攻关（scPOST 兼容性）

scPOST 对二进制布局校验严格；仅“语义正确”的数组不足以打开文件。以下按发现顺序列出。

### 4.1 场量节前导/尾区

**现象**：报 “FLD File may be broken”。

**原因**：每个场量节需 48 字节 preamble、f64 间 16 字节 chunk、48 字节 trailer、`LS_Scalar:*` 标签块。

**修复**：`fld_writer._field_section_preamble`、`_field_section_trailer`、`_write_linked_section_end` 等。

### 4.2 `LS_Nodes` / `LS_Elements`

**要点**：

- 80 字节 mesh preamble（非 48 字节）。
- X/Y/Z 间 `12,8,n_vertices,1`；Z 后 20 字节 tail。
- `LS_Elements`：meta 块(38) + sep + 连通长度 + sep + 扁平连通。

### 4.3 `LS_VolumeGeometryArray` 前导区偏移错误

**现象**：`Fatal Error (TITLE = LS_VolumeGeometryArray)` at offset ≈ 节起始 + 156。

**原因**：`_geometry_preamble_from_template` 将 `first_block_bc`（16384）写入字节偏移 **100**，正确位置为 **104**（块头 `12, 1, bc, 1` 的第三字段）。

| 偏移 | 正确值 | 错误写入后 |
|------|--------|------------|
| 100 | `1` | `16384` |
| 104 | `16384` | `16384` |

错误序列：`12, 16384, 16384, 1` → scPOST 在 `POST_read_APX` 失败。

**修复**：`pack_into(..., 104, first_block_bc)`。

### 4.4 `LS_VolumeGeometryArray` block2（体标志 / VolFlag）

**现象**：通过 Fatal Error 后，在 **“Relocating volumes”** / `CreateVolFlag64_2` 挂起或异常退出。

**原因**：每单元 8 字节体标志块全零或编码非法；官方为成对 `>i4` 索引（供 `CreateVolFlag64` 建体区域映射）。**`lo` 不得超过 `n_cells−1`**（官方 `lo_max = n_cells−1`）。

**修复策略**（`fld_writer._generate_vol_flag_buckets`）：

| 场景 | 做法 |
|------|------|
| 单元数与模板一致 | **整块复制** block2 |
| ex3 风格（模板 stem 匹配且 `count_val=10`） | 见下表 |
| 其他（如 ex2 + ex4 模板） | `_scale_entire_vol_flag_block` 整块 resample |
| 无模板 | 按 block1 桶线性 `+2` 回退（易超界，不推荐） |

**ex3 block2 分桶生成**（模板 `ex3_e_151.fld`）：

| 桶 | 单元数 | 生成方式 |
|----|--------|----------|
| 0 main | `n_cells − tail` | 按模板 bucket0 **比例缩放** lo/hi，保留 MPI 换码；`lo ≤ n_cells−1` |
| 1–2 | part2、part3 | 自 bucket0 末尾 **线性 +2** 链接 |
| 3 | part5 | 模板 bucket3 **整体 + offset**（offset = 本文件 bucket2 末 hi − 模板 bucket2 末 hi） |
| 4 | part6 | 与模板 **逐对相同**（count=300 时） |

**材料 ID**：导出时将 `LS_MatOfElements` 中的 **0 → 1**（SDAT 中 `_domain` / `Air_Conditioner` 等可为 0；vendor FLD 不含 0）。

**网格顺序**：`LS_Elements` / `LS_MatOfElements` 须与 vendor 一致——`cmat.reshape(-1)` 的 C 序与 k-j-i 遍历不同；ex2/ex3 等用 **MPI 两 I 块**（`I1=ni-ni//2`，先 i∈[0,I1) 再 i∈[I1,ni)）；ex4 用整行 k-j-i。节点编号在 `I1` 处分区（非固定 30）。

### 4.5 `LS_SurfaceGeometryArray`

**要点**：

- 112 字节几何前导区（同 Volume，注意偏移 104）。
- block0(4608)、block4(72) 宜从模板复制。
- block5 之后 **link 区**（BC 名称等）从模板复制至节尾前 20 字节；**仅当模板 stem 与 SDAT 案例 stem 一致**（ex2 用 ex4 模板时不复制 link，仅 6 个 surface 数据块 + 节尾）。

### 4.6 `LS_SFile` / `OverlapEnd`

- SDAT 文本嵌入 `LS_SFile` block1；UTF-8、LF、`SDAT` 前缀。
- `OverlapEnd`：仅 40 字节空节头，无 inner 数据块。

### 4.7 文件前缀

必须包含 `CRDL-FLD` 魔数（`_fld_prefix`），否则 scFLOW/scPOST 不识别。

---

## 5. 解析与 CGNS 导出

### 5.1 解析流程（`parse_fld`）

1. `LS_Nodes` → `vertices (n,3)`
2. `LS_MatOfElements` + `LS_Elements` → `material`, `cell_conn (n,8)`
3. `LS_VolumeGeometryArray` block0 → `volume_names`
4. `LS_SurfaceGeometryArray` → `faces`, `bc_plan`（复杂 seg1/seg2 重组）
5. 各场量节 → `fields` 字典

大文件（>512MB）使用 `mmap`（`open_fld_buffer`）。

### 5.2 CGNS 结构（`fld2cgns.py`）

- Base → Zone `FluidZone`
- `GridCoordinates` R8
- 多个 `Elements_t`：按材料/部件分的体六面体 + `GridElements_Faces` NGON
- `FlowSolution`：顶点中心场
- `ZoneBC`：由 `bc_plan` 生成

验证：`tests/test_ex1_e_100.py` 与 `ex1_e_100_orig.cgns` 对比。

---

## 6. 命令行用法速查

```bash
# SDAT + xemt → FLD + CGNS
python sxemt2fldcgns.py tests/ex4_e.s tests/ex4_e.xemt -o tests/ex4_e_from_sxemt.fld

# ex3 室内空调例（scPOST 已验证）
python sxemt2fldcgns.py tests/ex3_e.s tests/ex3_e.xemt -o tests/ex3_e_from_sxemt.fld --verify-parse
python tests/compare_ex3_fld.py
python tests/test_ex3_mesh.py

# FLD → CGNS
python fld2cgns.py tests/ex1_e_100.fld -o out.cgns

# 查看节布局 + scPOST 几何检查
python fld_parser.py tests/ex3_e_from_sxemt.fld --validate-scpost

# 模板补丁模式
python s2fld.py tests/ex1_e.s --template tests/ex1_e_100.fld -o out.fld

# 测试
python -m pytest tests/test_sxemt.py -q
python tests/test_ex4_mesh.py
python tests/test_ex3_mesh.py
python tests/test_ex2_mesh.py
```

---

## 7. 测试与验证

| 测试 | 内容 |
|------|------|
| `test_ex1_e_100.py` | CGNS 与官方参考逐数据集对比 |
| `test_sxemt.py` | ex1 `.s`+`.xemt` 管线冒烟 |
| `test_ex3_mesh.py` | ex3 scPOST 几何、10 体段名、block1、vol-flag 边界 |
| `test_ex2_mesh.py` | ex2 409188 单元、vol-flag 边界、validate_scpost_geometry |
| `compare_ex3_fld.py` | ex3 官方 vs 生成 FLD 节布局与 vol-flag 对比 |
| `test_ex4_mesh.py` | ex4 单元数、材料、体段名、与官方 FLD 解析对比 |
| `test_s2fld.py` | 模板补丁写出 |

**scPOST 验证**（手动）：打开生成 FLD，确认无 “broken”、Relocating volumes 完成。ex3/ex4 分别确认模型树与体段名。

---

## 8. 已知限制与后续工作

### 8.1 顶点与表面面数差异

- 顶点少 ~1,207：界面节点复制规则与 vendor 未完全对齐。
- 表面四边形少 ~40%：与顶点/界面处理相关，可能影响部分 BC 面完整性。
- **不影响** 单元数、材料分布、体段命名树。

### 8.2 体标志 block2

- **单元数与模板一致**时整块复制最可靠（ex4）。
- **ex3** 在单元数不同时按 §4.4 分桶策略自模板缩放/偏移；须保证 **`lo < n_cells`**。
- 校验：`python fld_parser.py file.fld --validate-scpost` 或 `vol_flag_pair_issues()`。

### 8.3 场量

自 SDAT 生成仅为 **初值**（温度可由 SDAT 初值区插值到顶点）；压力、湍流等为默认常数。含真实第 N 步结果仍需求解器输出或模板 FLD 场量补丁。

### 8.4 其他网格规模

`write_fld_from_mesh` 通过 `resolve_geometry_layout_fld` / `resolve_template_fld` 按 stem 自动匹配 `tests/{stem}_151.fld`、`_63.fld` 等；**几何模板不要求与输出同单元数**（ex3 已验证）。单元数一致时仍优先复制 block2。

---

## 9. 经验归纳（给后续开发者）

1. **vendor 二进制 = 数据 + 大量元数据**：scPOST 校验节内 preamble/trailer，不仅校验数组长度。
2. **用官方 FLD 作金标准**：十六进制对比优于猜测；`iter_data_blocks` 定位块边界。
3. **Fatal Error 的 offset**：多为节头/preamble 错误，不一定是最大数据块内容错误。
4. **几何前导区 112 字节**：`first_block_bc` 在 **偏移 104**，是高频踩坑点。
5. **模板复制 + 局部补丁** 是实用策略：头、`LS_STREAM*`、Volume block0/1/2、Surface link 区宜复制或按模板缩放；坐标、连通、材料、面列表自网格写出。
6. **VolFlag 边界**：`lo` 必须 `< n_cells`；ex3 bucket0 缩放时勿线性插值绝对值（会超界）。
7. **日志路径**：`%AppData%\Cradle\STwin2025\scPOST_Sx64net\~$CradlePostErrorLog_*.txt`。

---

## 10. 文档与代码交叉引用

- **格式细节**：[`docs/FLD_FORMAT.md`](FLD_FORMAT.md)
- **用户快速入门**：[`README.md`](../README.md)
- **解析核心**：`fld_model.py`
- **写出核心**：`fld_writer.py`

---

## 11. 版本与依赖

- Python 3.10+
- `numpy`, `h5py`（CGNS 导出）
- 参考软件：Cradle CFD 2025.2 scPOST / scFLOW（日志版本示例：`7325.21300.20251203`）
