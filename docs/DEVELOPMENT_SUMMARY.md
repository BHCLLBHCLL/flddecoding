# FLD 解码与生成项目 — 开发总结

本文档总结本仓库的目标、架构、关键技术决策，以及 **ex4_e（cellular_phone）** 案例中从 SDAT 生成 FLD 并被 scPOST 读入的完整攻关过程。

---

## 1. 项目目标

### 1.1 核心目标

1. **解析** Software Cradle scFLOW 的 **FLD**（`CRDL-FLD`）二进制场文件。
2. **转换** 为 CGNS/HDF5，布局对齐官方 `FLDUTIL` 导出（如 `ex1_e_100_orig.cgns`）。
3. **生成** 初始 FLD：仅依赖 **SDAT (.s)** + **EMT (.xemt)**，**无需** 求解结果文件 `.r` 或同网格模板 FLD（模板可选，用于提高 scPOST 兼容性）。

### 1.2 ex4_e 验收标准

| 指标 | 官方 `ex4_e_63.fld` | 本仓库生成（目标） |
|------|---------------------|-------------------|
| 单元数 | 1,470,392 | 一致 |
| 材料 bincount | 7 种材料分布 | 一致 |
| 体段名 | `PARTS1…PARTS32` + 32 部件名（64 个体段） | 一致 |
| 顶点数 | 1,619,610 | 约 1,618,403（界面节点策略差 ~1,207） |
| 表面四边形 | ~956,447 | ~530,737（网格/界面策略差异） |
| scPOST | 正常打开、显示 32 部件树 | 可读入并完成 Relocating volumes |

官方 FLD 含第 63 步求解结果；`.s` + `.xemt` 仅提供网格与初值，场量用初值/零填充可接受。

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

### 4.4 `LS_VolumeGeometryArray` block2 全零

**现象**：通过 Fatal Error 后，在 **“Relocating volumes”** / `CreateVolFlag64_2` 挂起或异常退出。

**原因**：每单元 8 字节体标志块全零；官方为成对 `>i4` 索引（供 `CreateVolFlag64` 建体区域映射）。

**修复**：

- 单元数与 `ex4_e_63.fld` 一致时 **复制 block2**；
- 否则生成 `1…2n` 的 i32 序列回退。

### 4.5 `LS_SurfaceGeometryArray`

**要点**：

- 112 字节几何前导区（同 Volume，注意偏移 104）。
- block0(4608)、block4(72) 宜从模板复制。
- block5 之后 **link 区**（BC 名称等）从模板复制至节尾前 20 字节。

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

# FLD → CGNS
python fld2cgns.py tests/ex1_e_100.fld -o out.cgns

# 查看节布局
python fld_parser.py tests/ex4_e_63.fld

# 模板补丁模式
python s2fld.py tests/ex1_e.s --template tests/ex1_e_100.fld -o out.fld

# 测试
python -m pytest tests/test_sxemt.py -q
python tests/test_ex4_mesh.py
```

---

## 7. 测试与验证

| 测试 | 内容 |
|------|------|
| `test_ex1_e_100.py` | CGNS 与官方参考逐数据集对比 |
| `test_sxemt.py` | ex1 `.s`+`.xemt` 管线冒烟 |
| `test_ex4_mesh.py` | ex4 单元数、材料、体段名、与官方 FLD 解析对比 |
| `test_s2fld.py` | 模板补丁写出 |

**scPOST 验证**（手动）：打开生成 FLD，确认无 “broken”、Relocating volumes 完成、模型树显示 32 部件。

---

## 8. 已知限制与后续工作

### 8.1 顶点与表面面数差异

- 顶点少 ~1,207：界面节点复制规则与 vendor 未完全对齐。
- 表面四边形少 ~40%：与顶点/界面处理相关，可能影响部分 BC 面完整性。
- **不影响** 单元数、材料分布、体段命名树。

### 8.2 体标志 block2

从模板复制在 **单元顺序与官方一致** 时最可靠。若未来网格单元排序不同，需逆向 vendor 的 `CreateVolFlag64` 编码规则，按单元生成正确 i32 对。

### 8.3 场量

自 SDAT 生成仅为 **初值**（温度可由 SDAT 初值区插值到顶点）；压力、湍流等为默认常数。含真实第 N 步结果仍需求解器输出或模板 FLD 场量补丁。

### 8.4 其他网格规模

`write_fld_from_mesh` 对 ex4 硬编码默认模板路径与 `n_cells==1470392`；其他案例需提供匹配模板或扩展自动模板选择逻辑。

---

## 9. 经验归纳（给后续开发者）

1. **vendor 二进制 = 数据 + 大量元数据**：scPOST 校验节内 preamble/trailer，不仅校验数组长度。
2. **用官方 FLD 作金标准**：十六进制对比优于猜测；`iter_data_blocks` 定位块边界。
3. **Fatal Error 的 offset**：多为节头/preamble 错误，不一定是最大数据块内容错误。
4. **几何前导区 112 字节**：`first_block_bc` 在 **偏移 104**，是高频踩坑点。
5. **模板复制 + 局部补丁** 是实用策略：头、`LS_STREAM*`、Volume block0/1/2、Surface link 区宜复制；坐标、连通、材料、面列表自网格写出。
6. **日志路径**：`%AppData%\Cradle\STwin2025\scPOST_Sx64net\~$CradlePostErrorLog_*.txt`。

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
