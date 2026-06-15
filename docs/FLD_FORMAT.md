# scFLOW FLD 二进制格式说明

本文档描述 Software Cradle **scFLOW / scPOST** 使用的 **FLD**（`CRDL-FLD`）场文件布局。本仓库通过逆向官方 `FLDUTIL` 导出结果与 scPOST 读入行为归纳而成，**非官方规范**；细节以 `fld_model.py` / `fld_writer.py` 实现为准。

---

## 1. 文件总览

| 属性 | 说明 |
|------|------|
| 魔数 | 大端 `I4=8` + ASCII `CRDL-FLD` + `I4=8` + `I4,I4,I4`（各 4） |
| 字节序 | **全程大端**（`>i4`, `>f8`, `>f64`） |
| 容器类型 | 与 GPH 网格文件共享 **CRDL** 节式容器，但 FLD 存六面体连通、顶点场量与表面 BC，而非多面体 `LS_Links` |
| 典型用途 | scFLOW 时间步结果；scPOST 后处理；本仓库可从 SDAT 生成“初值 FLD” |

文件由若干 **命名节（section）** 顺序拼接。每节结构：

```
[I4=32][节名 ASCII 32 字节，左对齐空格填充][I4=32][节体内数据]
```

节名示例：`LS_Nodes`、`Pressure`、`LS_VolumeGeometryArray`。定位方式：在文件中搜索 32 字节节名，且其前 4 字节为 `32`（见 `find_section`）。

---

## 2. 数据块（Data Block）编码

节体内由多个 **数据块** 组成。扫描逻辑见 `iter_data_blocks`。

### 2.1 标准负载块

```
[I4=12][I4=bc][payload: bc 字节][I4=bc]
```

- `bc`：负载字节数，必须 > 0
- 尾部重复 `bc` 作为校验

### 2.2 元数据小块（非负载）

扫描时若遇到：

```
[I4=12][I4=4或8][dim0][dim1]
```

且 `dim0`、`dim1` 在合理范围（< 10⁷），则视为 **维度描述**，跳过 16 字节，**不产生负载**。

常见模式：

| 模式 | 典型含义 |
|------|----------|
| `12, 4, 1, 1` | 通用标志 |
| `12, 4, n_vertices, 4` | 顶点数相关 |
| `12, 8, n_vertices, 1` | 顶点数 + 向量维 |
| `12, 4, n_cells, 4` | 单元数相关 |
| `12, 0, 0, 0` | 16 字节填充（标量 f64 块后） |

### 2.3 节尾

网格/几何节常以 **20 字节** 结束：

```
16 字节 pad（常为 12,0,0,0 形式）
[I4=12]   ← 节后缀标记
```

场量节使用 48 字节 trailer + `I4=12`，见下文。

---

## 3. 文件头与全局节

典型顺序（自 `OverlapStart_0` 起为重叠区主体）：

| 节名 | 作用 |
|------|------|
| `FileRevision` | 版本整数 |
| `Application` | 应用名（如 scFLOW） |
| `ApplicationVersion` | 版本字符串 |
| `ReleaseDate` / `GridType` / `Dimension` / `Bias` / `Date` / `Comments` | 元数据 |
| `Cycle` | 时间步编号 |
| `Unused` / `Encoding` / `HeaderDataEnd` | 头结束标记 |
| `OverlapStart_0` | 重叠区起点 |
| `LS_CoordinateSystem` | 坐标系标志 |

**scPOST 兼容性**：写出时建议从官方参考 FLD **整节复制**头至 `LS_CoordinateSystem`（`fld_writer._vendor_header_bytes`），否则后处理可能拒绝打开。

文件前缀（在第一节之前）：

```
[I4=8]["CRDL-FLD"][I4=8][I4=4][I4=4][I4=4]
```

---

## 4. 场量节（顶点中心）

所有场量按 **顶点数 `n_vertices`** 存储，与 `LS_Nodes` 一致。scPOST 要求完整的 **前导区 + 标签 + 尾区**，不能只写裸 f64 数组。

### 4.1 通用场量前导区（48 字节）

```
12, 4, 1, 1
12, 4, n_vertices, 4
12, 8, n_vertices, 1
```

### 4.2 标量 f64 块

```
[数据块: n_vertices × 8 字节 >f8]
[16 字节 pad: 12, 0, 0, 0]
```

### 4.3 向量 f64 块（三轴）

每个轴：

```
[数据块: n_vertices × 8 字节]
[16 字节: 12, 8, n_vertices, 1]
```

最后一轴后接 pad + 节尾。

### 4.4 场量节之间的标签 trailer（48 字节）

用于连接子场（如 `Pressure` → `Temperature` 内的 `LS_Scalar:TEMP`）：

```
12, 4, 1, 1
12, 4, 2, 4
12, 1, 32, 1    ← 32 字节 ASCII 标签块前的头
```

**链接 trailer**（如 `CN01` → `VECT`）第三组为 `12, 4, 1, 1` 而非 `12, 1, 32, 1`。

### 4.5 链接节尾（CN01 / VECT）

```
[数据块: 4 字节 0]
12, 1, 32, 1
[I4=12]
```

### 4.6 各场量节内容映射

| 节名 | 物理量 | 块结构概要 |
|------|--------|------------|
| `Pressure` | 压力 `PRES` | 1×f64 + 标签 `LS_Scalar:TEMP` |
| `Temperature` | 温度 `TEMP` | f64 + `LS_Scalar:TURK` + 描述 + f64 `TURK` + `LS_Scalar:TEPS` + … + `LS_Scalar:CN01` |
| `CN01` | 标量链 | `CN01`→`HTRC`→`SURT`→`HTFX`（向量入口） |
| `VECT` | 速度 `VECTX/Y/Z` | 3×f64 向量 + 链接 `LS_Vector:HVEC` |
| `HVEC` | 辅助向量 | 3×f64 向量 + 节尾 |

**初值惯例**（仅网格、无求解时）：`PRES/TEMP=20`，湍流 `TURK≈0.689`，`TEPS≈646.5`，墙量 `1e20`（见 `default_initial_fields`）。

---

## 5. 网格节

### 5.1 `LS_Nodes`

**前导区（80 字节）** = 48 字节 mesh base + 32 字节 vertices tail：

```
mesh base (48B):
  12,4,1,1 | 12,4,1,4 | 12,4,1,1

vertices tail (32B):
  12, 4, n_vertices, 4
  4
  12, 8
  n_vertices
  1
```

**数据块**：三个等大 f64 块，分别为 X、Y、Z 坐标（`>f8[n_vertices]`）。

块间：**16 字节** `12, 8, n_vertices, 1`（Y 与 Z 之间）。

**节尾**：16 字节 pad + `I4=12`。

### 5.2 `LS_MatOfElements`

**前导区（80 字节）** = mesh base + cells tail：

```
cells tail (32B):
  12, 4, n_cells, 4
  4
  12, 4, n_cells
  1
```

**数据块**：`>i4[n_cells]`，每单元 **材料 ID**（1=流体，2=固体等，与 SDAT PARTS 对应）。

### 5.3 `LS_Elements`

与 `LS_MatOfElements` 相同 80 字节 cells 前导区。

**块 1**：`>i4[n_cells]`，单元类型元数据（本仓库写 **38**，与官方 ex4 一致）。

**16 字节分隔**：`12, 4, 1, 1`。

**块 2**：4 字节整数 = 扁平连通数组长度（`n_cells × 8`）。

**16 字节分隔**：`12, 4, flat_len, 1`。

**块 3**：`>i4[n_cells × 8]`，六面体 **8 顶点索引**（1-based，与 `LS_Nodes` 一致）。

**节尾**：20 字节。

---

## 6. `LS_VolumeGeometryArray`（体区域几何）

scPOST **“Relocating volumes”** 阶段强依赖本节；布局错误会导致 `Fatal Error (TITLE = LS_VolumeGeometryArray)` 或 `CreateVolFlag64` 挂起。

### 6.1 前导区（112 字节）

在 mesh base (48B) 之后：

```
12, 4, count_val, 4          ← count_val 常为 64（体段名数量相关）
12, 4, 1, 1
12, 4, 256, 4                ← 注意：官方为 256，非 65536
12, 1, first_block_bc, 1     ← 第一数据块字节数（如 16384）
```

**关键细节**：`first_block_bc` 必须写在 **偏移 104**（节体内、第一数据块头内），**不是偏移 100**。偏移 100 处固定为 `1`。错误写入会导致 scPOST 在节起始附近报 Fatal Error。

从模板修补时使用 `_geometry_preamble_from_template`：`pack_into` 偏移 **56** 改 `count_val`，偏移 **104** 改 `first_block_bc`。

### 6.2 数据块布局

| 序号 | 大小 | 内容 |
|------|------|------|
| block0 | 16384 | ASCII 体区域名列表，空格分隔：`PARTS1 … PARTS32 lower_cover_01 …`（ex4 共 64 名） |
| sep | 16 | `12, 4, 64, 1` |
| block1 | 256 | 体元数据（官方为重复 i32 模式，宜从模板复制） |
| sep | 16 | `12, 4, n_cells×8, 1`（注意 sep 值为字节数×4 的编码方式） |
| block2 | `n_cells × 8` | **每单元 8 字节** 体标志数组 |
| tail | 20 | 节尾 |

### 6.3 block2 语义（体标志 / VolFlag）

- 物理上为 **每单元 8 字节**，宜按 **两个大端 i32** 解读：`[hi, lo]` 成对出现。
- 官方 ex4_e_63.fld：前段为 `(1,2), (3,4), …` 递增对；后段因单元重排/分区与简单 `2i+1, 2i+2` 不完全一致。
- **不可全零**：scPOST `CreateVolFlag64` 需要有效索引；全零会在 “Relocating volumes” 异常退出。
- **推荐策略**（本仓库）：
  - 单元数与模板一致时：**整块复制**自 `ex4_e_63.fld`；
  - 否则：生成 `1 … 2×n_cells` 的 `>i4` 序列作为回退。

读为 `>f8` 时，小整数对会显示为极小 denormal 浮点（如 `2.12e-314`），这是 **整数编码**，不是物理浮点场。

---

## 7. `LS_SurfaceGeometryArray`（表面几何与 BC）

表面由 **四边形面** + 元数据组成，供 scPOST 建 `Surface Flag` 与 CGNS `GridElements_Faces`。

### 7.1 前导区

与 `LS_VolumeGeometryArray` 同为 **112 字节** 几何前导区；`count_val` = meta 数组长度（约 18），`first_block_bc` 常为 4608。

### 7.2 数据块（典型 6+ 块）

| 序号 | 内容 |
|------|------|
| block0 | 4608 字节（常从模板复制） |
| block1 | `>i4[meta_len]` 各 BC 类别面数（ENTB, ENTF, MOM, PARTS, Xmax, …） |
| block2 | `>i4[n_faces]`，常全 134 |
| block3 | `>i4[n_faces]`，**相邻单元 flat 索引 +1**（1-based cell index） |
| block4 | 72 字节（宜从模板复制） |
| block5 | `>i4[n_faces×4]`，四边形顶点索引 |

**块间分隔**：16 字节 `12, 4, val, 1`，`val` 为下一块相关计数（面数×4 或 meta 长度等）。

### 7.3 block5 之后的链接区

官方 FLD 在 block5 后还有 **BC 名称** 等数据（18 字节 ASCII 块：`@UNDEFINEDENTB`、`Ymax` 等）。本仓库从模板 **复制 link 区至节尾前 20 字节**（`_write_surface_geometry_array`）。

### 7.4 面分类（`surface_builder.py`）

内部面按材料/部件变化分为：

- `@UNDEFINEDENTB` / `ENTF` / `ENTS` / `ENTX` / `MOM` / `VFWL`
- `PARTS`、`SURFACE`（材料或部件界面）
- 外边界：`Xmin/Xmax/Ymin/Ymax/Zmin/Zmax`

最终 **NGON 面列表** = seg1（按 meta 顺序拼接）+ seg2（按材料拆分的 ENTB/PARTS/SURFACE/Ymax 等）。详见 `fld_model._build_face_list_and_bcs`。

**已知差异**：自 SDAT 生成的表面面数（ex4 约 **530,737**）少于官方求解结果（约 **956,447**），因顶点数与界面节点复制策略不同；可能影响 BC 完整性，但不应导致 VolumeGeometry Fatal Error。

---

## 8. `LS_SFile`（嵌入 SDAT）

| 块 | 内容 |
|----|------|
| 0 | `>f64` 标志（常为 1.0） |
| 1 | UTF-8 文本，以 `SDAT` 开头，内含完整 `.s` 求解定义 |

块 1 有固定槽位大小（模板中约 6KB+）；超出会写失败。换行宜为 LF。

---

## 9. 收尾节

| 节名 | 说明 |
|------|------|
| `LS_STREAMcoc` / `LS_STREAMmultiblock` | 流线相关（宜从模板复制） |
| `OverlapEnd` | 重叠区结束；本仓库写 **空节体**（仅 40 字节节头） |

---

## 10. 节顺序（写出顺序）

`write_fld_from_mesh` 顺序：

1. 文件头（模板复制）
2. `Pressure` → `Temperature` → `CN01` → `VECT` → `HVEC`
3. `LS_STREAMcoc` / `LS_STREAMmultiblock`（可选，来自模板）
4. `LS_Nodes` → `LS_MatOfElements` → `LS_Elements`
5. `LS_VolumeGeometryArray` → `LS_SurfaceGeometryArray`
6. `LS_SFile` → `OverlapEnd`

解析时 `section_end` 通过已知节名列表取 **下一节偏移** 作为当前节结束。

---

## 11. 与 CGNS 的对应

`fld2cgns.py` 将 FLD 转为单 Zone `FluidZone`：

- 坐标 ← `LS_Nodes`
- 体元素 ← `LS_Elements` + `LS_MatOfElements`（`PARTS1…` + 部件名元素段）
- 面元素 ← 由 `LS_SurfaceGeometryArray` 重建的 NGON
- 场解 ← `Pressure` / `Temperature` / `CN01` / `VECT` / `HVEC`

---

## 12. 调试 scPOST 的实用方法

1. **错误日志**：`%AppData%\Cradle\STwin2025\scPOST_Sx64net\~$CradlePostErrorLog_*.txt`
2. **Fatal Error (TITLE = …)**：对照节名检查该节前导区与第一数据块头（尤其偏移 104）。
3. **Relocating volumes / CreateVolFlag64**：检查 `LS_VolumeGeometryArray` block2 是否全零。
4. **节布局**：`python fld_parser.py your.fld` 或 `describe_fld_sections`。
5. **十六进制对比**：`tests/analyze_fld_offset.py`（若存在）对比官方与生成文件指定偏移。

---

## 13. 参考实现索引

| 功能 | 模块 |
|------|------|
| 解析 | `fld_model.py` |
| 写出 | `fld_writer.py` |
| 节扫描 | `find_section`, `iter_data_blocks`, `section_end` |
| 模板补丁写出 | `compose_fld` |
| 从零网格写出 | `write_fld_from_mesh` |

官方参考文件（本仓库测试）：

- `tests/ex1_e_100.fld` — 小规模双材料例
- `tests/ex4_e_63.fld` — 手机散热 1470392 单元、64 体段名
