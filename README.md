# FLD ↔ CGNS / SDAT Converter

将 Software Cradle scFLOW 的 **FLD** 场数据文件（`CRDL-FLD` 大端二进制）解析并转换为 CGNS/HDF5，布局对齐官方 `FLDUTIL` 导出结果。支持从 **SDAT (.s)** 求解定义文件生成初始 FLD。

## 依赖

```bash
pip install -r requirements.txt
```

## 文档

- [开发总结](docs/DEVELOPMENT_SUMMARY.md) — 架构、攻关过程、scPOST 兼容性要点
- [FLD 格式说明](docs/FLD_FORMAT.md) — CRDL-FLD 节布局、数据块编码、各节二进制细节

## 用法

### FLD → CGNS

```bash
# 转换（默认测试例）
python fld2cgns.py tests/ex1_e_100.fld -o tests/ex1_e_100.cgns

# 查看 FLD 节布局与网格摘要
python fld_parser.py tests/ex1_e_100.fld

# 与官方参考 CGNS 对比验证
python tests/test_ex1_e_100.py tests/ex1_e_100.cgns tests/ex1_e_100_orig.cgns
```

### SDAT (.s) + EMT (.xemt) → FLD + CGNS（无需 .r / 模板 FLD）

仅需 `.s`（CXYZ、PARTS 铁块盒、初始/边界条件）和 `.xemt`（材料与部件名）：

```bash
# 同时生成 FLD 与 CGNS
python sxemt2fldcgns.py tests/ex1_e.s tests/ex1_e.xemt

# 或仅 FLD
python s2fld.py tests/ex1_e.s --xemt tests/ex1_e.xemt -o tests/ex1_e_0.fld

# 指定参考 FLD（scPOST 头/几何块）；未指定时按 stem 自动匹配 tests/{stem}_151.fld、_63.fld 等
python sxemt2fldcgns.py tests/ex4_e.s tests/ex4_e.xemt --template tests/ex4_e_63.fld

# 检查 scPOST 几何节
python fld_parser.py tests/ex4_e_from_sxemt.fld --validate-scpost

# 验证
python tests/test_sxemt.py

# 室内空调例 ex3_e（scPOST 已验证可读）
python sxemt2fldcgns.py tests/ex3_e.s tests/ex3_e.xemt -o tests/ex3_e_from_sxemt.fld --verify-parse
python tests/compare_ex3_fld.py
python tests/test_ex3_mesh.py

# 电子散热 ex2_e（CPU/PCB/FIN）
python sxemt2fldcgns.py tests/ex2_e.s tests/ex2_e.xemt -o tests/ex2_e_from_sxemt.fld --verify-parse
python tests/test_ex2_mesh.py

# 手机散热例 ex4_e（32 部件 + cellular_phone 分组）
python sxemt2fldcgns.py tests/ex4_e.s tests/ex4_e.xemt
python tests/test_ex4_mesh.py
```

网格由 CXYZ 结构化六面体生成；PARTS 支持多部件、多盒区域（`/` 分隔部件组）；固–流及多材料界面按材料 ID 复制节点。

`ex4_e` 导出与官方一致：`PARTS1`…`PARTS32`（按 SDAT 部件序号）及部件名元素段（共 64 个体段），`LS_VolumeGeometryArray` 同名标签列表。

`ex3_e` 使用 **10 个体段槽**（自 `ex3_e_151.fld`），体标志 5 桶缩放；**`ex3_e_from_sxemt.fld` 已在 scPOST 验证可读。**

`ex2_e` 无官方 FLD 时回退 `ex4_e_63.fld` 模板；vol-flag 自 ex4 **整块比例缩放**至 409188 单元；表面 BC link 区仅在模板 stem 与案例一致时复制。

`.xemt` 中部件分组（如 `cellular_phone`）会被解析并打印。

### SDAT (.s) → FLD（模板方式，可选）

若已有同网格模板 FLD：

```bash
python s2fld.py tests/ex1_e.s --template tests/ex1_e_100.fld -o out.fld
python s_parser.py tests/ex1_e.s
```
## FLD 格式要点（相对 GPH 网格文件）

| 节 | 内容 |
|----|------|
| `LS_Nodes` | 顶点坐标 R8[n,3]（X/Y/Z 三轴块） |
| `LS_MatOfElements` | 每单元材料 ID（1–7；写出时将 SDAT 的 0 映射为 1） |
| `LS_Elements` | 六面体单元连通 I4[n_cells×8] |
| `LS_VolumeGeometryArray` | 体区域名 + block1 桶计数 + 每单元 vol-flag 对（ex3：10 槽 / 5 桶） |
| `LS_SurfaceGeometryArray` | 边界面四边形 + BC 元数据 |
| `Pressure` / `Temperature` / `CN01` / `VECT` / `HVEC` | 顶点中心场变量 |

输出 CGNS 为单 Zone（`FluidZone`），含 `PARTS1`…`PARTS{n}` 与各部件名元素段、`GridElements_Faces`（NGON 边界面）及 `FlowSolution` 场数据。
