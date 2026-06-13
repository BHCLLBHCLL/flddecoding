# FLD ↔ CGNS / SDAT Converter

将 Software Cradle scFLOW 的 **FLD** 场数据文件（`CRDL-FLD` 大端二进制）解析并转换为 CGNS/HDF5，布局对齐官方 `FLDUTIL` 导出结果。支持从 **SDAT (.s)** 求解定义文件生成初始 FLD。

## 依赖

```bash
pip install -r requirements.txt
```

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

# 验证
python tests/test_sxemt.py
```

网格由 CXYZ 结构化六面体生成；PARTS 行 `i1 i2 j1 j2 k1 k2` 为铁块节点范围（1-based）；固–流界面自动复制节点。

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
| `LS_MatOfElements` | 每单元材料 ID（1/2 → PARTS1/PARTS2） |
| `LS_Elements` | 六面体单元连通 I4[n_cells×8] |
| `LS_VolumeGeometryArray` | 体区域名（PARTS1、Domain(cuboid)、Iron 等） |
| `LS_SurfaceGeometryArray` | 边界面四边形 + BC 元数据 |
| `Pressure` / `Temperature` / `CN01` / `VECT` / `HVEC` | 顶点中心场变量 |

输出 CGNS 为单 Zone（`FluidZone`），含 `PARTS1/PARTS2`、域名称元素段、`GridElements_Faces`（NGON 边界面）及 `FlowSolution` 场数据。
