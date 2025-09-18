# CSV文件分割方案说明

## 📊 问题背景
MHI项目包含多个超大CSV文件，超过GitHub LFS的2GB单文件限制：
- `panda.csv`: 21GB, 2260万行
- `internvid_yolo.csv`: 4.6GB, 1580万行
- `internvid.csv`: 3.7GB, 1750万行
- `panda_multi_aes45.csv`: 2.7GB, 280万行
- `mhi_multi_aes45_coarse_ocr.csv`: 2.6GB, 450万行

## ✂️ 分割策略
使用智能分割方案将大文件拆分为<2GB的小文件：

### 分割参数
- **panda.csv** → 12片，每片190万行 (~1.75GB)
- **internvid_yolo.csv** → 16片，每片100万行 (~290MB)
- **internvid.csv** → 22片，每片80万行 (~170MB)
- **panda_multi_aes45.csv** → 5片，每片60万行 (~570MB)
- **mhi_multi_aes45_coarse_ocr.csv** → 10片，每片50万行 (~260MB)

### 目录结构
```
csv/data/
├── panda/
│   ├── panda.csv                    # 原始文件(忽略)
│   └── parts/
│       ├── panda_part_aa.csv        # 分片1
│       ├── panda_part_ab.csv        # 分片2
│       └── ...
├── internvid/
│   ├── internvid.csv               # 原始文件(忽略)
│   ├── internvid_yolo.csv          # 原始文件(忽略)
│   └── parts/
│       ├── internvid_part_aa.csv
│       ├── internvid_yolo_part_aa.csv
│       └── ...
└── mhi/
    ├── mhi_multi_aes45_coarse_ocr.csv  # 原始文件(忽略)
    └── parts/
        ├── mhi_multi_aes45_coarse_ocr_part_aa.csv
        └── ...
```

## 🛠️ 使用方法

### 分割大文件
```bash
# 执行分割脚本
./split_large_csv.sh

# 检查分割结果
find csv/data -name "*_part_*.csv" | wc -l
```

### 合并分片文件
```bash
# 执行合并脚本
./merge_csv_parts.sh

# 验证合并结果
ls -lh csv/data/*/*_merged.csv
```

### Git同步
```bash
# 添加分片文件到Git LFS
git add csv/data/*/parts/

# 检查LFS状态
git lfs ls-files

# 提交和推送
git commit -m "Add CSV split files via LFS"
git push origin master
```

## 🔧 脚本说明

### split_large_csv.sh
- 自动检测超大CSV文件
- 保留原始CSV头部结构
- 按配置的行数分割文件
- 为每个分片添加完整的CSV头部
- 提供详细的进度和统计信息

### merge_csv_parts.sh
- 自动识别分片文件模式
- 合并时保持原始数据顺序
- 验证合并后的行数正确性
- 生成带有"_merged"后缀的完整文件

## 📋 Git配置

### .gitattributes
```bash
# CSV分片文件通过LFS管理
csv/data/*/parts/*_part_*.csv filter=lfs diff=lfs merge=lfs -text
```

### .gitignore
```bash
# 原始超大CSV文件被忽略
csv/data/panda/panda.csv
csv/data/internvid/internvid_yolo.csv
csv/data/internvid/internvid.csv
csv/data/panda/panda_multi_aes45.csv
csv/data/mhi/mhi_multi_aes45_coarse_ocr.csv
```

## ✅ 优势
1. **兼容GitHub LFS**: 所有分片文件<2GB
2. **保持完整性**: 分割和合并过程保持数据完整
3. **灵活访问**: 可以单独处理特定分片
4. **协作友好**: 团队成员可以选择性下载需要的分片
5. **版本控制**: 每个分片都可以独立版本控制

## 💡 最佳实践
1. **开发时**: 使用分片文件进行开发和测试
2. **生产时**: 使用合并脚本重建完整文件
3. **协作时**: 分享分片索引，让团队成员按需下载
4. **备份时**: 原始大文件和分片文件都要备份

## 🔄 数据一致性
- 合并后文件与原始文件完全一致
- 保持CSV格式和编码不变
- 维护原始的行顺序
- 验证总行数匹配