#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版目录结构分析工具
功能：显示目录结构并提供深入分析功能
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import hashlib

class DirectoryAnalyzer:
    def __init__(self, root_path=".", show_hidden=False, max_depth=None, exclude_file_path=None):
        self.root_path = Path(root_path)
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.exclude_file_path = Path(exclude_file_path).resolve() if exclude_file_path else None # 存储要排除的文件的绝对路径
        self.file_count = 0
        self.dir_count = 0
        self.total_size = 0
        self.largest_files = []
        self.file_types = {}

    def should_show(self, path):
        """判断是否应该显示该路径"""
        if not self.show_hidden and path.name.startswith('.'):
            return False
        # 检查是否是要排除的文件
        if self.exclude_file_path and path.resolve() == self.exclude_file_path:
            return False
        return True

    def get_file_size(self, file_path):
        """安全获取文件大小"""
        try:
            return file_path.stat().st_size
        except (OSError, PermissionError):
            return 0

    def format_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"

    def get_file_type(self, file_path):
        """获取文件类型"""
        return file_path.suffix.lower() or "无扩展名"

    def analyze_file(self, file_path):
        """分析单个文件"""
        # --- 新增：检查是否为要排除的文件 ---
        if self.exclude_file_path and file_path.resolve() == self.exclude_file_path:
            return # 如果是，则跳过分析
        # --- 新增结束 ---

        size = self.get_file_size(file_path)
        self.file_count += 1
        self.total_size += size
        # 记录文件类型统计
        file_type = self.get_file_type(file_path)
        self.file_types[file_type] = self.file_types.get(file_type, 0) + 1
        # 记录大文件（保留最大的10个）
        if len(self.largest_files) < 10 or size > self.largest_files[-1][1]:
            self.largest_files.append((str(file_path.relative_to(self.root_path)), size))
            self.largest_files.sort(key=lambda x: x[1], reverse=True)
            self.largest_files = self.largest_files[:10]

    def display_tree(self, path=None, prefix="", depth=0, is_last=True):
        """递归显示目录树结构"""
        if path is None:
            path = self.root_path
        # 检查深度限制
        if self.max_depth is not None and depth > self.max_depth:
            return
        # 检查是否应该显示 (包括是否是排除文件)
        if depth > 0 and not self.should_show(path):
            return
        # 获取目录/文件名
        name = path.name if path.name else str(path)
        # 打印当前项
        if depth > 0:  # 不是根目录
            connector = "└── " if is_last else "├── "
            size_info = ""
            if path.is_file():
                size_info = f" ({self.format_size(self.get_file_size(path))})"
            print(f"{prefix}{connector}{name}{size_info}")
        else:
            print(f"{name}")
        # 如果是文件，进行分析 (analyze_file内部已处理排除)
        if path.is_file():
            self.analyze_file(path)
            return
        # 如果是目录，继续处理
        self.dir_count += 1
        # 更新前缀
        if depth > 0:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
        else:
            new_prefix = ""
        # 获取子项
        try:
            items = list(path.iterdir())
            # 过滤隐藏文件和排除文件
            items = [item for item in items if self.should_show(item)] # should_show 已包含排除文件逻辑
            # 排序：目录在前，按名称排序
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            print(f"{new_prefix}└── [权限不足]")
            return
        except Exception as e:
            print(f"{new_prefix}└── [错误: {e}]")
            return
        # 处理每个子项
        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)
            self.display_tree(item, new_prefix, depth + 1, is_last_item)

    def analyze_deep(self):
        """深入分析目录"""
        print("正在深入分析目录...")
        start_time = time.time()
        # 重置统计数据
        self.file_count = 0
        self.dir_count = 0
        self.total_size = 0
        self.largest_files = []
        self.file_types = {}
        # 遍历所有文件进行深入分析
        try:
            for root, dirs, files in os.walk(self.root_path):
                # 过滤隐藏目录
                if not self.show_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                # 分析文件
                for file in files:
                    if not self.show_hidden and file.startswith('.'):
                        continue
                    file_path = Path(root) / file
                    # --- 新增：在分析前检查是否为要排除的文件 ---
                    if self.exclude_file_path and file_path.resolve() == self.exclude_file_path:
                        continue # 如果是，则跳过
                    # --- 新增结束 ---
                    self.analyze_file(file_path)
                # 统计目录 (注意：os.walk 的 dirs 列表已经根据 show_hidden 过滤过)
                # 不需要在这里特别排除目录，因为脚本文件是文件不是目录
                self.dir_count += len([d for d in dirs if self.show_hidden or not d.startswith('.')])
        except Exception as e:
            print(f"深入分析时出错: {e}")
        end_time = time.time()
        return end_time - start_time

    def display_statistics(self):
        """显示详细统计信息"""
        print("\n" + "=" * 60)
        print("详细统计信息")
        print("=" * 60)
        print(f"目录数量: {self.dir_count}")
        print(f"文件数量: {self.file_count}")
        print(f"总大小: {self.format_size(self.total_size)}")
        # 文件类型统计
        if self.file_types:
            print("\n文件类型统计:")
            sorted_types = sorted(self.file_types.items(), key=lambda x: x[1], reverse=True)
            for file_type, count in sorted_types[:10]:  # 显示前10种类型
                percentage = (count / self.file_count * 100) if self.file_count > 0 else 0
                print(f"  {file_type or '无扩展名'}: {count} 个 ({percentage:.1f}%)")
        # 最大文件
        if self.largest_files:
            print("\n最大的文件:")
            for i, (file_path, size) in enumerate(self.largest_files, 1):
                print(f"  {i:2d}. {file_path} ({self.format_size(size)})")

    def find_duplicate_files(self, max_check=100):
        """查找重复文件（基于文件大小）"""
        print("\n正在查找可能的重复文件...")
        # 按大小分组文件
        size_groups = {}
        file_count = 0
        try:
            for root, dirs, files in os.walk(self.root_path):
                if not self.show_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if not self.show_hidden and file.startswith('.'):
                        continue
                    # --- 新增：检查是否为要排除的文件 ---
                    file_path_obj = Path(root) / file
                    if self.exclude_file_path and file_path_obj.resolve() == self.exclude_file_path:
                        continue # 如果是，则跳过
                    # --- 新增结束 ---
                    if file_count >= max_check:
                        break
                    file_path = file_path_obj
                    size = self.get_file_size(file_path)
                    if size > 0:  # 只考虑非空文件
                        if size not in size_groups:
                            size_groups[size] = []
                        size_groups[size].append(str(file_path.relative_to(self.root_path)))
                        file_count += 1
                if file_count >= max_check:
                    break
            # 查找大小相同的文件组
            duplicates = {size: paths for size, paths in size_groups.items() if len(paths) > 1}
            if duplicates:
                print(f"\n发现 {len(duplicates)} 组可能的重复文件:")
                for i, (size, paths) in enumerate(list(duplicates.items())[:10]):  # 显示前10组
                    print(f"\n  组 {i+1} (大小: {self.format_size(size)}):")
                    # --- 新增：再次过滤显示的路径 ---
                    filtered_paths = [p for p in paths if not (self.exclude_file_path and Path(self.root_path / p).resolve() == self.exclude_file_path)]
                    # --- 新增结束 ---
                    for path in filtered_paths[:5]:  # 每组最多显示5个文件
                        print(f"    {path}")
                    if len(filtered_paths) > 5:
                        print(f"    ... 还有 {len(filtered_paths) - 5} 个文件")
            else:
                print("未发现明显的重复文件")
        except Exception as e:
            print(f"查找重复文件时出错: {e}")

def main():
    """主函数"""
    import argparse
    # --- 新增：获取当前脚本的绝对路径 ---
    current_script_path = Path(__file__).resolve()
    # --- 新增结束 ---

    parser = argparse.ArgumentParser(
        description="增强版目录结构分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python tree_analyzer.py                    # 显示当前目录结构
  python tree_analyzer.py -a                 # 显示包括隐藏文件的目录结构
  python tree_analyzer.py -d 2               # 只显示到2层深度
  python tree_analyzer.py -s                 # 显示详细统计信息
  python tree_analyzer.py -D                 # 查找重复文件
  python tree_analyzer.py /path/to/dir       # 分析指定目录
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="要分析的目录路径 (默认: 当前目录)"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="显示隐藏文件和目录"
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        help="限制显示深度"
    )
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="显示详细统计信息"
    )
    parser.add_argument(
        "-D", "--duplicates",
        action="store_true",
        help="查找重复文件"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细模式，显示更多信息"
    )
    # --- 新增：添加一个参数来指定要排除的文件 ---
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        default=True, # 默认排除自身
        help="排除当前脚本文件本身 (默认: True)"
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        help="指定要额外排除的文件名 (相对于分析目录)"
    )
    # --- 新增结束 ---

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.path):
        print(f"错误: 路径 '{args.path}' 不存在")
        sys.exit(1)
    if not os.path.isdir(args.path):
        print(f"错误: '{args.path}' 不是一个目录")
        sys.exit(1)

    # --- 修改：确定最终要排除的文件路径 ---
    exclude_path = None
    if args.exclude_self:
        exclude_path = current_script_path
    elif args.exclude_file:
        # 如果指定了额外排除的文件，优先使用它（相对于分析目录）
        potential_exclude = Path(args.path) / args.exclude_file
        if potential_exclude.exists():
             exclude_path = potential_exclude.resolve()
        else:
             print(f"警告: 指定的排除文件 '{args.exclude_file}' 在 '{args.path}' 中未找到。")

    # --- 修改结束 ---

    # 创建分析器，传入排除文件路径
    analyzer = DirectoryAnalyzer(args.path, args.all, args.depth, exclude_file_path=exclude_path)

    # 显示标题
    abs_path = os.path.abspath(args.path)
    print(f"目录分析: {abs_path}")
    if args.depth:
        print(f"深度限制: {args.depth} 层")
    if exclude_path and exclude_path.parent == Path(abs_path):
        pass
    elif exclude_path:
         pass
    print("=" * 60)

    # 显示目录树
    analyzer.display_tree()

    # 深入分析
    if args.stats or args.duplicates or args.verbose:
        analysis_time = analyzer.analyze_deep()
        print(f"\n深入分析完成，耗时: {analysis_time:.2f} 秒")

    # 显示统计信息
    if args.stats or args.verbose:
        analyzer.display_statistics()

    # 查找重复文件
    if args.duplicates:
        analyzer.find_duplicate_files()

if __name__ == "__main__":
    main()
