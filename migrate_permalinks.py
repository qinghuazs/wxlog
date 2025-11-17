#!/usr/bin/env python3
"""
迁移包含 permalink 的 Markdown 文件到正确的路径
"""
import os
import re
import shutil
from pathlib import Path

def extract_permalink(file_path):
    """从文件中提取 permalink"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'^permalink:\s*(.+)$', content, re.MULTILINE)
            if match:
                return match.group(1).strip()
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    return None

def remove_permalink(file_path):
    """从文件中移除 permalink 行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 移除 permalink 行（包括换行符）
        new_content = re.sub(r'^permalink:.*\n', '', content, flags=re.MULTILINE)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 失败: {e}")
        return False

def find_md_files_with_permalink(root_dir):
    """查找所有包含 permalink 的 .md 文件"""
    files_with_permalink = []

    # 排除 node_modules 目录
    exclude_dirs = {'node_modules', '.git', '.vitepress'}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 修改 dirnames 就地移除要排除的目录
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for filename in filenames:
            if filename.endswith('.md'):
                file_path = os.path.join(dirpath, filename)
                permalink = extract_permalink(file_path)
                if permalink:
                    files_with_permalink.append((file_path, permalink))

    return files_with_permalink

def migrate_file(source_path, permalink, dry_run=False):
    """将文件迁移到 permalink 指定的位置"""
    # 移除开头的 / 和结尾的 .html
    target_path = permalink.lstrip('/').replace('.html', '.md')

    # 确保目标目录存在
    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.exists(target_dir):
        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)
        print(f"  创建目录: {target_dir}")

    # 检查文件是否已经在正确位置
    source_normalized = os.path.normpath(source_path)
    target_normalized = os.path.normpath(target_path)

    if source_normalized == target_normalized:
        print(f"[OK] File already in correct location: {source_path}")
        if not dry_run:
            remove_permalink(source_path)
        return True

    # 检查目标文件是否已存在
    if os.path.exists(target_path):
        # 检查是否是同一个文件
        if os.path.samefile(source_path, target_path):
            print(f"[OK] File already exists: {target_path}")
            if not dry_run:
                remove_permalink(target_path)
            return True
        else:
            print(f"[WARN] Target file already exists: {target_path}")
            print(f"  Source file: {source_path}")
            return False

    print(f"Migrate: {source_path}")
    print(f"  -> {target_path}")

    if not dry_run:
        # 复制文件到新位置
        shutil.copy2(source_path, target_path)
        # 移除 permalink
        remove_permalink(target_path)
        print(f"[OK] Done")

    return True

def main():
    root_dir = '.'

    print("正在查找包含 permalink 的文件...")
    files = find_md_files_with_permalink(root_dir)

    print(f"\n找到 {len(files)} 个包含 permalink 的文件\n")

    # 首先进行 dry run 显示将要执行的操作
    print("=" * 60)
    print("预览迁移操作（dry run）")
    print("=" * 60)

    for source_path, permalink in files:
        migrate_file(source_path, permalink, dry_run=True)
        print()

    # 询问用户是否继续
    print("\n" + "=" * 60)
    response = input("是否执行迁移？ (y/n): ")

    if response.lower() != 'y':
        print("已取消")
        return

    print("\n" + "=" * 60)
    print("开始执行迁移")
    print("=" * 60 + "\n")

    success_count = 0
    for source_path, permalink in files:
        if migrate_file(source_path, permalink, dry_run=False):
            success_count += 1
        print()

    print("=" * 60)
    print(f"迁移完成: {success_count}/{len(files)} 个文件")
    print("=" * 60)

if __name__ == '__main__':
    main()
