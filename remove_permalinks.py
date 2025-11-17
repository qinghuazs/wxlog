#!/usr/bin/env python3
"""
从所有 Markdown 文件中移除 permalink 配置
"""
import os
import re
from pathlib import Path

def remove_permalink(file_path):
    """从文件中移除 permalink 行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否包含 permalink
        if not re.search(r'^permalink:', content, re.MULTILINE):
            return False

        # 移除 permalink 行（包括换行符）
        new_content = re.sub(r'^permalink:.*\n', '', content, flags=re.MULTILINE)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def find_and_remove_permalinks(root_dir):
    """查找并移除所有 permalink"""
    exclude_dirs = {'node_modules', '.git', '.vitepress'}
    count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for filename in filenames:
            if filename.endswith('.md'):
                file_path = os.path.join(dirpath, filename)
                if remove_permalink(file_path):
                    print(f"Removed permalink from: {file_path}")
                    count += 1

    return count

def main():
    root_dir = '.'
    print("Removing permalinks from all Markdown files...")
    count = find_and_remove_permalinks(root_dir)
    print(f"\nRemoved permalinks from {count} files")

if __name__ == '__main__':
    main()
