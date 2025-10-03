#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git Push Script - 自动推送到 GitHub
确保安全可靠地推送代码
"""

import subprocess
import sys
import os
from datetime import datetime

# 目标仓库
REPO_URL = "https://github.com/nichengfuben/nbot-server"

def run_command(cmd, check=True):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if check and result.returncode != 0:
            print(f"❌ 命令失败: {cmd}")
            if result.stderr:
                print(f"   错误: {result.stderr.strip()}")
            return None
        return result
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return None

def check_git_installed():
    """检查 Git 是否安装"""
    result = run_command("git --version", check=False)
    if not result or result.returncode != 0:
        print("❌ Git 未安装，请先安装: https://git-scm.com/")
        return False
    print(f"✅ {result.stdout.strip()}")
    return True

def init_git_repo():
    """初始化或检查 Git 仓库"""
    if not os.path.exists(".git"):
        print("⚙️  初始化 Git 仓库...")
        if run_command("git init") is None:
            return False
        print("✅ 仓库初始化成功")
    else:
        print("✅ Git 仓库已存在")
    return True

def check_git_config():
    """检查 Git 用户配置"""
    name = run_command("git config user.name", check=False)
    email = run_command("git config user.email", check=False)
    
    if not name or not name.stdout.strip():
        print("❌ 未配置 Git 用户名")
        print("   请执行: git config --global user.name \"Your Name\"")
        return False
    
    if not email or not email.stdout.strip():
        print("❌ 未配置 Git 邮箱")
        print("   请执行: git config --global user.email \"your@email.com\"")
        return False
    
    print(f"✅ Git 用户: {name.stdout.strip()} <{email.stdout.strip()}>")
    return True

def setup_remote():
    """配置远程仓库"""
    result = run_command("git remote get-url origin", check=False)
    
    if result and result.returncode == 0:
        current_url = result.stdout.strip()
        if current_url != REPO_URL:
            print(f"⚙️  更新远程仓库: {REPO_URL}")
            if run_command(f"git remote set-url origin {REPO_URL}") is None:
                return False
        print(f"✅ 远程仓库: {REPO_URL}")
    else:
        print(f"⚙️  添加远程仓库: {REPO_URL}")
        if run_command(f"git remote add origin {REPO_URL}") is None:
            return False
        print("✅ 远程仓库已添加")
    
    return True

def git_add():
    """添加所有更改"""
    print("\n📝 添加文件到暂存区...")
    if run_command("git add -A") is None:
        return False
    
    status = run_command("git status --short")
    if status and status.stdout.strip():
        print("✅ 已暂存的文件:")
        for line in status.stdout.strip().split('\n')[:10]:  # 只显示前10个
            print(f"   {line}")
    else:
        print("ℹ️  没有新的更改")
    
    return True

def git_commit(message=None):
    """提交更改"""
    # 检查是否有需要提交的内容
    diff = run_command("git diff --cached --quiet", check=False)
    
    if diff and diff.returncode == 0:
        print("ℹ️  没有需要提交的更改")
        return True
    
    if not message:
        message = f"Auto commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"\n💾 提交更改: {message}")
    if run_command(f'git commit -m "{message}"') is None:
        return False
    
    print("✅ 提交成功")
    return True

def git_push():
    """推送到远程仓库"""
    print("\n🚀 推送到远程仓库...")
    
    # 获取当前分支
    branch_result = run_command("git branch --show-current", check=False)
    if not branch_result or not branch_result.stdout.strip():
        print("⚙️  设置默认分支为 main")
        run_command("git branch -M main")
        branch = "main"
    else:
        branch = branch_result.stdout.strip()
    
    print(f"   分支: {branch}")
    
    # 尝试推送
    result = run_command(f"git push -u origin {branch}", check=False)
    
    if result and result.returncode == 0:
        print("✅ 推送成功!")
        return True
    
    # 推送失败，尝试先拉取
    print("⚙️  尝试拉取远程更改...")
    pull = run_command(f"git pull origin {branch} --rebase --allow-unrelated-histories", check=False)
    
    if pull and pull.returncode == 0:
        print("✅ 拉取成功，重新推送...")
        result = run_command(f"git push -u origin {branch}")
        if result:
            print("✅ 推送成功!")
            return True
    
    print("❌ 推送失败，可能需要:")
    print("   1. 检查网络连接")
    print("   2. 配置 GitHub 认证（SSH 密钥或 Personal Access Token）")
    print("   3. 确认仓库访问权限")
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 Git 自动推送脚本")
    print(f"📦 目标: {REPO_URL}")
    print("=" * 60 + "\n")
    
    # 执行步骤
    steps = [
        ("检查 Git", check_git_installed),
        ("初始化仓库", init_git_repo),
        ("检查配置", check_git_config),
        ("配置远程仓库", setup_remote),
        ("添加文件", git_add),
        ("提交更改", lambda: git_commit(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else None)),
        ("推送代码", git_push),
    ]
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                print(f"\n❌ 步骤失败: {step_name}")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ 步骤异常: {step_name}")
            print(f"   {str(e)}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 所有操作完成!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 未知错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
