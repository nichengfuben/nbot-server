import subprocess
import os

def run_command(cmd, description=""):
    """执行命令并处理输出"""
    if description:
        print(f"\n{'='*50}")
        print(description)
        print('='*50)
    
    print(f">>> 执行: {cmd}")
    process = subprocess.Popen(cmd, shell=True, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8')
    
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    
    process.wait()
    return process.returncode, ''.join(output_lines)

# 设置工作目录
os.chdir(r"E:\我的\python\new\MODEL - SERVER")

# 1. 检查当前状态
returncode, output = run_command("git status", "检查Git状态")

# 2. 初始化（如果需要）
if "not a git repository" in output:
    run_command("git init", "初始化Git仓库")

# 3. 添加文件
run_command("git add .", "添加文件到暂存区")

# 4. 提交更改
run_command('git commit -m "初始提交"', "提交更改")

# 5. 重命名分支
run_command("git branch -M main", "重命名分支为main")

# 6. 检查并配置远程仓库
returncode, output = run_command("git remote -v -y", "检查远程仓库配置")

if "origin" in output:
    choice = input("远程仓库origin已存在，是否移除? (y/n): ")
    if choice.lower() == 'y':
        run_command("git remote remove origin", "移除现有origin")

# 7. 添加正确的远程仓库
run_command("git remote add origin https://github.com/nichengfuben/nbot-server.git", "添加远程仓库")

# 8. 尝试拉取并合并
returncode, output = run_command("git pull origin main --allow-unrelated-histories", "拉取远程更改")

if returncode != 0:
    print("拉取失败，尝试强制推送（这将覆盖远程仓库）")
    choice = input("确定要强制推送吗? (y/n): ")
    if choice.lower() == 'y':
        run_command("git push -f -u origin main", "强制推送")
    else:
        print("操作取消")
else:
    # 9. 正常推送
    run_command("git push -u origin main", "推送到远程仓库")

print("\n操作完成!")
