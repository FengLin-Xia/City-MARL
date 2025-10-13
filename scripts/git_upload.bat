@echo off
echo 🚀 Git上传脚本
echo ========================

echo 📊 检查当前状态...
git status

echo.
echo 📦 添加所有文件到暂存区...
git add .

echo.
echo 📋 查看暂存区状态...
git status

echo.
echo 💾 提交更改...
git commit -m "feat: 完成Blender-IDE地形上传和RL训练流程

- 添加Blender地形上传脚本
- 扩展Flask服务器支持地形文件处理  
- 实现IDE端地形获取和训练脚本
- 修复回放可视化文件查找问题
- 完成端到端的地形RL训练流程"

echo.
echo 🚀 推送到远程仓库...
git push origin main

echo.
echo ✅ 上传完成！
echo 📊 查看最近提交记录...
git log --oneline -3

pause
