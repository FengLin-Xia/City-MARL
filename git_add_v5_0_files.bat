@echo off
echo 🚀 Adding Enhanced City Simulation v5.0 files to git...
echo.

echo 📁 Adding core v5.0 system files...
git add enhanced_city_simulation_v5_0.py

echo 📦 Adding integration/v5_0 modules...
git add integration/v5_0/
git add integration/v5_0/__init__.py
git add integration/v5_0/pipeline.py
git add integration/v5_0/training_pipeline.py
git add integration/v5_0/export_pipeline.py
git add integration/v5_0/integration_system.py

echo 🏗️ Adding envs/v5_0 modules...
git add envs/v5_0/
git add envs/v5_0/__init__.py
git add envs/v5_0/city_env.py
git add envs/v5_0/budget_pool.py

echo ⚙️ Adding v5.0 configuration files...
git add configs/city_config_v5_0.json
git add configs/city_config_v5_0_1022.json
git add configs/city_config_v5_0_backup_*.json

echo 📄 Adding v5.0 documentation...
git add 1025-3-task.md

echo 🧠 Adding action middleware modules (if any)...
git add action_mw/

echo 📊 Adding reward terms modules (if any)...
git add reward_terms/

echo 🔧 Adding scheduler modules (if any)...
git add scheduler/

echo 📝 Adding contracts (if updated)...
git add contracts/

echo 🛠️ Adding utils (if updated)...
git add utils/

echo.
echo ✅ All v5.0 files added to git staging area!
echo.
echo 📋 Next steps:
echo 1. Check status: git status
echo 2. Commit: git commit -m "feat: Enhanced City Simulation v5.0 - Dynamic Hub & Budget-Unlocked Actions"
echo 3. Push: git push origin main
echo.
echo 🎯 v5.0 Features:
echo   • Dynamic Hub activation with fade-in effects
echo   • Budget-unlocked actions with middleware control
echo   • Pipeline-based architecture
echo   • Configuration-driven system
echo.
pause
