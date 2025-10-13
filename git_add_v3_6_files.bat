@echo off
echo 🚀 Adding Enhanced City Simulation v3.6 files to git...
echo.

echo 📁 Adding core system files...
git add enhanced_city_simulation_v3_6.py
git add visualize_building_placement_v3_6.py
git add test_finance_system.py

echo 📄 Adding documentation...
git add enhanced_city_simulation_prd_v3.6.txt

echo ⚙️ Adding configuration files...
git add configs/city_config_v3_5.json
git add configs/city_config_v3_5_backup.json
git add restore_original_config.py

echo 🧠 Adding logic modules...
git add logic/enhanced_sdf_system.py

echo 📊 Adding output files...
git add enhanced_simulation_v3_6_output/simplified/
git add enhanced_simulation_v3_6_output/building_placement_animation_v3_6.gif
git add enhanced_simulation_v3_6_output/finance_visualizations/

echo.
echo ✅ All v3.6 files added to git staging area!
echo.
echo 📋 Next steps:
echo 1. Check status: git status
echo 2. Commit: git commit -m "feat: Enhanced City Simulation v3.6 complete implementation"
echo 3. Push: git push origin main
echo.
pause
