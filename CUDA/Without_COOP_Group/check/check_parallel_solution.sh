echo
echo "--------------------------------------------------------------------------------"
echo "Comparing parallel solution with reference data"
echo "--------------------------------------------------------------------------------"
echo
echo "--------------------------------------------------------------------------------"
echo "M: 256, N: 256, max iteration: 100000, snapshot frequency: 10000"
echo "--------------------------------------------------------------------------------"
./parallel -m 256 -n 256 1>/dev/null
./check/compare_solutions 256 256 data/00050.bin check/references/256/00050.bin
echo
echo "--------------------------------------------------------------------------------"
echo "M: 1024, N: 1024, max iteration: 100000, snapshot frequency: 10000"
echo "--------------------------------------------------------------------------------"
./parallel -m 1024 -n 1024 1>/dev/null
./check/compare_solutions 1024 1024 data/00050.bin check/references/1024/00050.bin
echo
