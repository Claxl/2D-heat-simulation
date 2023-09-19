echo
echo "--------------------------------------------------------------------------------"
echo "Comparing sequential solution with reference data"
echo "--------------------------------------------------------------------------------"
echo "M: 256, N: 256, max iteration: 100000, snapshot frequency: 1000"
echo "--------------------------------------------------------------------------------"
echo
./sequential 1>/dev/null
./check/compare_solutions 256 256 data/00050.bin check/references/00050.bin
echo
