echo
echo "--------------------------------------------------------------------------------"
echo "Comparing sequential solution with reference data"
echo "--------------------------------------------------------------------------------"
echo "M: 128, N: 128, max iteration: 100000, snapshot frequency: 10000"
echo "--------------------------------------------------------------------------------"
echo
./sequential 1>/dev/null
./check/compare_solutions 128 128 data/00050.bin check/references/00050.bin
echo
