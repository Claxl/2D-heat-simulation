echo
echo "--------------------------------------------------------------------------------"
echo "Comparing solution with reference data"
echo "--------------------------------------------------------------------------------"
echo "N: 2048, max iteration: 1000000, snapshot frequency: 10000"
echo
echo "--------------------------------------------------------------------------------"
echo "Jacobi solver"
echo "--------------------------------------------------------------------------------"
./heat -t 1 > /dev/null
./check/compare_solutions 2048 data/00050.dat check/references/jacobi/00050.dat

echo
echo "--------------------------------------------------------------------------------"
echo "Gauss-Seidel solver"
echo "--------------------------------------------------------------------------------"
./heat -t 2 > /dev/null
./check/compare_solutions 2048 data/00050.dat check/references/gauss_seidel/00050.dat

echo
echo "--------------------------------------------------------------------------------"
echo "Red-black Gauss-Seidel solver"
echo "--------------------------------------------------------------------------------"
./heat -t 3 > /dev/null
./check/compare_solutions 2048 data/00050.dat check/references/red_black_gauss_seidel/00050.dat
echo
