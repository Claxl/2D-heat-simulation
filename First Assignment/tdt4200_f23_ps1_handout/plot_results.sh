help()
{
    echo
    echo "Plot 1D heat"
    echo
    echo "Syntax"
    echo "---------------------------------------------"
    echo "./plot_results.sh [-n|h]                    "
    echo
    echo "Option    Description     Arguments   Default"
    echo "---------------------------------------------"
    echo "n         Grid size       Optional    2048   "
    echo "h         Help            None               "
    echo
    echo "Example"
    echo "---------------------------------------------"
    echo "./plot_solution.sh -n 2048                   "
    echo
}

#-----------------------------------------------------------------
set -e

N=2048

while getopts ":n:h" opt; do
    case $opt in
        n)
            N=$OPTARG;;
        h)
            help
            exit;;
        \?)
            echo "Invalid option"
            help
            exit;;
    esac
done

# Go through every data/#####.dat file
for data in `find data/*.dat`
do
    # Convert the file name into 'image/#####.png'
    image=`echo $data | sed s/data/plots/ | sed s/\.dat/.png/`
    # Give an indicator that something is happening
    echo "Plotting '$image'"

    # Pipe the plotting script into gnuplot, using a background process
    #  so that each plot doesn't have to wait for the others
    cat <<END_OF_SCRIPT | gnuplot - &
set term png
set output "$image"
set xrange [0:${N}]
set yrange [0:1]
plot "$data" binary array=${N} format='%double' with lines
END_OF_SCRIPT
done
