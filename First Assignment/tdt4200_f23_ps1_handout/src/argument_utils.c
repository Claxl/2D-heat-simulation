#include <getopt.h>
#include <stddef.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

#include "../inc/argument_utils.h"


OPTIONS
*parse_args ( int argc, char **argv )
{
    /*
     * Argument parsing: don't change this!
     */

    int_t N = 2048;
    int_t max_iteration = 1000000;
    int_t snapshot_frequency = 10000;
    int solver_type = 1; //jacobi

    static struct option const long_options[] =  {
        {"help",                no_argument,       0, 'h'},
        {"x_size",              required_argument, 0, 'n'},
        {"max_iteration",       required_argument, 0, 'i'},
        {"snapshot_frequency",  required_argument, 0, 'f'},
        {"solver_type",              required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    static char const * short_options = "hn:i:s:t:";
    {
        char *endptr;
        int c;
        int option_index = 0;

        while ( (c = getopt_long( argc, argv, short_options, long_options, &option_index )) != -1 )
        {
            switch (c)
            {
                case 'h':
                    help( argv[0], 0, NULL );
                    exit(0);
                    break;
                case 'n':
                    N = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || N < 0 )
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;
                case 'i':
                    max_iteration = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || max_iteration < 0 )
                    {
                        help( argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 's':
                    snapshot_frequency = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || snapshot_frequency < 0 )
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;
                case 't':
                    solver_type = strtol(optarg, &endptr, 10);
                    if ( endptr == optarg || solver_type < 1 || solver_type > 3)
                    {
                        help( argv[0], c, optarg );
                        return NULL;
                    }
                    break;
                default:
                    abort();
             }
        }
    }

    if ( argc < (optind) )
    {
        printf("argc/optind: %d/%d\n", argc, optind);

        help( argv[0], ' ', "Not enough arugments" );
        return NULL;
    }

  OPTIONS* args_parsed = malloc( sizeof(OPTIONS) );
  args_parsed->N = N;
  args_parsed->max_iteration = max_iteration;
  args_parsed->snapshot_frequency = snapshot_frequency;
  args_parsed->solver_type = solver_type;

  return args_parsed;
}


void
help ( char const *exec, char const opt, char const *optarg )
{
    FILE *out = stdout;

    if ( opt != 0 )
    {
        out = stderr;
        if ( optarg )
        {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        }
        else
        {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }

    fprintf(out, "%s [options]\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options                       Description                 Restriction     Default\n");
    fprintf(out, "  -n, --x_size                size of the x dimension     n>0             2048\n"  );
    fprintf(out, "  -i, --max_iteration         number of iterations        i>0             1000000\n");
    fprintf(out, "  -s, --snapshot_frequency    snapshot frequency          s>0             10000\n"  );
    fprintf(out, "  -t, --solver_type           solver type                 [1,2,3]      1\n"  );

    fprintf(out, "\n");
    fprintf(out, "Example: %s -n 2048 -i 1000000 -s 10000 -t 1\n", exec);
}
