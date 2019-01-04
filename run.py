import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tool', type=str, default=None)
    parser.add_argument('-mode', type=str, default=None)

    parser.add_argument('-net', type=str, default=None)
    parser.add_argument('-com', type=str, default=None)
    parser.add_argument('-label', type=str, default=None)

    parser.add_argument('-k', type=int, default=None)

    parser.add_argument('-cas', type=str, default=None)

    parser.add_argument('-N', type=int, default=None)
    parser.add_argument('-T', type=int, default=None)
    parser.add_argument('-m', type=float, default=None)
    parser.add_argument('-feature', type=int, default=None)
    parser.add_argument('-l', type=int, default=None)
    parser.add_argument('-ite', type=int, default=None)
    parser.add_argument('-batch', type=int, default=None)
    parser.add_argument('-lmd', type=float, default=None)
    parser.add_argument('-lr', type=float, default=None)
    parser.add_argument('-dat', type=str, default=None)
    parser.add_argument('-dat_tra', type=str, default=None)
    parser.add_argument('-dat_tes', type=str, default=None)
    parser.add_argument('-dat_val', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-algorithm', type=str, default=None)



    args = parser.parse_args()

    if args.tool == 'ed':
        if args.mode == 'benchmark':
            os.system('python ./ed/converter_benchmark.py -net ' + args.net + ' -com ' + args.com)
        if args.mode == 'SNAP':
            os.system('python ./ed/converter_SNAP.py -net ' + args.net + ' -label ' + args.label)
        os.system('python ./ed/Encoder-Decoder_tf.py -k ' + str(args.k))

    if args.tool == 'lis':
        if args.mode == 'with_time':
            os.system('python ./lis/converter.py -N ' + str(args.N) + ' -net ' + args.net
                      + ' -cas ' + args.cas + ' -out transformed_dat')
            args.dat = 'transformed_dat'
        os.system('python ./lis/LIS.py -N ' + str(args.N) + ' -feature ' + str(args.feature)
                  + ' -l ' + str(args.l) + ' -ite ' + str(args.ite) + ' -batch ' + str(args.batch)
                  + ' -lmd ' + str(args.l) + ' -lr ' + str(args.lr) + ' -dat ' + args.dat)

    if args.tool == 'rpp':
        if args.mode == 'without_prior':
            os.system('python ./rpp/RPP.py -N ' + str(args.N) + ' -T ' + str(args.T)
                  + ' -m ' + str(args.m) + ' -lr ' + str(args.lr) + ' -dat ' + args.dat)
        if args.mode == 'with_prior':
            os.system('python ./rpp/RPP_prior.py -N ' + str(args.N) + ' -T ' + str(args.T)
                  + ' -m ' + str(args.m) + ' -lr ' + str(args.lr) + ' -dat ' + args.dat
                  + ' -batch ' + str(args.batch) + ' -ite ' + str(args.ite))


    if args.tool == 'dh':
        os.system('python ./dh/gen_shortestpath.py')
        os.system('python ./dh/preprocess.py')
        os.system('python ./dh/run_sparse.py 0.005 0.0005 0.05 0.8')



    if args.tool == 'sg':
        if args.mode == 'UIC':
            if args.algorithm == 'CELF':
                if args.dataset in ['DBLP', 'hep', 'phy']:
                    os.system('python ./sg/StaticGreedyCELF_undirected_UIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
                if args.dataset == 'slashdot':
                    os.system('python ./sg/StaticGreedyCELF_directed_UIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
            if args.algorithm == 'DU':
                if args.dataset in ['DBLP', 'hep', 'phy']:
                    os.system('python ./sg/StaticGreedyDU_undirected_UIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
                if args.dataset == 'slashdot':
                    os.system('python ./sg/StaticGreedyDU_directed_UIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
        if args.mode == 'WIC':
            if args.algorithm == 'CELF':
                if args.dataset in ['DBLP', 'hep', 'phy']:
                    os.system('python ./sg/StaticGreedyCELF_undirected_WIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
                if args.dataset == 'slashdot':
                    os.system('python ./sg/StaticGreedyCELF_directed_WIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
            if args.algorithm == 'DU':
                if args.dataset in ['DBLP', 'hep', 'phy']:
                    os.system('python ./sg/StaticGreedyDU_undirected_WIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))
                if args.dataset == 'slashdot':
                    os.system('python ./sg/StaticGreedyDU_directed_WIC.py' + ' -dat ' + args.dat + ' -k ' + str(args.k))



    if args.tool == 'atten':
        os.system('python ./atten/data_processing_javadata.py' + ' -dat_tra ' + args.dat_tra +' -dat_tes ' + args.dat_tes +' -dat_val ' + args.dat_val)
        os.system('python ./atten/data_import_sever.py' + ' -batch ' + str(args.batch))
        
        








