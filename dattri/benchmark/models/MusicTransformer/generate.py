import torch
import torch.nn as nn
import os


from midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi


from utilities.constants import *
from utilities.device import get_device, use_cuda

import pickle

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    for idx in range(len(dataset)):
        # Can be None, an integer index to dataset, or a file path
        if(args.primer_file is None):
            # f = str(random.randrange(len(dataset)))
            f = str(idx)
        else:
            f = args.primer_file

        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())
        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

        model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights))

        # Saving primer first
        # f_path = os.path.join(args.output_dir, f"primer_{idx}.mid")
        # decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

        # GENERATION
        model.eval()

        generate_list = None
        with torch.set_grad_enabled(False):
            for generate_iter in range(args.target_seq_length - args.num_prime):
                print("start generation iter:", generate_iter)
                rand_seq = model.generate(primer[generate_iter:args.num_prime+generate_iter], args.num_prime+1, beam=0)
                if generate_list is None:
                    generate_list = rand_seq[0].cpu().numpy()
                else:
                    generate_list = np.concatenate((generate_list, rand_seq[0][-1:].cpu().numpy()))
                primer = torch.cat((primer, rand_seq[0][-1:]))
                print(generate_list.shape)
                print(primer.shape)

            # if(args.beam > 0):
            #     print("BEAM:", args.beam)
            #     beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            #     f_path = os.path.join(args.output_dir, f"beam_{idx}.mid")
            #     decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
            # else:
            #     print("RAND DIST")
                # rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

                # f_path = os.path.join(args.output_dir, f"rand_{idx}.mid")
                # decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)

            f_path_pickle = os.path.join(args.output_dir, f"rand_{idx}.mid.pickle")
            # print(rand_seq[0].cpu().numpy().shape)  # generate_list
            # with open(f_path_pickle, "wb") as file:
            #     pickle.dump([int(i) for i in list(rand_seq[0].cpu().numpy())], file)
            print(generate_list.shape)  # generate_list
            with open(f_path_pickle, "wb") as file:
                pickle.dump([int(i) for i in list(generate_list)], file)


if __name__ == "__main__":
    main()
