from isaacgym import gymapi, gymtorch, gymutil
import torch

from DexManipNet.dexmanipnet_oakinkv2 import DexManipNetOakInkV2
from DexManipNet.dexmanipnet_favor import DexManipNetFAVOR
import argparse

from DexManipNet.dexmanip_sh import DexManipSH_RH
from DexManipNet.dexmanip_sh import DexManipSH_LH
from DexManipNet.dexmanip_bih import DexManipBiH

from termcolor import cprint


if __name__ == "__main__":
    # Example usage
    args = gymutil.parse_arguments(
        description="Visualize DexManipNet Dataset",
        headless=True,
        custom_parameters=[
            {
                "name": "--seq",
                "type": str,
                "default": "",
                "help": "Sequence folder name, e.g. 82fc7@0_bih",
            },
            {
                "name": "--source",
                "type": str,
                "default": "oakinkv2",
                "help": "Dataset source: [oakinkv2 | favor]",
            },
            {
                "name": "--record",
                "action": "store_true",
                "default": False,
                "help": "Record offscreen video to --record_path (headless cluster)",
            },
            {
                "name": "--record_path",
                "type": str,
                "default": "",
                "help": "Output path for recorded video (default: vis_{seq_name}_{side}.mp4)",
            },
        ],
    )

    assert args.seq, "Must provide --seq, e.g. --seq 82fc7@0_bih"
    side = args.seq.rsplit("_", 1)[-1]  # infer side from folder name suffix

    if args.source == "favor":
        data_dir = "data/dexmanipnet/dexmanipnet_favor"
        assert side == "rh", "Only rh is supported for favor"
        dataset = DexManipNetFAVOR(data_dir=data_dir, side=side)
    elif args.source == "oakinkv2":
        data_dir = "data/dexmanipnet/dexmanipnet_oakinkv2"
        dataset = DexManipNetOakInkV2(data_dir=data_dir, side=side)
    else:
        raise ValueError("Invalid source. Choose from [oakinkv2 | favor].")

    assert args.seq in dataset.seq_list, f"{args.seq} not found in sequences directory"
    idx = dataset.seq_list.index(args.seq)
    item = dataset[idx]

    record_path = args.record_path if args.record_path else f"vis_{item['seq_name']}_{side}.mp4"
    args.record_path = record_path

    if side == "rh":
        vis_env = DexManipSH_RH(args, item)
    elif side == "lh":
        vis_env = DexManipSH_LH(args, item)
    elif side == "bih":
        vis_env = DexManipBiH(args, item)
    else:
        raise ValueError(f"Unknown side '{side}' inferred from --seq. Expected rh/lh/bih suffix.")

    cprint(f"seq_name: {item['seq_name']}", "blue")
    if "description" in item:
        cprint(f'description: {item["description"]}', "red")
    cprint(f'primitive: {item["primitive"]}', "green")

    vis_env.play()
