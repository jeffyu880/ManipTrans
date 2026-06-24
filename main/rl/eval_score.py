from isaacgym import gymapi, gymtorch
import torch
import json
import os
import h5py
from abc import ABC, abstractmethod
import glob
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from argparse import ArgumentParser
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat
from main.dataset.factory import ManipDataFactory


class Eval(ABC):
    def __init__(self, todo_list, dexhand, data_id=None, dataset_type=None):
        self.todo_list = todo_list
        self.dexhand = dexhand
        self.data_id = data_id  # if set, overrides path-based extraction
        # Dataset type for ground-truth demos. If None, auto-detect from data_id
        # (e.g. "m_160009" -> mydataset, "20aed@0" -> oakink2) via the factory.
        if dataset_type is None and data_id is not None:
            dataset_type = ManipDataFactory.dataset_type(data_id)
        self.dataset_type = dataset_type if dataset_type is not None else "oakink2"

    def _get_data_id(self, path, tag):
        if self.data_id is not None:
            return self.data_id
        return path.split(tag)[-1].split("_")[0]

    def get_env_offset(self):
        table_half_height = 0.015
        table_half_width = 0.4
        table_width_offset = 0.2

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        _table_surface_z = table_pos.z + table_half_height
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, _table_surface_z])
        mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device="cuda:0", dtype=torch.float32)
        return mujoco2gym_transf

    # obj traj tsl error
    def diff_t(self, gt, pred):
        # gt: [B, 3]
        # pred: [B, 3]
        res = torch.norm(gt - pred, dim=1)

        return res.mean(0)

    # obj traj rotation error
    def diff_r(self, gt, pred):
        # gt: [B, 3, 3]
        # pred: [B, 4]
        pred = pred[:, [3, 0, 1, 2]]
        pred = quat_to_rotmat(pred)
        res = gt @ pred.transpose(-1, -2)
        res = rotmat_to_aa(res)
        diff_angle = torch.norm(res, dim=1)
        diff_angle = torch.min(diff_angle, 2 * np.pi - diff_angle)
        diff_angle = diff_angle / np.pi * 180
        return diff_angle.mean(0)

    # hand joint position error
    def diff_joint(self, gt, pred):
        # gt: [B, N, 3]
        # pred: [B, N, 3]

        res = torch.norm(gt - pred, dim=2)
        return res.mean()

    # hand fingertip position error
    def diff_tips(self, gt, pred, dexhand):
        # gt: [B, N, 3]
        # pred: [B, N, 3]
        tip_idx = [v[0] for k, v in dexhand.weight_idx.items() if "tip" in k]
        gt_tips = gt[:, tip_idx]
        pred_tips = pred[:, tip_idx]

        res = torch.norm(gt_tips - pred_tips, dim=2)
        return res.mean()

    @abstractmethod
    def eval(self):
        pass


class EvalSH(Eval):

    @abstractmethod
    def __init__(self, todo_list, dexhand, data_id=None, dataset_type=None):
        super().__init__(todo_list, dexhand, data_id=data_id, dataset_type=dataset_type)
        self.side = None

    def eval(self):
        dexhand = DexHandFactory.create_hand(self.dexhand, self.side)
        demo_dataset_oakink = ManipDataFactory.create_data(
            manipdata_type=self.dataset_type,
            side=self.side,
            device="cuda:0",
            mujoco2gym_transf=self.get_env_offset(),
            max_seq_len=1200,
            dexhand=dexhand,
        )
        total_cnt, total_er, total_et, total_ej, total_eft, total_succ_num = 0, 0, 0, 0, 0, 0
        eval_res_list = []

        for path in tqdm(self.todo_list):

            f = h5py.File(path, "r")

            if len(f[f"rollouts/successful"]) == 0:
                cprint(f"No successful rollouts in {path}, skip!", "red")
                continue

            total_succ_num += len(f[f"rollouts/successful"]) / (
                len(f[f"rollouts/successful"]) + len(f[f"rollouts/failed"])
            )

            data_id = self._get_data_id(path, TAG)

            gt = demo_dataset_oakink[data_id]

            succ_et, succ_er, succ_e_j, succ_e_ft = 0, 0, 0, 0

            for succ_idx in f[f"rollouts/successful"]:

                pre_obj_trajectory = torch.tensor(
                    np.array(
                        f[f"rollouts/successful/{succ_idx}/state_manip_obj_{'rh' if self.side == 'right' else 'lh'}"]
                    ),
                    device="cuda:0",
                )

                length = len(pre_obj_trajectory)

                gt_obj_trajectory = gt["obj_trajectory"][0:length]

                et = self.diff_t(
                    gt_obj_trajectory[:, :3, 3],
                    pre_obj_trajectory[:, :3],
                )

                er = self.diff_r(
                    gt_obj_trajectory[:, :3, :3],
                    pre_obj_trajectory[:, 3 : 3 + 4],
                )

                gt_joint = gt["mano_joints"]

                gt_joint = torch.stack(
                    [
                        gt_joint[dexhand.to_hand(j_name)[0]][:length]
                        for j_name in dexhand.body_names
                        if dexhand.to_hand(j_name)[0] != "wrist"
                    ],
                    dim=1,
                )

                gt_joint = torch.cat(
                    [
                        gt["wrist_pos"][:length, None],
                        gt_joint,
                    ],
                    dim=1,
                )

                pre_joint = torch.tensor(
                    np.array(f[f"rollouts/successful/{succ_idx}/joint_state_{'rh' if self.side == 'right' else 'lh'}"]),
                    device="cuda:0",
                ).reshape(length, -1, 13)[:, :, :3]

                e_j = self.diff_joint(gt_joint, pre_joint)
                e_ft = self.diff_tips(gt_joint, pre_joint, dexhand)

                succ_er += er.item()
                succ_et += et.item()
                succ_e_j += e_j.item()
                succ_e_ft += e_ft.item()

            succ_er /= len(f[f"rollouts/successful"])
            succ_et /= len(f[f"rollouts/successful"])
            succ_e_j /= len(f[f"rollouts/successful"])
            succ_e_ft /= len(f[f"rollouts/successful"])

            dump_path = path.replace("rollouts.hdf5", "eval.json")
            dump_path = dump_path.replace("top_rollouts_oakink", "top_rollouts_oakink_score")
            dump_path = dump_path.replace("top_rollouts_favor", "top_rollouts_favor_score")
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)

            item_res = {
                "e_t": succ_et,
                "e_r": succ_er,
                "e_j": succ_e_j,
                "e_ft": succ_e_ft,
                "succ_rate": len(f[f"rollouts/successful"])
                / (len(f[f"rollouts/successful"]) + len(f[f"rollouts/failed"])),
            }
            if "oakink" in dump_path:
                item_res["seq_id"] = os.path.split(gt["data_path"])[-1].split(".")[0]

            json.dump(
                item_res,
                open(dump_path, "w"),
                indent=4,
            )
            eval_res_list.append(item_res)

            total_cnt += 1
            total_er += succ_er
            total_et += succ_et
            total_ej += succ_e_j
            total_eft += succ_e_ft

        if total_cnt == 0:
            cprint("No successful sequences, skip!", "red")
            return eval_res_list

        cprint(f"{self.side} er: {total_er / total_cnt}", "red")
        cprint(f"{self.side} et: {total_et / total_cnt}", "red")
        cprint(f"{self.side} ej: {total_ej / total_cnt}", "red")
        cprint(f"{self.side} eft: {total_eft / total_cnt}", "red")
        cprint(f"{self.side} succ rate: {total_succ_num / total_cnt}", "red")

        return eval_res_list


class EvalRH(EvalSH):
    def __init__(self, todo_list, dexhand, data_id=None, dataset_type=None):
        super().__init__(todo_list, dexhand, data_id=data_id, dataset_type=dataset_type)
        self.side = "right"


class EvalLH(EvalSH):
    def __init__(self, todo_list, dexhand, data_id=None, dataset_type=None):
        super().__init__(todo_list, dexhand, data_id=data_id, dataset_type=dataset_type)
        self.side = "left"


class EvalBiH(Eval):
    def __init__(self, todo_list, dexhand, data_id=None, dataset_type=None):
        super().__init__(todo_list, dexhand, data_id=data_id, dataset_type=dataset_type)

    def eval(self, results_path=None):
        dexhand_rh = DexHandFactory.create_hand(self.dexhand, "right")
        dexhand_lh = DexHandFactory.create_hand(self.dexhand, "left")
        demo_dataset_oakink_rh = ManipDataFactory.create_data(
            manipdata_type=self.dataset_type,
            side="right",
            device="cuda:0",
            mujoco2gym_transf=self.get_env_offset(),
            max_seq_len=1200,
            dexhand=dexhand_rh,
        )
        demo_dataset_oakink_lh = ManipDataFactory.create_data(
            manipdata_type=self.dataset_type,
            side="left",
            device="cuda:0",
            mujoco2gym_transf=self.get_env_offset(),
            max_seq_len=1200,
            dexhand=dexhand_lh,
        )
        total_cnt, total_er, total_et, total_ej, total_eft, total_succ_rate, total_demos, total_rollouts = 0, 0, 0, 0, 0, 0, 0, 0
        eval_res_list = []

        for path in tqdm(self.todo_list):

            f = h5py.File(path, "r")

            if len(f[f"rollouts/successful"]) == 0:
                cprint(f"No successful rollouts in {path}, skip!", "red")
                continue

            total_succ_rate += len(f[f"rollouts/successful"]) / (
                len(f[f"rollouts/successful"]) + len(f[f"rollouts/failed"])
            )

            data_id = self._get_data_id(path, TAG)

            gt_rh = demo_dataset_oakink_rh[data_id]
            gt_lh = demo_dataset_oakink_lh[data_id]

            succ_et, succ_er, succ_e_j, succ_e_ft = 0, 0, 0, 0

            for succ_idx in f[f"rollouts/successful"]:
                
                print(succ_idx)

                pre_obj_trajectory_lh = torch.tensor(
                    np.array(f[f"rollouts/successful/{succ_idx}/state_manip_obj_lh"]), device="cuda:0"
                )
                pre_obj_trajectory_rh = torch.tensor(
                    np.array(f[f"rollouts/successful/{succ_idx}/state_manip_obj_rh"]), device="cuda:0"
                )

                length = len(pre_obj_trajectory_rh)
                assert length == len(pre_obj_trajectory_lh)

                gt_obj_trajectory_rh = gt_rh["obj_trajectory"][0:length]
                gt_obj_trajectory_lh = gt_lh["obj_trajectory"][0:length]

                et_rh = self.diff_t(
                    gt_obj_trajectory_rh[:, :3, 3],
                    pre_obj_trajectory_rh[:, :3],
                )
                et_lh = self.diff_t(
                    gt_obj_trajectory_lh[:, :3, 3],
                    pre_obj_trajectory_lh[:, :3],
                )

                er_rh = self.diff_r(
                    gt_obj_trajectory_rh[:, :3, :3],
                    pre_obj_trajectory_rh[:, 3 : 3 + 4],
                )
                er_lh = self.diff_r(
                    gt_obj_trajectory_lh[:, :3, :3],
                    pre_obj_trajectory_lh[:, 3 : 3 + 4],
                )

                gt_joint_rh = gt_rh["mano_joints"]
                gt_joint_lh = gt_lh["mano_joints"]

                gt_joint_rh = torch.stack(
                    [
                        gt_joint_rh[dexhand_rh.to_hand(j_name)[0]][:length]
                        for j_name in dexhand_rh.body_names
                        if dexhand_rh.to_hand(j_name)[0] != "wrist"
                    ],
                    dim=1,
                )
                gt_joint_lh = torch.stack(
                    [
                        gt_joint_lh[dexhand_lh.to_hand(j_name)[0]][:length]
                        for j_name in dexhand_lh.body_names
                        if dexhand_lh.to_hand(j_name)[0] != "wrist"
                    ],
                    dim=1,
                )

                gt_joint_rh = torch.cat(
                    [
                        gt_rh["wrist_pos"][:length, None],
                        gt_joint_rh,
                    ],
                    dim=1,
                )
                gt_joint_lh = torch.cat(
                    [
                        gt_lh["wrist_pos"][:length, None],
                        gt_joint_lh,
                    ],
                    dim=1,
                )

                pre_joint_lh = torch.tensor(
                    np.array(f[f"rollouts/successful/{succ_idx}/joint_state_lh"]), device="cuda:0"
                ).reshape(length, -1, 13)[:, :, :3]
                pre_joint_rh = torch.tensor(
                    np.array(f[f"rollouts/successful/{succ_idx}/joint_state_rh"]), device="cuda:0"
                ).reshape(length, -1, 13)[:, :, :3]

                e_j_rh = self.diff_joint(gt_joint_rh, pre_joint_rh)
                e_j_lh = self.diff_joint(gt_joint_lh, pre_joint_lh)
                e_ft_rh = self.diff_tips(gt_joint_rh, pre_joint_rh, dexhand_rh)
                e_ft_lh = self.diff_tips(gt_joint_lh, pre_joint_lh, dexhand_lh)

                succ_er += (er_rh.item() + er_lh.item()) / 2
                succ_et += (et_rh.item() + et_lh.item()) / 2
                succ_e_j += (e_j_rh.item() + e_j_lh.item()) / 2
                succ_e_ft += (e_ft_rh.item() + e_ft_lh.item()) / 2

                cprint(f"  [{succ_idx}] er: {(er_rh.item() + er_lh.item()) / 2:.3f}° \
                        et: {(et_rh.item() + et_lh.item()) / 2 * 100:.3f}cm  \
                        ej: {(e_j_rh.item() + e_j_lh.item()) / 2 * 100:.3f}cm  \
                        eft: {(e_ft_rh.item() + e_ft_lh.item()) / 2 * 100:.3f}cm", "yellow")

            succ_er /= len(f[f"rollouts/successful"])
            succ_et /= len(f[f"rollouts/successful"])
            succ_e_j /= len(f[f"rollouts/successful"])
            succ_e_ft /= len(f[f"rollouts/successful"])

            dump_path = path.replace("rollouts.hdf5", "eval.json")
            dump_path = dump_path.replace("top_rollouts_oakink", "top_rollouts_oakink_score")
            dump_path = dump_path.replace("top_rollouts_favor", "top_rollouts_favor_score")
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)

            item_res = {
                "e_t": succ_et,
                "e_r": succ_er,
                "e_j": succ_e_j,
                "e_ft": succ_e_ft,
                "succ_rate": len(f[f"rollouts/successful"])
                / (len(f[f"rollouts/successful"]) + len(f[f"rollouts/failed"])),
                # "e_t_rh": et_rh.item(),
                # "e_r_rh": er_rh.item(),
                # "e_j_rh": e_j_rh.item(),
                # "e_ft_rh": e_ft_rh.item(),
                # "e_t_lh": et_lh.item(),
                # "e_r_lh": er_lh.item(),
                # "e_j_lh": e_j_lh.item(),
                # "e_ft_lh": e_ft_lh.item(),
            }
            if "oakink" in dump_path:
                item_res["seq_id"] = os.path.split(gt_rh["data_path"])[-1].split(".")[0]

            json.dump(
                item_res,
                open(dump_path, "w"),
                indent=4,
            )
            eval_res_list.append(item_res)

            n_succ = len(f[f"rollouts/successful"])
            n_fail = len(f[f"rollouts/failed"])
            total_cnt += n_succ
            total_rollouts += n_succ + n_fail
            total_er += succ_er * n_succ
            total_et += succ_et * n_succ
            total_ej += succ_e_j * n_succ
            total_eft += succ_e_ft * n_succ
            total_demos += 1

        if total_cnt == 0:
            cprint("No successful sequences, skip!", "red")
            return eval_res_list

        summary_lines = [
            "Average performance across all demos",
            f"Total rollouts: {total_rollouts} ({total_cnt} success, {total_rollouts - total_cnt} fail)",
            f"bih succ rate: {total_succ_rate / total_demos:.4f}",
            f"bih er: {total_er / total_cnt} deg",
            f"bih et: {total_et / total_cnt * 100} cm",
            f"bih ej: {total_ej / total_cnt * 100} cm",
            f"bih eft: {total_eft / total_cnt * 100} cm",
            str(eval_res_list),
        ]
        for line in summary_lines:
            cprint(line, "red") if line.startswith("bih") else print(line)

        if results_path is not None:
            with open(results_path, "w") as rf:
                rf.write("\n".join(summary_lines) + "\n")
            print(f"Results written to {results_path}")

        return eval_res_list


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tag", type=str, default="dump_baseline_")
    args.add_argument("--dexhand", type=str, default="inspire")
    args.add_argument("--side", type=str, default="bih", choices=["bih", "rh", "lh"])
    args.add_argument("--path", type=str, default=None, help="Direct path to rollouts.hdf5")
    args.add_argument("--data_id", type=str, default=None, help="Demo data ID (required when --path is set)")
    args.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="GT demo dataset type (e.g. oakink2, mydataset). If unset, auto-detected from --data_id.",
    )
    args = args.parse_args()

    TAG = args.tag
    DEXHAND = args.dexhand

    data_id = args.data_id
    dataset_type = args.dataset_type

    if args.path is not None:
        if data_id is None:
            raise ValueError("--data_id is required when --path is specified")
        todo_list_bih = [args.path] if args.side == "bih" else []
        todo_list_rh  = [args.path] if args.side == "rh"  else []
        todo_list_lh  = [args.path] if args.side == "lh"  else []
    else:
        todo_list_bih = glob.glob(f"dumps/{TAG}*_{DEXHAND}_bih*/rollouts.hdf5")
        todo_list_rh  = glob.glob(f"dumps/{TAG}*_{DEXHAND}_rh*/rollouts.hdf5")
        todo_list_lh  = glob.glob(f"dumps/{TAG}*_{DEXHAND}_lh*/rollouts.hdf5")

    print(f"Evaluating {todo_list_bih} bih sequences")
    eval_bih = EvalBiH(todo_list_bih, args.dexhand, data_id=data_id, dataset_type=dataset_type)
    results_path = os.path.join(os.path.dirname(args.path), "results.txt") if args.path else None
    bih_res = eval_bih.eval(results_path=results_path)
    print(bih_res)
    # print(f"Evaluating {todo_list_rh} rh sequences")
    # eval_rh = EvalRH(todo_list_rh, args.dexhand, data_id=data_id)
    # rh_res = eval_rh.eval()
    # print(rh_res)
    # print(f"Evaluating {todo_list_lh} lh sequences")
    # eval_lh = EvalLH(todo_list_lh, args.dexhand, data_id=data_id)
    # lh_res = eval_lh.eval()

    # * Report all results: ## THIS ASSUMES ONE ROLLOUT PER .hdf5 file
    # * 1. Number of successful sequences(at least one successful rollout in 512 rollouts)
    # * 1. Must equal to the total number of tested sequences
    # cprint("================ Overall Results ================", "cyan")
    # cprint(f"Number of successful sequences: {len(bih_res) + len(rh_res) + len(lh_res)}", "cyan")
    # # * 2. Average success rate
    # sh_succ_rate = (
    #     sum([r["succ_rate"] for r in rh_res + lh_res]) / (len(rh_res) + len(lh_res))
    #     if (len(rh_res) + len(lh_res)) > 0
    #     else 0
    # )
    # bih_succ_rate = sum([r["succ_rate"] for r in bih_res]) / len(bih_res) if len(bih_res) > 0 else 0
    # cprint(f"Average success rate: Single hand: {sh_succ_rate: 0.3f}, Bi-hand: {bih_succ_rate: 0.3f}", "cyan")
    # # * 3. Average et, er, ej, eft (only for successful sequences)
    # overall_et = sum([r["e_t"] for r in rh_res + lh_res + bih_res]) / (len(rh_res) + len(lh_res) + len(bih_res))
    # overall_er = sum([r["e_r"] for r in rh_res + lh_res + bih_res]) / (len(rh_res) + len(lh_res) + len(bih_res))
    # overall_ej = sum([r["e_j"] for r in rh_res + lh_res + bih_res]) / (len(rh_res) + len(lh_res) + len(bih_res))
    # overall_eft = sum([r["e_ft"] for r in rh_res + lh_res + bih_res]) / (len(rh_res) + len(lh_res) + len(bih_res))
    # cprint(f"Average et (cm): {overall_et * 100: 0.3f}", "cyan")
    # cprint(f"Average er (degree): {overall_er: 0.3f}", "cyan")
    # cprint(f"Average ej (cm): {overall_ej * 100: 0.3f}", "cyan")
    # cprint(f"Average eft (cm): {overall_eft * 100: 0.3f}", "cyan")
